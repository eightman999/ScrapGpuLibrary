// Unit tests of the quantization helpers: hand-built blocks with known values,
// quantize -> dequantize round trip error bounds, and the F16 conversion bits.

#include "scrapgpu/reference.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// CHECK stays active in Release builds (assert() is compiled out with NDEBUG)
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);   \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

namespace {

void test_hand_built_q4_0() {
    // one block: d = 1.0 (0x3C00), qs[j] = j: low nibbles j - 8, high nibbles -8
    uint8_t blk[18] = {};
    blk[0] = 0x00;
    blk[1] = 0x3C;
    for (int j = 0; j < 16; ++j) {
        blk[2 + j] = static_cast<uint8_t>(j);
    }
    float y[32] = {};
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, blk, y, 32) == SCRAPGPU_STATUS_OK);
    for (int j = 0; j < 16; ++j) {
        CHECK(y[j] == static_cast<float>(j - 8));
        CHECK(y[16 + j] == -8.0f);
    }
}

void test_hand_built_q8_0() {
    // one block: d = 0.5 (0x3800), qs[j] = j - 16
    uint8_t blk[34] = {};
    blk[0] = 0x00;
    blk[1] = 0x38;
    for (int j = 0; j < 32; ++j) {
        blk[2 + j] = static_cast<uint8_t>(j - 16);
    }
    float y[32] = {};
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q8_0, blk, y, 32) == SCRAPGPU_STATUS_OK);
    for (int j = 0; j < 32; ++j) {
        CHECK(y[j] == static_cast<float>(j - 16) * 0.5f);
    }
}

void test_hand_built_q4_k() {
    // d = 1.0, dmin = 0; sub-block scales 32 (6 bits: scales[0..3] = 32, mins 0),
    // sub-blocks 4..7 have scale 0; both nibbles of every byte carry the same value, so
    // y[j] = 32 * (j mod 16) in the first half of the super-block
    uint8_t blk[144] = {};
    blk[0] = 0x00;
    blk[1] = 0x3C; // d = 1.0
    blk[2] = 0x00;
    blk[3] = 0x00; // dmin = 0.0
    for (int j = 0; j < 4; ++j) {
        blk[4 + j] = 32; // ls = 32
    }
    for (int l = 0; l < 128; ++l) {
        blk[16 + l] = static_cast<uint8_t>((l & 0x0F) | ((l & 0x0F) << 4));
    }
    float y[256] = {};
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_K, blk, y, 256) == SCRAPGPU_STATUS_OK);
    for (int j = 0; j < 128; ++j) {
        CHECK(y[j] == 32.0f * static_cast<float>(j & 0x0F));
    }
    for (int j = 128; j < 256; ++j) {
        CHECK(y[j] == 0.0f); // sub-blocks 4..7: scale 0, min 0
    }
}

void test_hand_built_q6_k() {
    // d = 1.0 at offset 208; all 16 sub-block scales 8; qh = 0; the quants are packed so
    // that elements n+l / n+l+32 come from the low nibbles and n+l+64 / n+l+96 from the
    // high nibbles (see ggml's dequantize_row_q6_K)
    uint8_t blk[210] = {};
    blk[208] = 0x00;
    blk[209] = 0x3C; // d = 1.0
    for (int j = 0; j < 16; ++j) {
        blk[192 + j] = 8;
    }
    for (int half = 0; half < 2; ++half) {
        for (int l = 0; l < 64; ++l) {
            const int i = half * 64 + l;
            blk[i] = static_cast<uint8_t>((i % 16) | (((i + 3) % 16) << 4));
        }
    }
    float y[256] = {};
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q6_K, blk, y, 256) == SCRAPGPU_STATUS_OK);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            CHECK(y[n + l] == 8.0f * static_cast<float>((l % 16) - 32));
            CHECK(y[n + l + 32] == 8.0f * static_cast<float>(((l + 32) % 16) - 32));
            CHECK(y[n + l + 64] == 8.0f * static_cast<float>(((l + 3) % 16) - 32));
            CHECK(y[n + l + 96] == 8.0f * static_cast<float>(((l + 35) % 16) - 32));
        }
    }
}

float lcg_next(uint32_t & state) {
    state = state * 1664525u + 1013904223u;
    return (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 2.0f; // [-1, 1)
}

void round_trip(scrapgpu_dtype dtype, int64_t n, double bound_factor) {
    const size_t type_size = scrapgpu_dtype_type_size(dtype);
    const size_t block = scrapgpu_dtype_block_size(dtype);
    uint32_t state = 12345u + static_cast<uint32_t>(dtype);
    std::vector<float> x(static_cast<size_t>(n));
    float amax = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        x[static_cast<size_t>(i)] = lcg_next(state) * 3.0f;
        amax = std::max(amax, std::fabs(x[static_cast<size_t>(i)]));
    }
    // add a block with an all-negative range so the deducted min is exercised (Q4_K)
    if (scrapgpu_dtype_block_size(dtype) == 256) {
        for (int i = 0; i < 256; ++i) {
            x[static_cast<size_t>(i)] = -1.0f - 0.5f * std::fabs(lcg_next(state));
        }
        amax = std::max(amax, 1.5f);
    }
    std::vector<uint8_t> q(static_cast<size_t>(n) / block * type_size);
    CHECK(scrapgpu_ref_quantize(dtype, x.data(), q.data(), n) == SCRAPGPU_STATUS_OK);
    std::vector<float> y(static_cast<size_t>(n));
    CHECK(scrapgpu_ref_dequantize(dtype, q.data(), y.data(), n) == SCRAPGPU_STATUS_OK);
    double max_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        max_err = std::max(max_err, std::fabs(static_cast<double>(x[static_cast<size_t>(i)]) - static_cast<double>(y[static_cast<size_t>(i)])));
    }
    std::fprintf(stderr, "dtype %d round trip: n = %lld, amax = %g, max err = %g (%.3f%% of amax)\n",
                 static_cast<int>(dtype), static_cast<long long>(n), amax, max_err, 100.0 * max_err / amax);
    CHECK(max_err <= bound_factor * amax);
}

void test_round_trips() {
    round_trip(SCRAPGPU_DTYPE_Q4_0, 32 * 7, 1.0 / 7.0);    // ggml q4_0: the far end of the signed max is off by up to amax/8
    round_trip(SCRAPGPU_DTYPE_Q8_0, 32 * 5, 1.0 / 200.0);  // ~ step/2 of amax/127 plus fp16 slack
    round_trip(SCRAPGPU_DTYPE_Q4_K, 256 * 3, 0.10);        // ggml quantize_row_q4_K_ref round trip (~6% of amax)
    round_trip(SCRAPGPU_DTYPE_Q6_K, 256 * 3, 0.02);        // ggml quantize_row_q6_K_ref (+ negative super-block in this harness)
    // all-zero block
    {
        std::vector<float> z(32, 0.0f);
        std::vector<uint8_t> q(18);
        std::vector<float> y(32, 1.0f);
        CHECK(scrapgpu_ref_quantize(SCRAPGPU_DTYPE_Q4_0, z.data(), q.data(), 32) == SCRAPGPU_STATUS_OK);
        CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, q.data(), y.data(), 32) == SCRAPGPU_STATUS_OK);
        for (int i = 0; i < 32; ++i) {
            CHECK(y[i] == 0.0f);
        }
    }
}

void test_argument_errors() {
    float y[4] = {};
    uint8_t blk[64] = {};
    float x[4] = {};
    // not a multiple of the block size
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, blk, y, 31) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(scrapgpu_ref_quantize(SCRAPGPU_DTYPE_Q8_0, x, blk, 33) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    // non-quantized dtypes
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_F32, blk, y, 4) == SCRAPGPU_STATUS_UNSUPPORTED);
    CHECK(scrapgpu_ref_quantize(SCRAPGPU_DTYPE_F32, x, blk, 4) == SCRAPGPU_STATUS_UNSUPPORTED);
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_F16, blk, y, 4) == SCRAPGPU_STATUS_UNSUPPORTED);
    // null pointers / bad counts
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, nullptr, y, 32) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(scrapgpu_ref_quantize(SCRAPGPU_DTYPE_Q4_0, nullptr, blk, 32) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, blk, nullptr, 32) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    CHECK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, blk, y, 0) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
}

// F16 conversion bits through a public path: CPY F32 -> F16 writes the raw halves
void test_f16_bits() {
    const float in[] = {1.0f, -2.0f, 0.5f, 65504.0f, 65520.0f, 65512.0f, -0.0f,
                        5.960464477539063e-08f, 2.9802322387695312e-08f, 8.940696716308594e-08f};
    const uint16_t expect[] = {0x3C00, 0xC000, 0x3800, 0x7BFF, 0x7C00, 0x7BFF, 0x8000,
                               0x0001, 0x0000, 0x0002};
    const int n = static_cast<int>(sizeof(in) / sizeof(in[0]));

    scrapgpu_tensor_desc s{};
    s.struct_size = sizeof(s);
    s.dtype = SCRAPGPU_DTYPE_F32;
    s.ne[0] = n;
    s.nb[0] = 4;
    s.nb[1] = 4 * static_cast<uint64_t>(n);
    s.nb[2] = s.nb[1];
    s.nb[3] = s.nb[1];
    s.layout = SCRAPGPU_LAYOUT_NATURAL;
    scrapgpu_tensor_desc d = s;
    d.dtype = SCRAPGPU_DTYPE_F16;
    d.nb[0] = 2;
    d.nb[1] = 2 * static_cast<uint64_t>(n);
    d.nb[2] = d.nb[1];
    d.nb[3] = d.nb[1];

    scrapgpu_cpy_params p{sizeof(p), 0};
    scrapgpu_op_desc op{};
    op.struct_size = sizeof(op);
    op.op = SCRAPGPU_OP_CPY;
    op.inputs = &s;
    op.input_count = 1;
    op.outputs = &d;
    op.output_count = 1;
    op.params = &p;
    op.params_size = sizeof(p);

    uint16_t out[16] = {};
    void * in_ptr = const_cast<float *>(in);
    void * out_ptr = out;
    CHECK(scrapgpu_ref_execute(&op, &in_ptr, &out_ptr) == SCRAPGPU_STATUS_OK);
    for (int i = 0; i < n; ++i) {
        if (out[i] != expect[i]) {
            std::fprintf(stderr, "f16 bits %d: got 0x%04x want 0x%04x\n", i, out[i], expect[i]);
            CHECK(false);
        }
    }
}

} // namespace

int main() {
    test_hand_built_q4_0();
    test_hand_built_q8_0();
    test_hand_built_q4_k();
    test_hand_built_q6_k();
    test_round_trips();
    test_argument_errors();
    test_f16_bits();
    return 0;
}
