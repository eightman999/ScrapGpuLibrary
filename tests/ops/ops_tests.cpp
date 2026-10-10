// GPU tests of the ops through the public C API only: every op the selected backend reports as
// supported is run on the device and compared against the CPU reference (scrapgpu_ref_execute).
// The backend is selected with SCRAPGPU_TEST_BACKEND=opencl11|vulkan (default opencl11).
// Exit code 77 (ctest SKIP_RETURN_CODE) when no device of the backend exists.

#include "scrapgpu/api.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/op.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/status.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#if defined(SCRAPGPU_OPS_TEST_WITH_OPENCL11)
#include "scrapgpu/opencl11.h"
#endif
#if defined(SCRAPGPU_OPS_TEST_WITH_VULKAN)
#include "scrapgpu/vulkan.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CHECK(cond)                                                                          \
    do {                                                                                     \
        if (!(cond)) {                                                                       \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);    \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

#define CHECK_OK(status)                                                                     \
    do {                                                                                     \
        const scrapgpu_status st_ = (status);                                                \
        if (st_ != SCRAPGPU_STATUS_OK) {                                                     \
            std::fprintf(stderr, "%s:%d: unexpected status %s: %s\n",                        \
                         __FILE__, __LINE__, scrapgpu_status_string(st_), scrapgpu_last_error()); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

namespace {

std::mt19937 g_rng(12345);

float frand() {
    return std::uniform_real_distribution<float>(-1.0f, 1.0f)(g_rng);
}

float frand_pos() {
    return std::uniform_real_distribution<float>(0.05f, 2.0f)(g_rng);
}

std::vector<float> frand_vec(size_t n, bool positive = false) {
    std::vector<float> v(n);
    for (auto & x : v) {
        x = positive ? frand_pos() : frand();
    }
    return v;
}

// ---- fp16 (round to nearest even; test values are never denormal range) ----------------------

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    const uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        return (uint16_t) sign;
    }
    if (e >= 31) {
        return (uint16_t) (sign | 0x7c00u);
    }
    uint32_t h = sign | ((uint32_t) e << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x3fffu) == 0x3000u)) {
        ++h;
    }
    return (uint16_t) h;
}

float f16_to_f32(uint16_t h) {
    const bool neg = (h & 0x8000u) != 0;
    const int32_t e = (h >> 10) & 0x1f;
    const uint32_t m = h & 0x3ffu;
    float f;
    if (e == 0) {
        f = (float) m * 5.9604645e-8f; // denormal: m * 2^-24 (exact)
    } else if (e == 31) {
        f = m != 0 ? std::nanf("") : INFINITY;
    } else {
        const uint32_t x = (uint32_t) (e - 15 + 127) << 23 | (m << 13);
        std::memcpy(&f, &x, 4);
    }
    return neg ? -f : f;
}

// ---- tensor descriptors ---------------------------------------------------------------------

scrapgpu_tensor_desc tdesc(scrapgpu_dtype dt, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = dt;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = n2;
    t.ne[3] = n3;
    t.nb[0] = scrapgpu_dtype_type_size(dt);
    t.nb[1] = scrapgpu_tensor_row_size(dt, n0);
    t.nb[2] = t.nb[1] * (uint64_t) (n1 > 0 ? n1 : 1);
    t.nb[3] = t.nb[2] * (uint64_t) (n2 > 0 ? n2 : 1);
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    return t;
}

// bytes a tensor image spans (the offset of the last byte + 1), for arbitrary strides
uint64_t tspan(const scrapgpu_tensor_desc & t) {
    const size_t block = scrapgpu_dtype_block_size(t.dtype);
    uint64_t n = block > 1 ? (uint64_t) t.ne[0] * t.nb[0] / block
                         : scrapgpu_dtype_type_size(t.dtype) + (uint64_t) (t.ne[0] - 1) * t.nb[0];
    for (int i = 1; i < 4; ++i) {
        if (t.ne[i] > 1) {
            n += (uint64_t) (t.ne[i] - 1) * t.nb[i];
        }
    }
    return n;
}

int64_t tn(const scrapgpu_tensor_desc & t) {
    return t.ne[0] * t.ne[1] * t.ne[2] * t.ne[3];
}

// ---- device state ----------------------------------------------------------------------------

scrapgpu_device_t * g_dev = nullptr;
scrapgpu_stream_t * g_stream = nullptr;
std::vector<scrapgpu_buffer_t *> g_buffers;
// the layout of quantized weight tensors on the selected backend: R32 for opencl11 (the repacked
// layout), the natural GGUF block layout for vulkan
scrapgpu_layout g_weight_layout = SCRAPGPU_LAYOUT_R32;

scrapgpu_buffer_t * g_alloc(size_t bytes) {
    scrapgpu_buffer_t * buf = nullptr;
    CHECK_OK(scrapgpu_buffer_alloc(g_dev, bytes, &buf));
    g_buffers.push_back(buf);
    return buf;
}

// a test tensor: the host image in the natural layout + a device buffer + the descs of the GPU
// side (buffer set, layout as passed) and the reference side (always NATURAL, no buffer)
struct TT {
    scrapgpu_tensor_desc gdesc = {};
    scrapgpu_tensor_desc rdesc = {};
    std::vector<uint8_t> host;
    scrapgpu_buffer_t * buf = nullptr;
    bool prefilled = false; // the output image has content the GPU must see first (SET_ROWS dst)
};

TT tmake(const scrapgpu_tensor_desc & d, scrapgpu_layout layout = SCRAPGPU_LAYOUT_NATURAL) {
    TT t;
    t.gdesc = d;
    t.gdesc.layout = layout;
    t.gdesc.offset = 0;
    t.rdesc = d;
    t.rdesc.layout = SCRAPGPU_LAYOUT_NATURAL;
    t.rdesc.buffer = nullptr;
    t.rdesc.offset = 0;
    t.host.resize((size_t) tspan(d));
    t.buf = g_alloc((size_t) tspan(d));
    t.gdesc.buffer = t.buf;
    return t;
}

// like, but on the host only (the reference side of a fused op)
TT tmake_ref(const scrapgpu_tensor_desc & d) {
    TT t;
    t.rdesc = d;
    t.rdesc.layout = SCRAPGPU_LAYOUT_NATURAL;
    t.rdesc.buffer = nullptr;
    t.rdesc.offset = 0;
    t.gdesc = t.rdesc;
    t.host.assign((size_t) tspan(d), 0);
    return t;
}

// write the logical elements of the image through the strides
void tfill(TT & t, const std::vector<float> & v) {
    CHECK((size_t) tn(t.rdesc) == v.size());
    const bool f16 = t.rdesc.dtype == SCRAPGPU_DTYPE_F16;
    CHECK(t.rdesc.dtype == SCRAPGPU_DTYPE_F32 || f16);
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t.rdesc.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t.rdesc.ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t.rdesc.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t.rdesc.ne[0]; ++i0) {
                    uint8_t * q = t.host.data() + t.rdesc.offset + i0 * t.rdesc.nb[0] +
                                  i1 * t.rdesc.nb[1] + i2 * t.rdesc.nb[2] + i3 * t.rdesc.nb[3];
                    if (f16) {
                        const uint16_t h = f32_to_f16(v[k++]);
                        std::memcpy(q, &h, 2);
                    } else {
                        std::memcpy(q, &v[k++], 4);
                    }
                }
}

void tfill_i32(TT & t, const std::vector<int32_t> & v) {
    CHECK((size_t) tn(t.rdesc) == v.size());
    CHECK(t.rdesc.dtype == SCRAPGPU_DTYPE_I32);
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t.rdesc.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t.rdesc.ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t.rdesc.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t.rdesc.ne[0]; ++i0) {
                    uint8_t * q = t.host.data() + t.rdesc.offset + i0 * t.rdesc.nb[0] +
                                  i1 * t.rdesc.nb[1] + i2 * t.rdesc.nb[2] + i3 * t.rdesc.nb[3];
                    std::memcpy(q, &v[k++], 4);
                }
}

void tfill_i64(TT & t, const std::vector<int64_t> & v) {
    CHECK((size_t) tn(t.rdesc) == v.size());
    CHECK(t.rdesc.dtype == SCRAPGPU_DTYPE_I64);
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t.rdesc.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t.rdesc.ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t.rdesc.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t.rdesc.ne[0]; ++i0) {
                    uint8_t * q = t.host.data() + t.rdesc.offset + i0 * t.rdesc.nb[0] +
                                  i1 * t.rdesc.nb[1] + i2 * t.rdesc.nb[2] + i3 * t.rdesc.nb[3];
                    std::memcpy(q, &v[k++], 8);
                }
}

// quantize a float image into the natural quant bytes of t (contiguous rows)
void tquantize(TT & t, const std::vector<float> & f) {
    CHECK((size_t) tn(t.rdesc) == f.size());
    CHECK_OK(scrapgpu_ref_quantize(t.rdesc.dtype, f.data(), t.host.data(), tn(t.rdesc)));
}

void tupload(TT & t) {
    if (t.gdesc.layout == SCRAPGPU_LAYOUT_R32 || scrapgpu_tensor_is_contiguous(&t.gdesc)) {
        CHECK_OK(scrapgpu_tensor_upload(g_stream, &t.gdesc, t.host.data(), t.host.size()));
    } else {
        CHECK_OK(scrapgpu_buffer_write(g_stream, t.buf, 0, t.host.data(), t.host.size(), 1));
    }
}

void tdownload(TT & t) {
    if (scrapgpu_tensor_is_contiguous(&t.gdesc)) {
        CHECK_OK(scrapgpu_tensor_download(g_stream, &t.gdesc, t.host.data(), t.host.size()));
    } else {
        CHECK_OK(scrapgpu_buffer_read(g_stream, t.buf, 0, t.host.data(), t.host.size(), 1));
    }
}

// the logical elements of the image as floats (F32 / F16 / I32 / I64)
std::vector<float> tfloats(const TT & t) {
    std::vector<float> v((size_t) tn(t.rdesc));
    size_t k = 0;
    for (int64_t i3 = 0; i3 < t.rdesc.ne[3]; ++i3)
        for (int64_t i2 = 0; i2 < t.rdesc.ne[2]; ++i2)
            for (int64_t i1 = 0; i1 < t.rdesc.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < t.rdesc.ne[0]; ++i0) {
                    const uint8_t * q = t.host.data() + t.rdesc.offset + i0 * t.rdesc.nb[0] +
                                        i1 * t.rdesc.nb[1] + i2 * t.rdesc.nb[2] + i3 * t.rdesc.nb[3];
                    float x = 0.0f;
                    switch (t.rdesc.dtype) {
                        case SCRAPGPU_DTYPE_F32: std::memcpy(&x, q, 4); break;
                        case SCRAPGPU_DTYPE_F16: { uint16_t h; std::memcpy(&h, q, 2); x = f16_to_f32(h); break; }
                        case SCRAPGPU_DTYPE_I32: { int32_t i; std::memcpy(&i, q, 4); x = (float) i; break; }
                        case SCRAPGPU_DTYPE_I64: { int64_t i; std::memcpy(&i, q, 8); x = (float) i; break; }
                        default: CHECK(false);
                    }
                    v[k++] = x;
                }
    return v;
}

double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    CHECK(a.size() == b.size());
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return den > 0.0 ? num / den : num;
}

bool exact_equal(const std::vector<float> & a, const std::vector<float> & b) {
    CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::fprintf(stderr, "element %zu: %g != %g\n", i, (double) a[i], (double) b[i]);
            return false;
        }
    }
    return true;
}

// ---- op runner -------------------------------------------------------------------------------

int g_skipped = 0;
int g_run = 0;

// run a ref op on TTs (natural descs, host images)
void ref_exec(scrapgpu_op op, std::vector<TT> & ins, std::vector<TT> & outs, const void * params, size_t params_size) {
    std::vector<scrapgpu_tensor_desc> in_d(ins.size());
    std::vector<scrapgpu_tensor_desc> out_d(outs.size());
    std::vector<void *> in_p(ins.size()), out_p(outs.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        in_d[i] = ins[i].rdesc;
        in_p[i] = ins[i].host.data();
    }
    for (size_t i = 0; i < outs.size(); ++i) {
        out_d[i] = outs[i].rdesc;
        out_p[i] = outs[i].host.data();
    }
    scrapgpu_op_desc d{};
    d.struct_size = sizeof(d);
    d.op = op;
    d.inputs = in_d.data();
    d.input_count = in_d.size();
    d.outputs = out_d.data();
    d.output_count = out_d.size();
    d.params = params;
    d.params_size = params_size;
    CHECK_OK(scrapgpu_ref_execute(&d, in_p.data(), out_p.data()));
}

// run one op on the GPU and compare each output with `ref` (the floats the reference produced)
void run_one(const char * name, scrapgpu_op op, std::vector<TT> & ins, std::vector<TT> & outs,
             const void * params, size_t params_size, const std::vector<std::vector<float>> & ref,
             double tol, bool exact) {
    std::vector<scrapgpu_tensor_desc> in_d(ins.size()), out_d(outs.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        in_d[i] = ins[i].gdesc;
    }
    for (size_t i = 0; i < outs.size(); ++i) {
        out_d[i] = outs[i].gdesc;
    }
    scrapgpu_op_desc d{};
    d.struct_size = sizeof(d);
    d.op = op;
    d.inputs = in_d.data();
    d.input_count = in_d.size();
    d.outputs = out_d.data();
    d.output_count = out_d.size();
    d.params = params;
    d.params_size = params_size;

    const char * reason = nullptr;
    if (scrapgpu_op_supported(g_dev, &d, &reason) != SCRAPGPU_STATUS_OK) {
        std::printf("SKIP  %-24s %s\n", name, reason != nullptr ? reason : "");
        ++g_skipped;
        return;
    }

    for (auto & t : ins) {
        tupload(t);
    }
    for (auto & t : outs) {
        if (t.prefilled) {
            tupload(t);
        }
    }
    CHECK_OK(scrapgpu_op_execute(g_stream, &d));
    CHECK_OK(scrapgpu_stream_sync(g_stream));
    for (auto & t : outs) {
        tdownload(t);
    }

    for (size_t i = 0; i < outs.size(); ++i) {
        const std::vector<float> got = tfloats(outs[i]);
        if (exact) {
            if (exact_equal(got, ref[i])) {
                std::printf("PASS  %-24s out[%zu] exact (%zu elements)\n", name, i, got.size());
            } else {
                std::printf("FAIL  %-24s out[%zu] mismatch\n", name, i);
                std::exit(1);
            }
        } else {
            const double e = nmse(got, ref[i]);
            const bool ok = e <= tol;
            std::printf("%s  %-24s out[%zu] NMSE %.3e (tol %.0e)\n", ok ? "PASS" : "FAIL", name, i, e, tol);
            if (!ok) {
                std::exit(1);
            }
        }
    }
    ++g_run;
}

// reference + GPU run of one op (the reference runs on copies of the output images)
void test_op(const char * name, scrapgpu_op op, std::vector<TT> ins, std::vector<TT> outs,
             const void * params, size_t params_size, double tol = 1e-7, bool exact = false) {
    std::vector<TT> ref_outs = outs;
    ref_exec(op, ins, ref_outs, params, params_size);
    std::vector<std::vector<float>> ref(ref_outs.size());
    for (size_t i = 0; i < ref_outs.size(); ++i) {
        ref[i] = tfloats(ref_outs[i]);
    }
    run_one(name, op, ins, outs, params, params_size, ref, tol, exact);
}

// ----------------------------------------------------------------------------------------------
// the cases
// ----------------------------------------------------------------------------------------------

void test_binary() {
    for (scrapgpu_op op : {SCRAPGPU_OP_ADD, SCRAPGPU_OP_SUB, SCRAPGPU_OP_MUL, SCRAPGPU_OP_DIV}) {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 37, 13));
        TT s1 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 37)); // row broadcast over dim 1
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 37, 13));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        tfill(s1, frand_vec(tn(s1.rdesc), op == SCRAPGPU_OP_DIV));
        static const char * const names[] = {"ADD", "SUB", "MUL", "DIV"};
        test_op(names[op - SCRAPGPU_OP_ADD], op, {s0, s1}, {d}, nullptr, 0);
    }
    // 3D + non-contiguous src0 (padded rows, transferred as a raw image)
    for (scrapgpu_op op : {SCRAPGPU_OP_ADD, SCRAPGPU_OP_MUL}) {
        scrapgpu_tensor_desc sd = tdesc(SCRAPGPU_DTYPE_F32, 7, 4, 3);
        sd.nb[1] = 5 * sizeof(float);  // padded row
        sd.nb[2] = sd.nb[1] * 4;
        TT s0 = tmake(sd);
        tfill(s0, frand_vec(tn(s0.rdesc)));
        TT s1 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 7, 4, 1));
        tfill(s1, frand_vec(tn(s1.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 7, 4, 3));
        test_op(op == SCRAPGPU_OP_ADD ? "ADD strided+bcast" : "MUL strided+bcast", op, {s0, s1}, {d}, nullptr, 0);
    }
}

void test_unary_family() {
    TT s = tmake(tdesc(SCRAPGPU_DTYPE_F32, 97));
    tfill(s, frand_vec(97));

    scrapgpu_scale_params scale{sizeof(scale), 1.5f, -0.25f};
    scrapgpu_clamp_params clamp{sizeof(clamp), -0.5f, 0.5f};
    test_op("SCALE", SCRAPGPU_OP_SCALE, {s}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 97))}, &scale, sizeof(scale));
    test_op("CLAMP", SCRAPGPU_OP_CLAMP, {s}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 97))}, &clamp, sizeof(clamp));
    test_op("SQR", SCRAPGPU_OP_SQR, {s}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 97))}, nullptr, 0);

    TT sp = tmake(tdesc(SCRAPGPU_DTYPE_F32, 97));
    tfill(sp, frand_vec(97, true));
    test_op("SQRT", SCRAPGPU_OP_SQRT, {sp}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 97))}, nullptr, 0);

    static const char * const kinds[] = {"SILU", "RELU", "NEG", "ABS", "SIGMOID", "TANH"};
    for (int k = SCRAPGPU_UNARY_SILU; k <= SCRAPGPU_UNARY_TANH; ++k) {
        scrapgpu_unary_params p{sizeof(p), (scrapgpu_unary_kind) k};
        char name[40];
        snprintf(name, sizeof(name), "UNARY/%s", kinds[k]);
        test_op(name, SCRAPGPU_OP_UNARY, {s}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 97))}, &p, sizeof(p));
    }
}

void test_glu() {
    for (int swapped = 0; swapped < 2; ++swapped) {
        for (scrapgpu_glu_kind kind : {SCRAPGPU_GLU_SWIGLU, SCRAPGPU_GLU_REGLU}) {
            TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 2 * 33, 9));
            tfill(s0, frand_vec(tn(s0.rdesc)));
            TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 33, 9));
            scrapgpu_glu_params p{sizeof(p), kind, 0, swapped};
            char name[48];
            snprintf(name, sizeof(name), "GLU fused %s sw=%d", kind == SCRAPGPU_GLU_SWIGLU ? "swiglu" : "reglu", swapped);
            test_op(name, SCRAPGPU_OP_GLU, {s0}, {d}, &p, sizeof(p));
        }
    }
    for (scrapgpu_glu_kind kind : {SCRAPGPU_GLU_SWIGLU, SCRAPGPU_GLU_REGLU}) {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 33, 9));
        TT s1 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 33, 9));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        tfill(s1, frand_vec(tn(s1.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 33, 9));
        scrapgpu_glu_params p{sizeof(p), kind, 1, 0};
        test_op(kind == SCRAPGPU_GLU_SWIGLU ? "GLU split swiglu" : "GLU split reglu", SCRAPGPU_OP_GLU, {s0, s1}, {d}, &p, sizeof(p));
    }
}

void test_rms_family() {
    TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 37, 9));
    tfill(x, frand_vec(tn(x.rdesc)));
    TT w = tmake(tdesc(SCRAPGPU_DTYPE_F32, 37));
    tfill(w, frand_vec(37));
    scrapgpu_rms_norm_params eps{sizeof(eps), 1e-5f};

    test_op("RMS_NORM", SCRAPGPU_OP_RMS_NORM, {x}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 37, 9))}, &eps, sizeof(eps));
    test_op("RMS_NORM_MUL", SCRAPGPU_OP_RMS_NORM_MUL, {x, w}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 37, 9))}, &eps, sizeof(eps));

    // the ADD + RMS_NORM * w fusion works on a single row
    TT a = tmake(tdesc(SCRAPGPU_DTYPE_F32, 37));
    TT b = tmake(tdesc(SCRAPGPU_DTYPE_F32, 37));
    tfill(a, frand_vec(37));
    tfill(b, frand_vec(37));
    test_op("ADD_RMS_NORM_MUL", SCRAPGPU_OP_ADD_RMS_NORM_MUL, {a, b, w},
            {tmake(tdesc(SCRAPGPU_DTYPE_F32, 37)), tmake(tdesc(SCRAPGPU_DTYPE_F32, 37))}, &eps, sizeof(eps));
}

void test_soft_max() {
    TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 19, 7));
    tfill(s0, frand_vec(tn(s0.rdesc)));
    scrapgpu_soft_max_params p{sizeof(p), 0.8f, nullptr};
    test_op("SOFT_MAX", SCRAPGPU_OP_SOFT_MAX, {s0}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 19, 7))}, &p, sizeof(p));

    TT m32 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 19, 7));
    tfill(m32, frand_vec(tn(m32.rdesc)));
    scrapgpu_soft_max_params pm{sizeof(pm), 0.8f, &m32.rdesc};
    test_op("SOFT_MAX mask f32", SCRAPGPU_OP_SOFT_MAX, {s0, m32}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 19, 7))}, &pm, sizeof(pm));

    TT m16 = tmake(tdesc(SCRAPGPU_DTYPE_F16, 19, 7));
    tfill(m16, frand_vec(tn(m16.rdesc)));
    scrapgpu_soft_max_params pm16{sizeof(pm16), 0.8f, &m16.rdesc};
    test_op("SOFT_MAX mask f16", SCRAPGPU_OP_SOFT_MAX, {s0, m16}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 19, 7))}, &pm16, sizeof(pm16));
}

void test_rope() {
    // x [head_dim, heads, tokens]; the positions tensor indexes dim 2
    for (scrapgpu_rope_mode mode : {SCRAPGPU_ROPE_MODE_NORMAL, SCRAPGPU_ROPE_MODE_NEOX}) {
        for (int with_ff = 0; with_ff < 2; ++with_ff) {
            for (int f16 = 0; f16 < 2; ++f16) {
                const scrapgpu_dtype dt = f16 ? SCRAPGPU_DTYPE_F16 : SCRAPGPU_DTYPE_F32;
                TT x = tmake(tdesc(dt, 32, 3, 5));
                tfill(x, frand_vec(tn(x.rdesc)));
                TT pos = tmake(tdesc(SCRAPGPU_DTYPE_I32, 5));
                std::vector<int32_t> pv(5);
                for (auto & i : pv) {
                    i = std::uniform_int_distribution<int>(0, 100)(g_rng);
                }
                tfill_i32(pos, pv);
                TT d = tmake(tdesc(dt, 32, 3, 5));
                scrapgpu_rope_params p{};
                p.struct_size = sizeof(p);
                p.mode = mode;
                p.n_dims = 32;
                p.n_ctx_orig = 128;
                p.freq_base = 10000.0f;
                p.freq_scale = 1.0f;
                p.ext_factor = 0.0f;
                p.attn_factor = 1.0f;
                p.beta_fast = 32.0f;
                p.beta_slow = 1.0f;
                std::vector<TT> ins = {x, pos};
                TT ff = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16));
                tfill(ff, frand_vec(16, true));
                if (with_ff) {
                    ins.push_back(ff);
                    p.freq_factors = &ins[2].rdesc;
                }
                char name[64];
                snprintf(name, sizeof(name), "ROPE %s %s%s", mode == SCRAPGPU_ROPE_MODE_NEOX ? "neox" : "norm",
                         f16 ? "f16" : "f32", with_ff ? " ff" : "");
                test_op(name, SCRAPGPU_OP_ROPE, ins, {d}, &p, sizeof(p));
            }
        }
    }
}

void test_get_rows() {
    const auto ids_of = [](int n, int rows) {
        TT t = tmake(tdesc(SCRAPGPU_DTYPE_I32, n));
        std::vector<int32_t> v(n);
        for (auto & i : v) {
            i = std::uniform_int_distribution<int>(0, rows - 1)(g_rng);
        }
        tfill_i32(t, v);
        return t;
    };
    scrapgpu_get_rows_params p{sizeof(p), 0};
    {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 23, 50));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        test_op("GET_ROWS f32", SCRAPGPU_OP_GET_ROWS, {s0, ids_of(9, 50)}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 23, 9))}, &p, sizeof(p), 0.0, true);
    }
    {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F16, 23, 50));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        test_op("GET_ROWS f16", SCRAPGPU_OP_GET_ROWS, {s0, ids_of(9, 50)}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 23, 9))}, &p, sizeof(p), 0.0, true);
    }
    for (scrapgpu_dtype qt : {SCRAPGPU_DTYPE_Q4_0, SCRAPGPU_DTYPE_Q8_0, SCRAPGPU_DTYPE_Q4_K, SCRAPGPU_DTYPE_Q6_K}) {
        const int64_t k = qt == SCRAPGPU_DTYPE_Q4_K || qt == SCRAPGPU_DTYPE_Q6_K ? 256 : 64;
        TT s0 = tmake(tdesc(qt, k, 32), g_weight_layout);
        tquantize(s0, frand_vec(tn(s0.rdesc)));
        char name[48];
        snprintf(name, sizeof(name), "GET_ROWS %s r32",
                 qt == SCRAPGPU_DTYPE_Q4_0 ? "q4_0" : qt == SCRAPGPU_DTYPE_Q8_0 ? "q8_0" : qt == SCRAPGPU_DTYPE_Q4_K ? "q4_k" : "q6_k");
        test_op(name, SCRAPGPU_OP_GET_ROWS, {s0, ids_of(7, 32)}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, k, 7))}, &p, sizeof(p), 0.0, true);
    }
}

// n distinct row indices in [0, rows) (a repeated dst row would race)
std::vector<int64_t> unique_ids(int64_t n, int64_t rows) {
    std::vector<int64_t> v((size_t) rows);
    for (int64_t i = 0; i < rows; ++i) {
        v[(size_t) i] = i;
    }
    std::shuffle(v.begin(), v.end(), g_rng);
    v.resize((size_t) n);
    return v;
}

void test_set_rows() {
    {
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 23, 50));
        d.prefilled = true;
        tfill(d, frand_vec(tn(d.rdesc)));
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 23, 9));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        TT ids = tmake(tdesc(SCRAPGPU_DTYPE_I32, 9));
        const std::vector<int64_t> u = unique_ids(9, 50);
        std::vector<int32_t> iv(u.begin(), u.end());
        tfill_i32(ids, iv);
        scrapgpu_set_rows_params p{sizeof(p), 0};
        test_op("SET_ROWS f32 i32", SCRAPGPU_OP_SET_ROWS, {s0, ids}, {d}, &p, sizeof(p), 0.0, true);
    }
    {
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F16, 16, 20));
        d.prefilled = true;
        tfill(d, frand_vec(tn(d.rdesc)));
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F16, 16, 5));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        TT ids = tmake(tdesc(SCRAPGPU_DTYPE_I64, 5));
        tfill_i64(ids, unique_ids(5, 20));
        scrapgpu_set_rows_params p{sizeof(p), 1};
        test_op("SET_ROWS f16 i64", SCRAPGPU_OP_SET_ROWS, {s0, ids}, {d}, &p, sizeof(p), 0.0, true);
    }
    {
        // the pair fusion (one launch on devices with fuse_set_rows, two SET_ROWS otherwise)
        TT da = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 20));
        TT db = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 20));
        da.prefilled = db.prefilled = true;
        tfill(da, frand_vec(tn(da.rdesc)));
        tfill(db, frand_vec(tn(db.rdesc)));
        TT sa = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 4));
        TT sb = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 4));
        tfill(sa, frand_vec(tn(sa.rdesc)));
        tfill(sb, frand_vec(tn(sb.rdesc)));
        TT ia = tmake(tdesc(SCRAPGPU_DTYPE_I32, 4));
        TT ib = tmake(tdesc(SCRAPGPU_DTYPE_I32, 4));
        const std::vector<int64_t> ua = unique_ids(4, 20), ub = unique_ids(4, 20);
        std::vector<int32_t> va(ua.begin(), ua.end()), vb(ub.begin(), ub.end());
        tfill_i32(ia, va);
        tfill_i32(ib, vb);

        scrapgpu_set_rows_params p{sizeof(p), 0};
        std::vector<TT> ri_a = {sa, ia}, ro_a = {da};
        std::vector<TT> ri_b = {sb, ib}, ro_b = {db};
        ref_exec(SCRAPGPU_OP_SET_ROWS, ri_a, ro_a, &p, sizeof(p));
        ref_exec(SCRAPGPU_OP_SET_ROWS, ri_b, ro_b, &p, sizeof(p));
        std::vector<TT> ins = {sa, ia, sb, ib};
        std::vector<TT> outs = {da, db};
        run_one("SET_ROWS_PAIR", SCRAPGPU_OP_SET_ROWS_PAIR, ins, outs, nullptr, 0,
                {tfloats(ro_a[0]), tfloats(ro_b[0])}, 0.0, true);
    }
}

void test_cpy() {
    {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 11, 7));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        scrapgpu_cpy_params p{sizeof(p), 0};
        test_op("CPY f32->f32", SCRAPGPU_OP_CPY, {s0}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 11, 7))}, &p, sizeof(p), 0.0, true);
    }
    {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 11, 7));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        test_op("DUP f32->f16", SCRAPGPU_OP_DUP, {s0}, {tmake(tdesc(SCRAPGPU_DTYPE_F16, 11, 7))}, nullptr, 0, 0.0, true);
    }
    {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F16, 11, 7));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        test_op("CONT f16->f32", SCRAPGPU_OP_CONT, {s0}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 11, 7))}, nullptr, 0, 0.0, true);
    }
    {
        // transposed view: logical [6, 8] over an [8][6] image
        scrapgpu_tensor_desc sd = tdesc(SCRAPGPU_DTYPE_F32, 6, 8);
        sd.nb[0] = 8 * sizeof(float);
        sd.nb[1] = sizeof(float);
        sd.nb[2] = 6 * 8 * sizeof(float);
        TT s0 = tmake(sd);
        tfill(s0, frand_vec(tn(s0.rdesc)));
        scrapgpu_cpy_params p{sizeof(p), 1};
        test_op("CPY transposed", SCRAPGPU_OP_CPY, {s0}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 6, 8))}, &p, sizeof(p), 0.0, true);
    }
}

void test_mul_mat() {
    scrapgpu_mul_mat_params p{sizeof(p), 1};
    for (scrapgpu_dtype dt : {SCRAPGPU_DTYPE_F32, SCRAPGPU_DTYPE_F16}) {
        TT s0 = tmake(tdesc(dt, 64, 16));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 64, 5));
        tfill(x, frand_vec(tn(x.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 5));
        test_op(dt == SCRAPGPU_DTYPE_F32 ? "MUL_MAT f32" : "MUL_MAT f16", SCRAPGPU_OP_MUL_MAT, {s0, x}, {d}, &p, sizeof(p));
    }
    // quantized weights, uploaded as R32 (the upload packs them)
    for (scrapgpu_dtype qt : {SCRAPGPU_DTYPE_Q4_0, SCRAPGPU_DTYPE_Q8_0, SCRAPGPU_DTYPE_Q4_K, SCRAPGPU_DTYPE_Q6_K}) {
        const int64_t k = qt == SCRAPGPU_DTYPE_Q4_K || qt == SCRAPGPU_DTYPE_Q6_K ? 256 : 96;
        for (int64_t cols : {1, 4}) {
            TT s0 = tmake(tdesc(qt, k, 24), g_weight_layout);
            tquantize(s0, frand_vec(tn(s0.rdesc)));
            TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, k, cols));
            tfill(x, frand_vec(tn(x.rdesc)));
            TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 24, cols));
            char name[48];
            snprintf(name, sizeof(name), "MUL_MAT %s n=%lld",
                     qt == SCRAPGPU_DTYPE_Q4_0 ? "q4_0" : qt == SCRAPGPU_DTYPE_Q8_0 ? "q8_0" : qt == SCRAPGPU_DTYPE_Q4_K ? "q4_k" : "q6_k",
                     (long long) cols);
            test_op(name, SCRAPGPU_OP_MUL_MAT, {s0, x}, {d}, &p, sizeof(p), 1e-6);
        }
    }
    // batch broadcast of src0 over src1's ne2
    {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 32, 8));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 32, 4, 3));
        tfill(x, frand_vec(tn(x.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 4, 3));
        test_op("MUL_MAT batch", SCRAPGPU_OP_MUL_MAT, {s0, x}, {d}, &p, sizeof(p));
    }
    // a row-padded src0 (raw image upload, strides honoured)
    {
        scrapgpu_tensor_desc sd = tdesc(SCRAPGPU_DTYPE_F32, 32, 8);
        sd.nb[1] = 36 * sizeof(float);
        sd.nb[2] = sd.nb[1] * 8;
        TT s0 = tmake(sd);
        tfill(s0, frand_vec(tn(s0.rdesc)));
        TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 32, 3));
        tfill(x, frand_vec(tn(x.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 3));
        test_op("MUL_MAT padded s0", SCRAPGPU_OP_MUL_MAT, {s0, x}, {d}, &p, sizeof(p));
    }
}

void test_mul_mat_multi() {
    for (int n : {2, 3}) {
        std::vector<TT> ins, outs;
        for (int i = 0; i < n; ++i) {
            TT w = tmake(tdesc(SCRAPGPU_DTYPE_Q4_0, 96, 8), g_weight_layout);
            tquantize(w, frand_vec(tn(w.rdesc)));
            ins.push_back(w);
        }
        TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 96));
        tfill(x, frand_vec(96));
        ins.push_back(x);
        for (int i = 0; i < n; ++i) {
            outs.push_back(tmake(tdesc(SCRAPGPU_DTYPE_F32, 8)));
        }
        // reference: one MUL_MAT per weight with the shared vector
        scrapgpu_mul_mat_params p{sizeof(p), 0};
        std::vector<std::vector<float>> ref;
        for (int i = 0; i < n; ++i) {
            std::vector<TT> ri = {ins[i], x};
            std::vector<TT> ro = {tmake_ref(outs[i].rdesc)};
            ref_exec(SCRAPGPU_OP_MUL_MAT, ri, ro, &p, sizeof(p));
            ref.push_back(tfloats(ro[0]));
        }
        char name[40];
        snprintf(name, sizeof(name), "MUL_MAT_MULTI n=%d", n);
        run_one(name, SCRAPGPU_OP_MUL_MAT_MULTI, ins, outs, nullptr, 0, ref, 1e-6, false);
    }
}

void test_mul_mat_id() {
    scrapgpu_mul_mat_id_params p{sizeof(p), 2, 0};
    for (scrapgpu_dtype qt : {SCRAPGPU_DTYPE_Q4_0, SCRAPGPU_DTYPE_Q4_K}) {
        const int64_t k = qt == SCRAPGPU_DTYPE_Q4_0 ? 96 : 256;
        const int64_t m = 8, e = 4, used = 2, tok = 3;
        TT s0 = tmake(tdesc(qt, k, m, e), g_weight_layout);
        tquantize(s0, frand_vec(tn(s0.rdesc)));
        TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, k, used, tok));
        tfill(x, frand_vec(tn(x.rdesc)));
        TT ids = tmake(tdesc(SCRAPGPU_DTYPE_I32, used, tok));
        std::vector<int32_t> iv((size_t) (used * tok));
        for (auto & i : iv) {
            i = std::uniform_int_distribution<int>(0, (int) e - 1)(g_rng);
        }
        tfill_i32(ids, iv);
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, m, used, tok));
        test_op(qt == SCRAPGPU_DTYPE_Q4_0 ? "MUL_MAT_ID q4_0" : "MUL_MAT_ID q4_k",
                SCRAPGPU_OP_MUL_MAT_ID, {s0, x, ids}, {d}, &p, sizeof(p), 1e-6);
    }
    {
        TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 64, 8, 4));
        tfill(s0, frand_vec(tn(s0.rdesc)));
        TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, 64, 2, 3));
        tfill(x, frand_vec(tn(x.rdesc)));
        TT ids = tmake(tdesc(SCRAPGPU_DTYPE_I32, 2, 3));
        std::vector<int32_t> iv(6);
        for (auto & i : iv) {
            i = std::uniform_int_distribution<int>(0, 3)(g_rng);
        }
        tfill_i32(ids, iv);
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 2, 3));
        test_op("MUL_MAT_ID f32", SCRAPGPU_OP_MUL_MAT_ID, {s0, x, ids}, {d}, &p, sizeof(p));
    }
}

void test_add_id() {
    TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 2, 3));
    tfill(s0, frand_vec(tn(s0.rdesc)));
    TT s1 = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 4));
    tfill(s1, frand_vec(tn(s1.rdesc)));
    TT ids = tmake(tdesc(SCRAPGPU_DTYPE_I32, 2, 3));
    std::vector<int32_t> iv(6);
    for (auto & i : iv) {
        i = std::uniform_int_distribution<int>(0, 3)(g_rng);
    }
    tfill_i32(ids, iv);
    TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 2, 3));
    test_op("ADD_ID", SCRAPGPU_OP_ADD_ID, {s0, s1, ids}, {d}, nullptr, 0);
}

void test_ssm_conv() {
    // s0 [d_conv-1 + seq, embd, n], s1 [d_conv, embd], dst [embd, seq, n]
    const int64_t d_conv = 4, seq = 6, embd = 8, n = 2;
    TT s0 = tmake(tdesc(SCRAPGPU_DTYPE_F32, d_conv - 1 + seq, embd, n));
    tfill(s0, frand_vec(tn(s0.rdesc)));
    TT s1 = tmake(tdesc(SCRAPGPU_DTYPE_F32, d_conv, embd));
    tfill(s1, frand_vec(tn(s1.rdesc)));
    TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, embd, seq, n));
    scrapgpu_ssm_conv_params p{sizeof(p), (uint32_t) d_conv};
    test_op("SSM_CONV", SCRAPGPU_OP_SSM_CONV, {s0, s1}, {d}, &p, sizeof(p));
}

void test_concat_pad_sumrows() {
    {
        TT a = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 5));
        TT b = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 3));
        tfill(a, frand_vec(tn(a.rdesc)));
        tfill(b, frand_vec(tn(b.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 8));
        scrapgpu_concat_params p{sizeof(p), 1};
        test_op("CONCAT dim1", SCRAPGPU_OP_CONCAT, {a, b}, {d}, &p, sizeof(p), 0.0, true);
    }
    {
        TT a = tmake(tdesc(SCRAPGPU_DTYPE_F16, 8, 5));
        TT b = tmake(tdesc(SCRAPGPU_DTYPE_F16, 8, 3));
        tfill(a, frand_vec(tn(a.rdesc)));
        tfill(b, frand_vec(tn(b.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F16, 8, 8));
        scrapgpu_concat_params p{sizeof(p), 1};
        test_op("CONCAT f16", SCRAPGPU_OP_CONCAT, {a, b}, {d}, &p, sizeof(p), 0.0, true);
    }
    {
        TT s = tmake(tdesc(SCRAPGPU_DTYPE_F32, 5, 3));
        tfill(s, frand_vec(tn(s.rdesc)));
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, 8, 7));
        scrapgpu_pad_params p{};
        p.struct_size = sizeof(p);
        p.before[0] = 1; p.after[0] = 2;
        p.before[1] = 2; p.after[1] = 2;
        test_op("PAD", SCRAPGPU_OP_PAD, {s}, {d}, &p, sizeof(p), 0.0, true);
    }
    {
        TT s = tmake(tdesc(SCRAPGPU_DTYPE_F32, 17, 9));
        tfill(s, frand_vec(tn(s.rdesc)));
        test_op("SUM_ROWS", SCRAPGPU_OP_SUM_ROWS, {s}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 1, 9))}, nullptr, 0);
    }
    {
        TT s = tmake(tdesc(SCRAPGPU_DTYPE_F32, 16, 5, 3));
        tfill(s, frand_vec(tn(s.rdesc)));
        test_op("SUM_ROWS 3d", SCRAPGPU_OP_SUM_ROWS, {s}, {tmake(tdesc(SCRAPGPU_DTYPE_F32, 1, 5, 3))}, nullptr, 0);
    }
}

// "exact, ties allowed": every row must be a valid argsort (a permutation, values sorted) /
// a valid top-k (k distinct indices whose values are the k largest, descending)
void check_argsort(const char * name, const TT & s, const TT & d, scrapgpu_argsort_order order, int64_t k) {
    const std::vector<float> sv = tfloats(s);
    const std::vector<float> idx = tfloats(d);
    const int64_t n = s.rdesc.ne[0];
    const int64_t kk = k > 0 ? k : n;
    const int64_t rows = tn(s.rdesc) / n;
    for (int64_t r = 0; r < rows; ++r) {
        const float * row = sv.data() + (size_t) (r * n);
        std::vector<char> seen((size_t) n, 0);
        std::vector<char> chosen((size_t) n, 0);
        float prev = order == SCRAPGPU_ARGSORT_ASC ? -INFINITY : INFINITY;
        for (int64_t i = 0; i < kk; ++i) {
            const int64_t e = (int64_t) idx[(size_t) (r * kk + i)];
            CHECK(e >= 0 && e < n);
            CHECK(seen[e]++ == 0);
            const float v = row[e];
            CHECK(order == SCRAPGPU_ARGSORT_ASC ? v >= prev : v <= prev);
            prev = v;
            chosen[e] = 1;
        }
        for (int64_t i = 0; i < n && k > 0; ++i) { // every unchosen element is not larger than the smallest chosen
            if (!chosen[i]) {
                CHECK(order == SCRAPGPU_ARGSORT_ASC ? row[i] >= prev : row[i] <= prev);
            }
        }
    }
    std::printf("PASS  %-24s sorted order verified (%lld rows x %lld)\n", name, (long long) rows, (long long) kk);
    ++g_run;
}

void gpu_argsort(const char * name, scrapgpu_op op, const TT & s_, TT & d_, const scrapgpu_argsort_params & p) {
    TT s = s_, d = d_;
    std::vector<TT> ins = {s}, outs = {d};
    std::vector<scrapgpu_tensor_desc> in_d = {ins[0].gdesc}, out_d = {outs[0].gdesc};
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = op;
    desc.inputs = in_d.data();
    desc.input_count = 1;
    desc.outputs = out_d.data();
    desc.output_count = 1;
    desc.params = &p;
    desc.params_size = sizeof(p);
    const char * reason = nullptr;
    if (scrapgpu_op_supported(g_dev, &desc, &reason) != SCRAPGPU_STATUS_OK) {
        std::printf("SKIP  %-24s %s\n", name, reason != nullptr ? reason : "");
        ++g_skipped;
        return;
    }
    tupload(ins[0]);
    CHECK_OK(scrapgpu_op_execute(g_stream, &desc));
    CHECK_OK(scrapgpu_stream_sync(g_stream));
    tdownload(outs[0]);
    check_argsort(name, s, outs[0], p.order, (int64_t) p.top_k);
}

void test_argsort() {
    TT s = tmake(tdesc(SCRAPGPU_DTYPE_F32, 13, 5));
    tfill(s, frand_vec(tn(s.rdesc)));
    {
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_I32, 13, 5));
        scrapgpu_argsort_params p{sizeof(p), SCRAPGPU_ARGSORT_ASC, 0};
        gpu_argsort("ARGSORT asc", SCRAPGPU_OP_ARGSORT, s, d, p);
    }
    {
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_I32, 13, 5));
        scrapgpu_argsort_params p{sizeof(p), SCRAPGPU_ARGSORT_DESC, 0};
        gpu_argsort("ARGSORT desc", SCRAPGPU_OP_ARGSORT, s, d, p);
    }
    {
        // TOP_K is the k largest indices, always in descending order (both the reference and the
        // kernel treat top_k as desc)
        TT d = tmake(tdesc(SCRAPGPU_DTYPE_I32, 4, 5));
        scrapgpu_argsort_params p{sizeof(p), SCRAPGPU_ARGSORT_DESC, 4};
        gpu_argsort("TOP_K k=4 desc", SCRAPGPU_OP_TOP_K, s, d, p);
    }
}

void test_cont_mul_mat() {
    // a is the transposed F16 view of a V cache: logical [head_dim, n_kv, kv_heads] with the
    // elements of dim 1 contiguous in memory (storage [kv_heads][n_kv][head_dim])
    const int64_t head_dim = 64, n_kv = 8, kv_heads = 2, q_heads = 4;
    scrapgpu_tensor_desc ad = tdesc(SCRAPGPU_DTYPE_F16, head_dim, n_kv, kv_heads);
    ad.nb[0] = (uint64_t) n_kv * 2;
    ad.nb[1] = 2;
    ad.nb[2] = (uint64_t) n_kv * head_dim * 2;
    ad.nb[3] = ad.nb[2] * kv_heads;
    TT a = tmake(ad);
    tfill(a, frand_vec(tn(a.rdesc)));

    TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, head_dim, 1, q_heads));
    tfill(x, frand_vec(tn(x.rdesc)));
    TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, n_kv, 1, q_heads));

    // reference: CONT(a) -> contiguous [head_dim, n_kv, kv_heads], then MUL_MAT
    // (ref_exec writes into the TTs of the vector, so the intermediate tensor stays there)
    std::vector<TT> cont = {tmake_ref(tdesc(SCRAPGPU_DTYPE_F16, head_dim, n_kv, kv_heads))};
    {
        std::vector<TT> ci = {a};
        ref_exec(SCRAPGPU_OP_CONT, ci, cont, nullptr, 0);
    }
    std::vector<TT> dref = {tmake_ref(d.rdesc)};
    {
        scrapgpu_mul_mat_params p{sizeof(p), 1};
        std::vector<TT> mi = {cont[0], x};
        ref_exec(SCRAPGPU_OP_MUL_MAT, mi, dref, &p, sizeof(p));
    }
    std::vector<TT> ins = {a, x};
    std::vector<TT> outs = {d};
    run_one("CONT_MUL_MAT", SCRAPGPU_OP_CONT_MUL_MAT, ins, outs, nullptr, 0, {tfloats(dref[0])}, 1e-7, false);
}

// The V cache case of llama.cpp (kq * V with the transposed V view): a is [n_kv, head_dim, kv_heads] with the
// elements of dim 1 contiguous, one row of the cache (all heads of a token) per n_kv step. This is the layout the
// k_mvt_f16 kernel of the devices without mvt_groups (NVIDIA) needs (cl11_attn_ok: hd and n_kv multiples of 8,
// 16-byte aligned rows, at most 8 query heads per cache head).
// n_kv = 24 is below the slice count of a 1024-wide work-group (S = 64 for 16 lanes) and n_kv = 520 does not split
// evenly into the S slices: both lost positions with the old slice size of k_mvt_f16 (floor(K / S) rounded to 8).
void test_cont_mul_mat_vcache_k(int64_t n_kv) {
    const int64_t head_dim = 64, kv_heads = 2, q_heads = 8;
    scrapgpu_tensor_desc ad = tdesc(SCRAPGPU_DTYPE_F16, n_kv, head_dim, kv_heads);
    ad.nb[0] = (uint64_t) head_dim * kv_heads * 2; // one cache row: every head of a token
    ad.nb[1] = 2;
    ad.nb[2] = (uint64_t) head_dim * 2;
    ad.nb[3] = ad.nb[0] * n_kv;
    TT a = tmake(ad);
    tfill(a, frand_vec(tn(a.rdesc)));

    TT x = tmake(tdesc(SCRAPGPU_DTYPE_F32, n_kv, 1, q_heads));
    tfill(x, frand_vec(tn(x.rdesc)));
    TT d = tmake(tdesc(SCRAPGPU_DTYPE_F32, head_dim, 1, q_heads));

    std::vector<TT> cont = {tmake_ref(tdesc(SCRAPGPU_DTYPE_F16, n_kv, head_dim, kv_heads))};
    {
        std::vector<TT> ci = {a};
        ref_exec(SCRAPGPU_OP_CONT, ci, cont, nullptr, 0);
    }
    std::vector<TT> dref = {tmake_ref(d.rdesc)};
    {
        scrapgpu_mul_mat_params p{sizeof(p), 1};
        std::vector<TT> mi = {cont[0], x};
        ref_exec(SCRAPGPU_OP_MUL_MAT, mi, dref, &p, sizeof(p));
    }
    std::vector<TT> ins = {a, x};
    std::vector<TT> outs = {d};
    const std::string name = "CONT_MUL_MAT V n_kv=" + std::to_string(n_kv);
    run_one(name.c_str(), SCRAPGPU_OP_CONT_MUL_MAT, ins, outs, nullptr, 0, {tfloats(dref[0])}, 1e-7, false);
}

void test_cont_mul_mat_vcache() {
    test_cont_mul_mat_vcache_k(24);
    test_cont_mul_mat_vcache_k(520);
}

void test_buffer_ops() {
    TT t = tmake(tdesc(SCRAPGPU_DTYPE_F32, 64));
    tfill(t, frand_vec(64));
    tupload(t);
    TT c = tmake(tdesc(SCRAPGPU_DTYPE_F32, 64));
    CHECK_OK(scrapgpu_buffer_copy(g_stream, c.buf, 0, t.buf, 0, 64 * sizeof(float), 1));
    CHECK_OK(scrapgpu_buffer_fill(g_stream, c.buf, 32 * sizeof(float), 4 * sizeof(float), 0, 1));
    CHECK_OK(scrapgpu_stream_sync(g_stream));
    tdownload(c);
    const std::vector<float> a = tfloats(t);
    const std::vector<float> b = tfloats(c);
    for (int i = 0; i < 64; ++i) {
        const float want = (i >= 32 && i < 36) ? 0.0f : a[i];
        CHECK(b[i] == want);
    }
    std::printf("PASS  buffer copy/fill\n");
}

} // namespace

int main() {
    const char * backend = std::getenv("SCRAPGPU_TEST_BACKEND");
    if (backend == nullptr || backend[0] == '\0') {
        backend = "opencl11";
    }
    if (std::strcmp(backend, "opencl11") == 0) {
#if defined(SCRAPGPU_OPS_TEST_WITH_OPENCL11)
        if (scrapgpu_opencl11_register() != SCRAPGPU_STATUS_OK) {
            std::fprintf(stderr, "scrapgpu_opencl11_register failed: %s\n", scrapgpu_last_error());
            return 1;
        }
#else
        std::printf("SKIP: built without the OpenCL 1.1 backend\n");
        return 77;
#endif
    } else if (std::strcmp(backend, "vulkan") == 0) {
#if defined(SCRAPGPU_OPS_TEST_WITH_VULKAN)
        if (scrapgpu_vulkan_register() != SCRAPGPU_STATUS_OK) {
            std::fprintf(stderr, "scrapgpu_vulkan_register failed: %s\n", scrapgpu_last_error());
            return 1;
        }
        g_weight_layout = SCRAPGPU_LAYOUT_NATURAL;
#else
        std::printf("SKIP: built without the Vulkan backend\n");
        return 77;
#endif
    } else {
        std::fprintf(stderr, "unknown SCRAPGPU_TEST_BACKEND '%s'\n", backend);
        return 1;
    }
    if (scrapgpu_get_device_count() == 0) {
        std::printf("SKIP: no %s device\n", backend);
        return 77;
    }
    CHECK_OK(scrapgpu_get_device(0, &g_dev));

    {
        scrapgpu_device_info info{};
        CHECK_OK(scrapgpu_device_info_init(&info, sizeof(info)));
        CHECK_OK(scrapgpu_device_get_info(g_dev, &info));
        std::printf("device: %s (%s, %s)\n", info.name, info.vendor, info.tuning_profile_name);
    }
    CHECK_OK(scrapgpu_stream_create(g_dev, &g_stream));

    test_buffer_ops();
    test_binary();
    test_unary_family();
    test_glu();
    test_rms_family();
    test_soft_max();
    test_rope();
    test_get_rows();
    test_set_rows();
    test_cpy();
    test_mul_mat();
    test_mul_mat_multi();
    test_mul_mat_id();
    test_add_id();
    test_ssm_conv();
    test_concat_pad_sumrows();
    test_argsort();
    test_cont_mul_mat();
    test_cont_mul_mat_vcache();

    for (scrapgpu_buffer_t * b : g_buffers) {
        CHECK_OK(scrapgpu_buffer_free(b));
    }
    CHECK_OK(scrapgpu_stream_free(g_stream));

    std::printf("%d op runs, %d skipped\n", g_run, g_skipped);
    return 0;
}
