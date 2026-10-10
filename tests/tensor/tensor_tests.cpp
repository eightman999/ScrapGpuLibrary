// tests of the ggml-independent tensor module (src/tensor) and of the kernel ABI structs
// (src/backends/opencl11/tensor_abi.h). No ggml, no OpenCL, no assert(): CHECK stays active in
// Release builds.

#include "tensor/tensor.h"
#include "opencl11/tensor_abi.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                                               \
        }                                                                               \
    } while (0)

namespace {

sgt::TensorDesc desc(scrapgpu_dtype dtype, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    const int64_t ne[4] = { n0, n1, n2, n3 };
    return sgt::make_contiguous(dtype, ne);
}

void test_dtype() {
    CHECK(sgt::dtype_block_size(SCRAPGPU_DTYPE_F32) == 1);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_F32) == 4);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_F16) == 2);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_I32) == 4);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_I64) == 8);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_I16) == 2);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_BF16) == 2);
    // the quantized dtypes: elements per block, bytes per block
    CHECK(sgt::dtype_block_size(SCRAPGPU_DTYPE_Q4_0) == 32);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_Q4_0) == 18);
    CHECK(sgt::dtype_block_size(SCRAPGPU_DTYPE_Q8_0) == 32);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_Q8_0) == 34);
    CHECK(sgt::dtype_block_size(SCRAPGPU_DTYPE_Q4_K) == 256);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_Q4_K) == 144);
    CHECK(sgt::dtype_block_size(SCRAPGPU_DTYPE_Q6_K) == 256);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_Q6_K) == 210);
    CHECK(sgt::dtype_block_size(SCRAPGPU_DTYPE_UNKNOWN) == 0);
    CHECK(sgt::dtype_type_size(SCRAPGPU_DTYPE_UNKNOWN) == 0);

    CHECK(!sgt::dtype_is_quantized(SCRAPGPU_DTYPE_F32));
    CHECK(sgt::dtype_is_quantized(SCRAPGPU_DTYPE_Q4_0));
    CHECK(sgt::dtype_is_quantized(SCRAPGPU_DTYPE_Q6_K));
    CHECK(sgt::dtype_is_repackable(SCRAPGPU_DTYPE_Q4_0));
    CHECK(sgt::dtype_is_repackable(SCRAPGPU_DTYPE_Q8_0));
    CHECK(sgt::dtype_is_repackable(SCRAPGPU_DTYPE_Q4_K));
    CHECK(sgt::dtype_is_repackable(SCRAPGPU_DTYPE_Q6_K));
    CHECK(!sgt::dtype_is_repackable(SCRAPGPU_DTYPE_F32));
    CHECK(!sgt::dtype_is_repackable(SCRAPGPU_DTYPE_F16));

    // a row holds whole blocks (rounded up)
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_F32, 10) == 40);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_Q4_0, 32) == 18);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_Q4_0, 33) == 36);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_Q8_0, 64) == 68);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_Q4_K, 512) == 288);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_Q6_K, 512) == 420);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_F32, -1) == 0);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_UNKNOWN, 64) == 0);
    CHECK(sgt::row_size(SCRAPGPU_DTYPE_I64, INT64_MAX) == 0); // overflow -> 0
}

void test_contiguous() {
    const sgt::TensorDesc c = desc(SCRAPGPU_DTYPE_F32, 8, 4, 2, 1);
    CHECK(sgt::is_contiguous(c));
    CHECK(sgt::is_contiguous_strict(c));
    CHECK(c.nb[0] == 4 && c.nb[1] == 32 && c.nb[2] == 128 && c.nb[3] == 256);
    CHECK(sgt::nelements(c) == 64);
    CHECK(sgt::nbytes(c) == 256);

    // quantized contiguous: nb[1] is the row bytes, not ne0 * type size
    const sgt::TensorDesc q = desc(SCRAPGPU_DTYPE_Q4_0, 64, 3, 1, 1);
    CHECK(sgt::is_contiguous(q));
    CHECK(q.nb[0] == 18 && q.nb[1] == 36 && q.nb[2] == 108);
    CHECK(sgt::nbytes(q) == 108);

    // padded row stride -> not contiguous
    sgt::TensorDesc pad = c;
    pad.nb[1] += 4;
    CHECK(!sgt::is_contiguous(pad));
    CHECK(!sgt::is_contiguous_strict(pad));

    // the stride of a dim of size 1 is not constrained (relaxed), but is constrained strictly
    sgt::TensorDesc s1 = c;
    s1.ne[2] = 1;
    s1.nb[2] = 12345;
    CHECK(sgt::is_contiguous(s1));
    CHECK(!sgt::is_contiguous_strict(s1));

    // a row of a single block: nb[0] is not constrained (relaxed)
    sgt::TensorDesc single_block = desc(SCRAPGPU_DTYPE_Q4_0, 32, 4, 1, 1);
    single_block.nb[0] = 99;
    CHECK(sgt::is_contiguous(single_block));
    CHECK(!sgt::is_contiguous_strict(single_block));

    // a dim of size 0 counts as 1 for the strict check; nelements is 0
    sgt::TensorDesc empty = desc(SCRAPGPU_DTYPE_F32, 8, 0, 1, 1);
    CHECK(sgt::is_contiguous_strict(empty));
    CHECK(sgt::nelements(empty) == 0);
    CHECK(sgt::nbytes(empty) == 0);
}

void test_transposed_permuted() {
    const sgt::TensorDesc m = desc(SCRAPGPU_DTYPE_F16, 4, 3, 2, 1);
    const sgt::TensorDesc t = sgt::transpose(m);
    CHECK(t.ne[0] == 3 && t.ne[1] == 4);
    CHECK(t.nb[0] == m.nb[1] && t.nb[1] == m.nb[0]);
    CHECK(sgt::is_transposed(t));
    CHECK(sgt::is_permuted(t));
    CHECK(!sgt::is_transposed(m));
    CHECK(!sgt::is_permuted(m));

    const sgt::TensorDesc p = sgt::permute(m, 2, 1, 0, 3);
    CHECK(p.ne[0] == 2 && p.ne[1] == 3 && p.ne[2] == 4 && p.ne[3] == 1);
    CHECK(p.nb[0] == m.nb[2] && p.nb[2] == m.nb[0]);
    CHECK(sgt::is_permuted(p));
    CHECK(!sgt::is_contiguous(p));
}

void test_desc_of() {
    // the storage part of the public C descriptor converts to a TensorDesc field by field
    scrapgpu_tensor_desc s = {};
    s.struct_size = sizeof(s);
    s.dtype  = SCRAPGPU_DTYPE_Q4_K;
    s.layout = SCRAPGPU_LAYOUT_R32;
    s.offset = 77;
    s.ne[0] = 512;
    s.ne[1] = 4;
    const sgt::TensorDesc d = sgt::desc_of(s);
    CHECK(d.dtype == SCRAPGPU_DTYPE_Q4_K && d.layout == SCRAPGPU_LAYOUT_R32 && d.offset == 77);
    CHECK(d.ne[0] == 512 && d.ne[1] == 4 && d.ne[2] == 0 && d.ne[3] == 0);
}

void test_view() {
    const sgt::TensorDesc p = desc(SCRAPGPU_DTYPE_F32, 8, 4, 2, 1);
    const int64_t  ne[4] = { 4, 2, 1, 1 };
    const uint64_t nb[4] = { p.nb[0], p.nb[1], p.nb[2], p.nb[3] };
    const sgt::TensorDesc v = sgt::view(p, 7 * 4, ne, nb);
    CHECK(v.offset == 28);
    CHECK(v.dtype == p.dtype);
    CHECK(v.layout == p.layout);
    // a view of a view accumulates the offset
    const sgt::TensorDesc v2 = sgt::view(v, 4, ne, nb);
    CHECK(v2.offset == 32);

    // nbytes is the last byte touched + 1, not the parent size
    CHECK(sgt::nbytes(v) == 4 + 3 * 4 + 1 * 32); // element (3, 1, 0, 0)
    CHECK(sgt::nbytes(v) < sgt::nbytes(p));

    // a strided view is not contiguous
    const uint64_t nb_col[4] = { 4, 8, 128, 256 };
    const sgt::TensorDesc col = sgt::view(p, 0, ne, nb_col);
    CHECK(!sgt::is_contiguous(col));
    CHECK(sgt::nbytes(col) == 4 + 3 * 4 + 1 * 8);
}

void test_reshape() {
    const sgt::TensorDesc m = desc(SCRAPGPU_DTYPE_F32, 4, 6, 1, 1);
    sgt::TensorDesc r;
    const int64_t ne_ok[4] = { 8, 3, 1, 1 };
    CHECK(sgt::reshape(m, ne_ok, r));
    CHECK(sgt::is_contiguous(r));
    CHECK(r.ne[0] == 8 && r.ne[1] == 3);
    CHECK(r.nb[0] == 4 && r.nb[1] == 32);
    CHECK(sgt::nbytes(r) == sgt::nbytes(m));

    const int64_t ne_bad[4] = { 5, 5, 1, 1 };
    CHECK(!sgt::reshape(m, ne_bad, r)); // element count differs

    sgt::TensorDesc nc = m;
    nc.nb[1] += 4;
    CHECK(!sgt::reshape(nc, ne_ok, r)); // not contiguous

    sgt::TensorDesc q = desc(SCRAPGPU_DTYPE_Q4_0, 64, 4, 1, 1);
    const int64_t ne_q[4] = { 96, 1, 1, 1 };
    CHECK(!sgt::reshape(q, ne_q, r));   // 96 is not a multiple of the block size
    const int64_t ne_qok[4] = { 32, 8, 1, 1 };
    CHECK(sgt::reshape(q, ne_qok, r));

    sgt::TensorDesc rp = desc(SCRAPGPU_DTYPE_Q4_0, 64, 32, 1, 1);
    rp.layout = SCRAPGPU_LAYOUT_R32;
    CHECK(!sgt::reshape(rp, ne_qok, r)); // an R32 tensor has no views
}

void test_broadcast() {
    const sgt::TensorDesc from = desc(SCRAPGPU_DTYPE_F32, 4, 2, 1, 1);
    const sgt::TensorDesc to   = desc(SCRAPGPU_DTYPE_F32, 4, 2, 3, 5);
    int64_t r[4] = {};
    CHECK(sgt::broadcast_factors(from, to, r));
    CHECK(r[0] == 1 && r[1] == 1 && r[2] == 3 && r[3] == 5);

    const sgt::TensorDesc from3 = desc(SCRAPGPU_DTYPE_F32, 4, 2, 3, 1);
    const sgt::TensorDesc bad   = desc(SCRAPGPU_DTYPE_F32, 4, 2, 2, 1); // 2 % 3 != 0
    CHECK(!sgt::broadcast_factors(from3, bad, r));

    sgt::TensorDesc zero = desc(SCRAPGPU_DTYPE_F32, 4, 2, 1, 1);
    zero.ne[2] = 0;
    CHECK(!sgt::broadcast_factors(zero, to, r)); // cannot repeat a dim of 0
}

void test_fits_int32() {
    const sgt::TensorDesc small = desc(SCRAPGPU_DTYPE_F32, 8, 4, 2, 1);
    CHECK(sgt::fits_int32(small));

    sgt::TensorDesc big_ne = desc(SCRAPGPU_DTYPE_F32, 1, 1, 1, 1);
    big_ne.ne[0] = (int64_t) INT32_MAX + 1;
    CHECK(!sgt::fits_int32(big_ne));

    sgt::TensorDesc big_nb = desc(SCRAPGPU_DTYPE_F32, 2, 1, 1, 1);
    big_nb.nb[1] = (uint64_t) INT32_MAX + 1;
    CHECK(!sgt::fits_int32(big_nb));

    // many elements (the launch grid margin): nelements > INT32_MAX - 65536 with a small footprint
    // (a broadcast view: the strides of the repeated dims are 0)
    sgt::TensorDesc many;
    many.ne[0] = 1 << 28;
    many.ne[1] = 9; // nelements = 2415919104 > INT32_MAX - 65536
    many.nb[0] = 4;
    CHECK(sgt::nelements(many) > INT32_MAX - 65536);
    CHECK(sgt::nbytes(many) == 4 + ((uint64_t) 1 << 28) * 4 - 4);
    CHECK(sgt::nbytes(many) <= (uint64_t) INT32_MAX - 65536);
    CHECK(!sgt::fits_int32(many));
    many.ne[1] = 1;
    CHECK(sgt::fits_int32(many));

    // nbytes above the limit with every ne / nb below INT32_MAX (a sparse view)
    sgt::TensorDesc sparse;
    sparse.ne[0] = 2;
    sparse.ne[1] = 2;
    sparse.nb[0] = 4;
    sparse.nb[1] = 2147420000;
    sparse.nb[2] = 4;
    sparse.nb[3] = 4;
    CHECK(sparse.nb[1] <= (uint64_t) INT32_MAX);
    CHECK(sgt::nbytes(sparse) > (uint64_t) INT32_MAX - 65536);
    CHECK(!sgt::fits_int32(sparse));

    // audit M-4: the element count is multiplied overflow-checked — dims whose product wraps to a
    // small value must not pass the int32 limits
    {
        sgt::TensorDesc wrap;
        wrap.ne[0] = 1 << 21;
        wrap.ne[1] = 1 << 21;
        wrap.ne[2] = 1 << 22; // ne0*ne1*ne2 == 2^64, wraps to 0 in an unchecked int64 product
        wrap.nb[0] = wrap.nb[1] = wrap.nb[2] = wrap.nb[3] = 4; // small strides, like a broadcast view
        CHECK(sgt::nelements(wrap) > INT32_MAX - 65536);
        CHECK(!sgt::fits_int32(wrap));
    }
    {
        // the same for the byte count of a sparse view
        sgt::TensorDesc w;
        w.ne[0] = INT32_MAX;
        w.ne[1] = INT32_MAX;
        w.nb[0] = (uint64_t) 1 << 40; // (ne0 - 1) * nb0 overflows uint64
        w.nb[1] = w.nb[2] = w.nb[3] = 4;
        CHECK(sgt::nbytes(w) == UINT64_MAX); // the span saturates instead of wrapping
        CHECK(!sgt::fits_int32(w));
    }
    {
        // large but representable counts keep their exact value
        sgt::TensorDesc big;
        big.ne[0] = INT32_MAX;
        big.ne[1] = INT32_MAX;
        CHECK(sgt::nelements(big) == (int64_t) INT32_MAX * INT32_MAX);
        CHECK(!sgt::fits_int32(big));
    }
}

void ref_fastdiv_init(uint32_t d, uint32_t & m, uint32_t & l);

void test_fastdiv() {
    // Preserve the exact magic-number ABI, including divisors outside the
    // kernel's n < 2^31 range and the power-of-two shortcut.
    uint32_t random_d = 12345;
    for (int i = 0; i < 10000; ++i) {
        random_d = random_d * 1664525u + 1013904223u;
        uint32_t actual_m, actual_l, expected_m, expected_l;
        sgt::fastdiv_init(random_d, actual_m, actual_l);
        ref_fastdiv_init(random_d, expected_m, expected_l);
        CHECK(actual_m == expected_m && actual_l == expected_l);
    }
    for (uint32_t d : {0u, 1u, 2u, 4u, 256u, 1024u, 1u << 31, UINT32_MAX}) {
        uint32_t actual_m, actual_l, expected_m, expected_l;
        sgt::fastdiv_init(d, actual_m, actual_l);
        ref_fastdiv_init(d, expected_m, expected_l);
        CHECK(actual_m == expected_m && actual_l == expected_l);
    }
    // d = 0: no division
    uint32_t m = 1, l = 1;
    sgt::fastdiv_init(0, m, l);
    CHECK(m == 0 && l == 0);

    const uint32_t divisors[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17, 31, 32, 33, 63, 64, 100, 127, 128, 255, 256, 257,
        512, 1000, 1024, 4095, 4096, 9999, 65535, 65536, 65537, 100003, 1000003, 999999937,
        (1u << 30), (1u << 30) + 1, (1u << 31) - 2, (1u << 31) - 1,
    };
    for (uint32_t d : divisors) {
        sgt::fastdiv_init(d, m, l);
        CHECK(l < 32);
        // boundary and neighbourhood values
        for (uint32_t n : { 0u, 1u, d - 1, d, d + 1, 2 * d, 2 * d + 1, 3 * d,
                            (uint32_t) INT32_MAX - 1, (uint32_t) INT32_MAX }) {
            if (n > (uint32_t) INT32_MAX) {
                continue; // the magic numbers cover [0, 2^31)
            }
            const int32_t q = sgt::fastdiv_apply(n, m, l);
            if (q != (int32_t) (n / d)) {
                std::fprintf(stderr, "fastdiv: d=%u n=%u -> %d, want %u\n", d, n, q, n / d);
            }
            CHECK(q == (int32_t) (n / d));
        }
        // dense range around 0 and a sweep over the whole range
        const uint32_t dense = d < 4096 ? 3 * d + 3 : 4096;
        for (uint32_t n = 0; n < dense; ++n) {
            CHECK(sgt::fastdiv_apply(n, m, l) == (int32_t) (n / d));
        }
        for (uint32_t k = 0; k < 2000; ++k) {
            const uint32_t n = k * 1048583u; // odd step over [0, 2^31)
            if (n >= (uint32_t) INT32_MAX + 1) {
                continue;
            }
            CHECK(sgt::fastdiv_apply(n, m, l) == (int32_t) (n / d));
        }
    }
}

void test_r32() {
    // a single slice is always eligible (the alignment applies to slice strides only)
    CHECK(sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_0, 32, 3, 1, 1)));
    CHECK(sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q8_0, 64, 16, 1, 1)));
    CHECK(sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_K, 256, 16, 1, 1)));
    CHECK(sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q6_K, 256, 16, 1, 1)));

    // several slices: the slice strides keep the 16-byte alignment
    CHECK(sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_0, 64, 32, 2, 1)));  // nb2 = 36*32 = 1152
    CHECK(!sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_0, 64, 3, 2, 1)));  // nb2 = 36*3 = 108
    CHECK(!sgt::r32_eligible(desc(SCRAPGPU_DTYPE_F32, 64, 32, 2, 1)));  // not a repackable dtype
    CHECK(!sgt::r32_eligible(desc(SCRAPGPU_DTYPE_I32, 64, 32, 1, 1)));

    sgt::TensorDesc nc = desc(SCRAPGPU_DTYPE_Q4_0, 64, 32, 1, 1);
    nc.nb[1] += 16;
    CHECK(!sgt::r32_eligible(nc)); // not contiguous

    // audit H-5: ne[0] must cover whole blocks — a partial-block row would give rowb == 0 and a
    // division by zero in the transfer
    CHECK(!sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_0, 16, 1, 1, 1)));  // 16 < block size 32
    CHECK(!sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_0, 48, 4, 1, 1)));  // 48 % 32 != 0
    CHECK(!sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q8_0, 16, 4, 1, 1)));
    CHECK(!sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_K, 128, 1, 1, 1))); // 128 < block size 256
    CHECK(sgt::r32_eligible(desc(SCRAPGPU_DTYPE_Q4_0, 32, 4, 1, 1)));   // exactly one block: fine

    // the geometry: blocks per row, block bytes, the areas
    const sgt::r32_geom g40 = sgt::r32_geometry(desc(SCRAPGPU_DTYPE_Q4_0, 64, 16, 2, 1));
    CHECK(g40.nb == 2 && g40.bs == 18 && g40.na == 2 && g40.M == 16 && g40.rowb == 36);
    CHECK(g40.area[0].src == 2 && g40.area[0].len == 16 && g40.area[0].xr == 0);
    CHECK(g40.area[1].src == 0 && g40.area[1].len == 2);
    CHECK(sgt::r32_area_before(g40, 0) == 0);
    CHECK(sgt::r32_area_before(g40, 1) == 16);

    const sgt::r32_geom g6k = sgt::r32_geometry(desc(SCRAPGPU_DTYPE_Q6_K, 512, 4, 1, 1));
    CHECK(g6k.nb == 2 && g6k.bs == 210 && g6k.na == 3);
    CHECK(g6k.area[0].src == 0 && g6k.area[0].len == 192);
    CHECK(g6k.area[1].src == 192 && g6k.area[1].len == 16 && g6k.area[1].xr == 0x80);
    CHECK(g6k.area[2].src == 208 && g6k.area[2].len == 2);
    CHECK(sgt::r32_area_before(g6k, 2) == 208);
}

// ---- the kernel ABI ---------------------------------------------------------

// the reference implementation: the math the ggml adapter ran before the structs moved to
// src/backends/opencl11/tensor_abi.h (cl11_ref / cl11_td_matrix / cl11_make_bcast)
void ref_fastdiv_init(uint32_t d, uint32_t & m, uint32_t & l) {
    m = 0;
    l = 0;
    if (d == 0) {
        return;
    }
    while (l < 32 && ((uint64_t) 1 << l) < d) {
        ++l;
    }
    m = (uint32_t) ((((uint64_t) 1 << 32) * (((uint64_t) 1 << l) - d)) / d + 1);
}

cl11_td ref_td(int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
               size_t nb0, size_t nb1, size_t nb2, size_t nb3, size_t off) {
    const int64_t ne[4] = { ne0, ne1, ne2, ne3 };
    const size_t  nb[4] = { nb0, nb1, nb2, nb3 };
    cl11_td t = {};
    for (int i = 0; i < 4; ++i) {
        t.ne[i] = (int32_t) ne[i];
        t.nb[i] = (int32_t) nb[i];
    }
    t.off = (int32_t) off;
    for (int i = 0; i < 4; ++i) {
        ref_fastdiv_init((uint32_t) ne[i], t.fm[i], t.fl[i]);
    }
    return t;
}

cl11_td ref_td_matrix(int64_t ne0, int64_t ne1, size_t nb0, size_t nb1, size_t off) {
    cl11_td t = {};
    t.ne[0] = (int32_t) ne0;
    t.ne[1] = (int32_t) ne1;
    t.ne[2] = 1;
    t.ne[3] = 1;
    t.nb[0] = (int32_t) nb0;
    t.nb[1] = (int32_t) nb1;
    t.nb[2] = (int32_t) (nb1 * (size_t) ne1);
    t.nb[3] = t.nb[2];
    t.off   = (int32_t) off;
    for (int i = 0; i < 4; ++i) {
        ref_fastdiv_init((uint32_t) t.ne[i], t.fm[i], t.fl[i]);
    }
    return t;
}

cl11_bcast ref_bcast(int64_t r2, int64_t r3, int64_t tiles, int64_t tiles2) {
    cl11_bcast b = {};
    b.r2 = (int32_t) r2;
    b.r3 = (int32_t) r3;
    ref_fastdiv_init((uint32_t) r2, b.m2, b.l2);
    ref_fastdiv_init((uint32_t) r3, b.m3, b.l3);
    b.tiles = (int32_t) tiles;
    ref_fastdiv_init((uint32_t) tiles, b.mt, b.lt);
    b.tiles2 = (int32_t) tiles2;
    ref_fastdiv_init((uint32_t) tiles2, b.mt2, b.lt2);
    return b;
}

bool same_td(const cl11_td & a, const cl11_td & b) {
    return memcmp(&a, &b, sizeof(cl11_td)) == 0;
}

void test_td_abi() {
    // the kernel ABI: `tdesc` of common.cl is 9 x int + 8 x uint, `bcast` is 12 x int, `padp` of
    // moe_ops.cl is 13 x int
    CHECK(sizeof(cl11_td) == 17 * sizeof(int32_t));
    CHECK(sizeof(cl11_bcast) == 12 * sizeof(int32_t));
    CHECK(sizeof(cl11_padp) == 13 * sizeof(int32_t));

    // a contiguous tensor
    sgt::TensorDesc d = desc(SCRAPGPU_DTYPE_F16, 4, 3, 2, 1);
    d.offset = 4096;
    CHECK(same_td(cl11_make_td(d), ref_td(4, 3, 2, 1, 2, 8, 24, 48, 4096)));

    // a transposed / padded view
    sgt::TensorDesc v = sgt::transpose(d);
    v.offset = 40;
    CHECK(same_td(cl11_make_td(v), ref_td(3, 4, 2, 1, 8, 2, 24, 48, 40)));

    // ne / nb are truncated to 32 bits
    sgt::TensorDesc h = desc(SCRAPGPU_DTYPE_F32, 1, 1, 1, 1);
    h.ne[0] = (int64_t) INT32_MAX + 5;
    h.nb[1] = (uint64_t) INT32_MAX + 9;
    CHECK(cl11_make_td(h).ne[0] == (int32_t) ((int64_t) INT32_MAX + 5));
    CHECK(cl11_make_td(h).nb[1] == (int32_t) ((uint64_t) INT32_MAX + 9));

    // cl11_td_matrix of the MUL_MAT_ID paths
    CHECK(same_td(cl11_td_matrix(128, 32, 4, 512, 96),
                  ref_td_matrix(128, 32, 4, 512, 96)));
    CHECK(same_td(cl11_td_matrix(7, 5, 2, 14, 3),
                  ref_td_matrix(7, 5, 2, 14, 3)));

    // cl11_make_bcast field by field
    const cl11_bcast b = cl11_make_bcast(4, 2, 9, 7);
    const cl11_bcast r = ref_bcast(4, 2, 9, 7);
    CHECK(memcmp(&b, &r, sizeof(cl11_bcast)) == 0);
    CHECK(b.r2 == 4 && b.r3 == 2 && b.tiles == 9 && b.tiles2 == 7);
    CHECK(sgt::fastdiv_apply(41, b.m2, b.l2) == 41 / 4);
    CHECK(sgt::fastdiv_apply(41, b.m3, b.l3) == 41 / 2);
    CHECK(sgt::fastdiv_apply(61, b.mt, b.lt) == 61 / 9);
    CHECK(sgt::fastdiv_apply(61, b.mt2, b.lt2) == 61 / 7);
}

} // namespace

int main() {
    test_dtype();
    test_contiguous();
    test_desc_of();
    test_transposed_permuted();
    test_view();
    test_reshape();
    test_broadcast();
    test_fits_int32();
    test_fastdiv();
    test_r32();
    test_td_abi();
    return 0;
}
