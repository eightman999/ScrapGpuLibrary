// host-only tests of the R32 repacked layout: the pack / unpack of src/backends/opencl11/r32.h.
// No ggml, no OpenCL, no assert(): CHECK stays active in Release builds.

#include "opencl11/r32.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// repacked bytes of a slice (the same size as the natural slice: the areas hold every byte of the block)
size_t slice_bytes(const sgt::r32_geom & g) {
    return (size_t) g.M * (size_t) g.nb * (size_t) g.bs;
}

// inside of a slice, area k starts at M * nb * (the bytes per block of the areas before it)
void area_ptrs(const sgt::r32_geom & g, uint8_t * rep, int64_t slice, uint8_t ** out) {
    for (int k = 0; k < g.na; ++k) {
        out[k] = rep + (size_t) slice * slice_bytes(g) + (size_t) g.M * (size_t) g.nb * sgt::r32_area_before(g, k);
    }
}

void area_ptrs(const sgt::r32_geom & g, const uint8_t * rep, int64_t slice, const uint8_t ** out) {
    for (int k = 0; k < g.na; ++k) {
        out[k] = rep + (size_t) slice * slice_bytes(g) + (size_t) g.M * (size_t) g.nb * sgt::r32_area_before(g, k);
    }
}

// pack / unpack every slice in one call per slice, as cl11_rp_transfer does for whole tensors
void pack_tensor(const sgt::r32_geom & g, const uint8_t * nat, int64_t slices, uint8_t * rep) {
    for (int64_t s = 0; s < slices; ++s) {
        uint8_t * ap[3] = {};
        area_ptrs(g, rep, s, ap);
        cl11_rp_pack(g, nat + (size_t) s * (size_t) g.M * g.rowb, g.M, ap);
    }
}

void unpack_tensor(const sgt::r32_geom & g, const uint8_t * rep, int64_t slices, uint8_t * nat) {
    for (int64_t s = 0; s < slices; ++s) {
        const uint8_t * ap[3] = {};
        area_ptrs(g, rep, s, ap);
        cl11_rp_unpack(g, ap, g.M, nat + (size_t) s * (size_t) g.M * g.rowb);
    }
}

void test_identity(scrapgpu_dtype dtype, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
    const sgt::TensorDesc d = desc(dtype, n0, n1, n2, n3);
    CHECK(sgt::r32_eligible(d));
    const sgt::r32_geom g = sgt::r32_geometry(d);
    const int64_t slices = n2 * n3;
    const size_t nat_bytes = (size_t) g.M * g.rowb * (size_t) slices;
    std::vector<uint8_t> nat(nat_bytes), rep(nat_bytes), back(nat_bytes);
    for (size_t i = 0; i < nat_bytes; ++i) {
        nat[i] = (uint8_t) (i * 7 + 13);
    }
    pack_tensor(g, nat.data(), slices, rep.data());
    unpack_tensor(g, rep.data(), slices, back.data());
    CHECK(memcmp(nat.data(), back.data(), nat_bytes) == 0);
}

// the transfer converts whole groups of rows per chunk: packing [0, rA) and [rA, M) separately must
// give the same bytes as packing the slice at once
void test_chunked(scrapgpu_dtype dtype, int64_t n0, int64_t n1, int64_t groups_a) {
    const sgt::TensorDesc d = desc(dtype, n0, n1, 1, 1);
    const sgt::r32_geom g = sgt::r32_geometry(d);
    const int64_t rA = groups_a * 32;
    CHECK(rA > 0 && rA < g.M);
    const size_t nat_bytes = (size_t) g.M * g.rowb;
    std::vector<uint8_t> nat(nat_bytes), rep1(nat_bytes), rep2(nat_bytes);
    for (size_t i = 0; i < nat_bytes; ++i) {
        nat[i] = (uint8_t) (i * 11 + 5);
    }
    pack_tensor(g, nat.data(), 1, rep1.data());
    {
        uint8_t * ap[3] = {};
        area_ptrs(g, rep2.data(), 0, ap);
        cl11_rp_pack(g, nat.data(), rA, ap);
        for (int k = 0; k < g.na; ++k) {
            ap[k] += (size_t) rA * (size_t) g.nb * (size_t) g.area[k].len; // group (rA / 32) of the area
        }
        cl11_rp_pack(g, nat.data() + (size_t) rA * g.rowb, g.M - rA, ap);
    }
    CHECK(memcmp(rep1.data(), rep2.data(), nat_bytes) == 0);
}

// a small hand-checkable case: Q4_0, 2 blocks per row, one group of 3 rows. Per the layout comment of
// src/tensor/tensor.h the 16-byte unit j of the block b of the row `lane` is the unit (b * U + j) * rs
// + lane of the group (area 0: the nibbles, U = 1) and the half scales are halves at b * rs + lane.
void test_layout_q4_0() {
    const sgt::TensorDesc d = desc(SCRAPGPU_DTYPE_Q4_0, 64, 3, 1, 1); // nb = 2, M = 3, one group (rs = 3)
    const sgt::r32_geom g = sgt::r32_geometry(d);
    CHECK(g.nb == 2 && g.M == 3 && g.na == 2);
    CHECK(g.area[0].src == 2 && g.area[0].len == 16 && g.area[0].xr == 0); // nibbles
    CHECK(g.area[1].src == 0 && g.area[1].len == 2);                       // d

    // every byte of the natural tensor is distinct and encodes its (row, block, byte) position
    uint8_t nat[3 * 36] = {};
    for (int lane = 0; lane < 3; ++lane) {
        for (int b = 0; b < 2; ++b) {
            for (int i = 0; i < 18; ++i) {
                nat[lane * 36 + b * 18 + i] = (uint8_t) (1 + lane * 40 + b * 20 + i);
            }
        }
    }

    // the expected repacked bytes: [nibbles of (b, lane) at unit b * 3 + lane][d halves at b * 3 + lane]
    uint8_t expect[3 * 36] = {};
    for (int lane = 0; lane < 3; ++lane) {
        for (int b = 0; b < 2; ++b) {
            for (int i = 0; i < 16; ++i) {
                expect[(b * 3 + lane) * 16 + i] = (uint8_t) (1 + lane * 40 + b * 20 + 2 + i);
            }
            for (int j = 0; j < 2; ++j) {
                expect[96 + (b * 3 + lane) * 2 + j] = (uint8_t) (1 + lane * 40 + b * 20 + j);
            }
        }
    }

    uint8_t rep[sizeof(nat)] = {};
    uint8_t * ap[3] = {};
    area_ptrs(g, rep, 0, ap);
    CHECK(ap[0] == rep && ap[1] == rep + 96); // area 1 starts at M * nb * 16
    cl11_rp_pack(g, nat, g.M, ap);
    CHECK(memcmp(rep, expect, sizeof(rep)) == 0);

    uint8_t back[sizeof(nat)] = {};
    const uint8_t * cap[3] = { rep, rep + 96, nullptr };
    cl11_rp_unpack(g, cap, g.M, back);
    CHECK(memcmp(back, nat, sizeof(back)) == 0);
}

// the xor masks: the Q8_0 quants are stored xor 0x80 (2 units per block), the half scale is not masked
void test_layout_q8_0() {
    const sgt::TensorDesc d = desc(SCRAPGPU_DTYPE_Q8_0, 32, 2, 1, 1); // nb = 1, M = 2, one group (rs = 2)
    const sgt::r32_geom g = sgt::r32_geometry(d);
    CHECK(g.nb == 1 && g.M == 2 && g.na == 2);
    CHECK(g.area[0].src == 2 && g.area[0].len == 32 && g.area[0].xr == 0x80);
    CHECK(g.area[1].src == 0 && g.area[1].len == 2 && g.area[1].xr == 0);

    uint8_t nat[2 * 34] = {};
    for (int lane = 0; lane < 2; ++lane) {
        for (int i = 0; i < 34; ++i) {
            nat[lane * 34 + i] = (uint8_t) (1 + lane * 50 + i);
        }
    }

    // area 0: unit j of block 0 of the row `lane` is the unit j * 2 + lane, every byte xor 0x80
    uint8_t expect[2 * 34] = {};
    for (int lane = 0; lane < 2; ++lane) {
        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 16; ++i) {
                expect[(j * 2 + lane) * 16 + i] = (uint8_t) ((1 + lane * 50 + 2 + 16 * j + i) ^ 0x80);
            }
        }
        for (int j = 0; j < 2; ++j) {
            expect[64 + lane * 2 + j] = (uint8_t) (1 + lane * 50 + j); // d halves, not masked
        }
    }

    uint8_t rep[sizeof(nat)] = {};
    uint8_t * ap[3] = {};
    area_ptrs(g, rep, 0, ap);
    CHECK(ap[0] == rep && ap[1] == rep + 64);
    cl11_rp_pack(g, nat, g.M, ap);
    CHECK(memcmp(rep, expect, sizeof(rep)) == 0);
}

} // namespace

int main() {
    // pack then unpack is the identity (rows that are not a multiple of 32, several slices)
    test_identity(SCRAPGPU_DTYPE_Q4_0, 64, 33, 1, 1);
    test_identity(SCRAPGPU_DTYPE_Q4_0, 32, 40, 2, 1);  // ne1 = 32 + 8 (nb2 = 720 % 16 == 0)
    test_identity(SCRAPGPU_DTYPE_Q4_0, 32, 64, 2, 2);  // ne2 * ne3 = 4 slices
    test_identity(SCRAPGPU_DTYPE_Q4_0, 32, 1, 1, 1);   // a single row (rs = 1)
    test_identity(SCRAPGPU_DTYPE_Q8_0, 64, 40, 2, 1);
    test_identity(SCRAPGPU_DTYPE_Q8_0, 32, 33, 1, 1);
    test_identity(SCRAPGPU_DTYPE_Q4_K, 256, 33, 2, 1);
    test_identity(SCRAPGPU_DTYPE_Q4_K, 512, 16, 2, 2);
    test_identity(SCRAPGPU_DTYPE_Q6_K, 256, 72, 3, 1); // ne1 = 64 + 8 (nb2 = 15120 % 16 == 0)
    test_identity(SCRAPGPU_DTYPE_Q6_K, 512, 32, 1, 1);

    // whole groups of rows per chunk (the partial groups of cl11_rp_transfer)
    test_chunked(SCRAPGPU_DTYPE_Q4_0, 64, 100, 1); // [0, 32) + [32, 100)
    test_chunked(SCRAPGPU_DTYPE_Q6_K, 256, 100, 2); // [0, 64) + [64, 100)

    test_layout_q4_0();
    test_layout_q8_0();
    return 0;
}
