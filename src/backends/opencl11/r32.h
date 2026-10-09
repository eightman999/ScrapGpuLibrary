#ifndef SCRAPGPU_OPENCL11_R32_H
#define SCRAPGPU_OPENCL11_R32_H

#include "tensor/tensor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

//
// Repacked layout "R32" of the Q4_0 / Q8_0 / Q4_K / Q6_K tensors
//
// The blocks of Q8_0 (34 bytes), Q4_0 (18 bytes) and Q6_K (210 bytes) are only 2-byte aligned, and the blocks of all of them have
// a lot of fields: the natural layout needs 16-bit loads and a lot of address arithmetic on an old GPU. The repackable tensors
// (sgt::r32_eligible on their descriptor; a repacked tensor cannot have views) are therefore stored in a layout made for
// coalesced 128-bit loads; the host <-> device transfers convert. The size is the same as the natural layout. For every slice
// of the tensor (ne2 * ne3 slices of M = ne1 rows, nb = ne0 / block size blocks per row), the rows are grouped by 32
// (rs = number of rows of the group, 32 except for the last group). The bytes of a block are split into up to 3 areas, one
// after the other in the slice (area k starts at M * nb * (bytes of the areas before it per block)), see sgt::r32_area.
// Inside of an area of 16 * U bytes per block the group g starts at g * 32 * nb * 16 * U and the 16-byte unit j
// (0 <= j < U) of the block b of the row `lane` of the group is the unit (b * U + j) * rs + lane. An area of 2 bytes (the half
// scale) is an array of halves at b * rs + lane:
//   Q8_0: quants (32 bytes = 2 units, xor 0x80), d              Q4_0: nibbles (16 bytes = 1 unit), d
//   Q4_K: quants (the 128 bytes of nibbles, 8 units), header (d, dmin, 12 bytes of scales: 1 unit)
//   Q6_K: ql + qh (192 bytes, 12 units), scales (16 bytes, xor 0x80, 1 unit), d
// (see src/backends/opencl11/kernels/mul_mv_r.cl, src/backends/opencl11/kernels/mul_mv_k.cl, dqr_* in common.cl)
//

// declared without the OpenCL headers so that the host-side pack / unpack can be used (and tested) without OpenCL
struct cl11_device;
typedef struct _cl_mem * cl_mem;

// natural -> repacked: `nrows` consecutive rows (a multiple of 32, except at the end of a slice) -> the buffers of the areas
inline void cl11_rp_pack(const sgt::r32_geom & G, const uint8_t * nat, int64_t nrows, uint8_t * const * out) {
    for (int64_t g0 = 0; g0 < nrows; g0 += 32) {
        const int rs = (int) std::min<int64_t>(32, nrows - g0);
        for (int lane = 0; lane < rs; ++lane) {
            const uint8_t * row = nat + (size_t) (g0 + lane) * G.rowb;
            for (int b = 0; b < G.nb; ++b) {
                const uint8_t * p = row + (size_t) b * G.bs;
                for (int k = 0; k < G.na; ++k) {
                    const sgt::r32_area & A = G.area[k];
                    uint8_t * og = out[k] + (size_t) (g0 / 32) * 32 * G.nb * A.len;
                    if (A.len == 2) {
                        memcpy(og + ((size_t) b * rs + lane) * 2, p + A.src, 2);
                    } else {
                        const int U = A.len / 16;
                        for (int j = 0; j < U; ++j) {
                            uint8_t * o = og + (((size_t) b * U + j) * rs + lane) * 16;
                            const uint8_t * s = p + A.src + 16 * j;
                            for (int i = 0; i < 16; ++i) {
                                o[i] = s[i] ^ A.xr;
                            }
                        }
                    }
                }
            }
        }
    }
}

// repacked -> natural
inline void cl11_rp_unpack(const sgt::r32_geom & G, const uint8_t * const * in, int64_t nrows, uint8_t * nat) {
    for (int64_t g0 = 0; g0 < nrows; g0 += 32) {
        const int rs = (int) std::min<int64_t>(32, nrows - g0);
        for (int lane = 0; lane < rs; ++lane) {
            uint8_t * row = nat + (size_t) (g0 + lane) * G.rowb;
            for (int b = 0; b < G.nb; ++b) {
                uint8_t * p = row + (size_t) b * G.bs;
                for (int k = 0; k < G.na; ++k) {
                    const sgt::r32_area & A = G.area[k];
                    const uint8_t * ig = in[k] + (size_t) (g0 / 32) * 32 * G.nb * A.len;
                    if (A.len == 2) {
                        memcpy(p + A.src, ig + ((size_t) b * rs + lane) * 2, 2);
                    } else {
                        const int U = A.len / 16;
                        for (int j = 0; j < U; ++j) {
                            const uint8_t * s = ig + (((size_t) b * U + j) * rs + lane) * 16;
                            uint8_t * o = p + A.src + 16 * j;
                            for (int i = 0; i < 16; ++i) {
                                o[i] = s[i] ^ A.xr;
                            }
                        }
                    }
                }
            }
        }
    }
}

// copy `size` bytes at byte `offset` of the (natural layout of the) tensor described by `t` (t.offset is the byte offset of
// the tensor inside of `mem`) from / to the device. The conversion is done per chunk of whole groups of rows of a slice;
// a chunk that is only partly written is read, patched and written back. The caller holds c->mutex.
bool cl11_rp_transfer(cl11_device * c, cl_mem mem, const sgt::TensorDesc & t, uint8_t * host, size_t offset, size_t size, bool to_dev);

#endif
