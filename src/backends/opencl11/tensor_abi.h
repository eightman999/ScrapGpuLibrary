#ifndef SCRAPGPU_OPENCL11_TENSOR_ABI_H
#define SCRAPGPU_OPENCL11_TENSOR_ABI_H

#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>

// The kernel argument structs of the CL11 backend. They are passed to the kernels by value and must
// stay byte-identical to the structs of src/backends/opencl11/kernels/*.cl. This header is OpenCL
// free so that the host-side tests can build the structs without OpenCL headers.

// must match `tdesc` in src/backends/opencl11/kernels/common.cl (9 x int + 8 x uint)
struct cl11_td {
    int32_t  ne[4];
    int32_t  nb[4];
    int32_t  off;
    uint32_t fm[4]; // fast division by ne[i]
    uint32_t fl[4];
};
static_assert(sizeof(cl11_td) == 17 * sizeof(int32_t), "cl11_td layout");

// must match `bcast` in src/backends/opencl11/kernels/common.cl
struct cl11_bcast {
    int32_t  r2, r3;
    uint32_t m2, l2, m3, l3;
    int32_t  tiles;
    uint32_t mt, lt;
    int32_t  tiles2;
    uint32_t mt2, lt2;
};
static_assert(sizeof(cl11_bcast) == 12 * sizeof(int32_t), "cl11_bcast layout");

// must match `padp` in src/backends/opencl11/kernels/moe_ops.cl
struct cl11_padp {
    int32_t lp[4];
    int32_t rp[4];
    int32_t w[4];
    int32_t circ;
};
static_assert(sizeof(cl11_padp) == 13 * sizeof(int32_t), "cl11_padp layout");

// the kernel descriptor of a tensor (the 32-bit truncation of the descriptor is checked by
// sgt::fits_int32 before the op is launched)
inline cl11_td cl11_make_td(const sgt::TensorDesc & d) {
    cl11_td t = {};
    for (int i = 0; i < sgt::kMaxDims; ++i) {
        t.ne[i] = (int32_t) d.ne[i];
        t.nb[i] = (int32_t) d.nb[i];
        sgt::fastdiv_init((uint32_t) d.ne[i], t.fm[i], t.fl[i]);
    }
    t.off = (int32_t) d.offset;
    return t;
}

// broadcast / tiling constants of the mul_mat kernels: r2 = ne12 / ne02, r3 = ne13 / ne03
inline cl11_bcast cl11_make_bcast(int64_t r2, int64_t r3, int64_t tiles, int64_t tiles2) {
    cl11_bcast b = {};
    b.r2 = (int32_t) r2;
    b.r3 = (int32_t) r3;
    sgt::fastdiv_init((uint32_t) r2, b.m2, b.l2);
    sgt::fastdiv_init((uint32_t) r3, b.m3, b.l3);
    b.tiles = (int32_t) tiles;
    sgt::fastdiv_init((uint32_t) tiles, b.mt, b.lt);
    b.tiles2 = (int32_t) tiles2;
    sgt::fastdiv_init((uint32_t) tiles2, b.mt2, b.lt2);
    return b;
}

// the descriptor of the 2-D tensor [ne0, ne1] at the byte offset `off` (the slices of an experts
// tensor, the scratch columns of MUL_MAT_ID)
inline cl11_td cl11_td_matrix(int64_t ne0, int64_t ne1, size_t nb0, size_t nb1, size_t off) {
    sgt::TensorDesc d;
    d.ne[0] = ne0;
    d.ne[1] = ne1;
    d.nb[0] = (uint64_t) nb0;
    d.nb[1] = (uint64_t) nb1;
    d.nb[2] = (uint64_t) nb1 * (uint64_t) ne1;
    d.nb[3] = d.nb[2];
    d.offset = (uint64_t) off;
    return cl11_make_td(d);
}

#endif
