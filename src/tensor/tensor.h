#ifndef SCRAPGPU_TENSOR_TENSOR_H
#define SCRAPGPU_TENSOR_TENSOR_H

#include "scrapgpu/tensor.h"

#include <cstddef>
#include <cstdint>
#include <limits>

// ggml-independent tensor descriptors and the pure math around them: element counts (ne), strides in
// bytes (nb), the byte offset of the first element inside of the storage and the physical layout.
//
// The definitions follow ggml (b11384), which Phase 2 was written against: ne[0] is the dim of the
// contiguous elements of the natural layout, a quantized dtype is stored in blocks of
// dtype_block_size() elements, and the stride of a dim of size 1 (and nb[0] of a single-block row) is
// not constrained. Everything is allocation-free; the functions used on the per-node / per-launch
// path of the backends are inline.

namespace sgt {

constexpr int kMaxDims = 4;

// ---- dtype ------------------------------------------------------------------

// bytes per block (per element for the unblocked dtypes), 0 for an unknown dtype
inline size_t dtype_type_size(scrapgpu_dtype dtype) {
    switch (dtype) {
        case SCRAPGPU_DTYPE_F32:  return 4;
        case SCRAPGPU_DTYPE_F16:  return 2;
        case SCRAPGPU_DTYPE_I32:  return 4;
        case SCRAPGPU_DTYPE_I64:  return 8;
        case SCRAPGPU_DTYPE_I16:  return 2;
        case SCRAPGPU_DTYPE_BF16: return 2;
        case SCRAPGPU_DTYPE_Q4_0: return 18;
        case SCRAPGPU_DTYPE_Q8_0: return 34;
        case SCRAPGPU_DTYPE_Q4_K: return 144;
        case SCRAPGPU_DTYPE_Q6_K: return 210;
        case SCRAPGPU_DTYPE_UNKNOWN: break;
    }
    return 0;
}

// elements per block (1 for the unblocked dtypes), 0 for an unknown dtype
inline size_t dtype_block_size(scrapgpu_dtype dtype) {
    switch (dtype) {
        case SCRAPGPU_DTYPE_Q4_0: return 32;
        case SCRAPGPU_DTYPE_Q8_0: return 32;
        case SCRAPGPU_DTYPE_Q4_K: return 256;
        case SCRAPGPU_DTYPE_Q6_K: return 256;
        default:                  return dtype_type_size(dtype) != 0 ? 1 : 0;
    }
}

inline bool dtype_is_quantized(scrapgpu_dtype dtype) {
    return dtype_block_size(dtype) > 1;
}

// the dtypes that can be stored in the repacked layout "R32" (groups of 32 rows, 128-bit units)
inline bool dtype_is_repackable(scrapgpu_dtype dtype) {
    return dtype == SCRAPGPU_DTYPE_Q4_0 || dtype == SCRAPGPU_DTYPE_Q8_0 ||
           dtype == SCRAPGPU_DTYPE_Q4_K || dtype == SCRAPGPU_DTYPE_Q6_K;
}

// bytes of a row of ne0 elements, the blocks rounded up; 0 on an invalid dtype / ne0 / an overflow
inline size_t row_size(scrapgpu_dtype dtype, int64_t ne0) {
    const size_t block_size = dtype_block_size(dtype);
    const size_t type_size  = dtype_type_size(dtype);
    if (ne0 < 0 || block_size == 0) {
        return 0;
    }
    const uint64_t blocks = ((uint64_t) ne0 + block_size - 1u) / block_size;
    if (blocks > std::numeric_limits<size_t>::max() / type_size) {
        return 0;
    }
    return (size_t) blocks * type_size;
}

// ---- descriptor -------------------------------------------------------------

// A tensor viewed as a block of storage: the counts, the strides in bytes, the byte offset of the
// first element and the physical layout. This is the storage part of scrapgpu_tensor_desc (the
// public C struct also carries the buffer); the C++ module works on this plain struct.
struct TensorDesc {
    scrapgpu_dtype  dtype  = SCRAPGPU_DTYPE_F32;
    int64_t         ne[4]  = { 1, 1, 1, 1 };
    uint64_t        nb[4]  = { 0, 0, 0, 0 };
    uint64_t        offset = 0;
    scrapgpu_layout layout = SCRAPGPU_LAYOUT_NATURAL;
};

// the storage part of the public descriptor (the buffer is not part of the descriptor)
inline TensorDesc desc_of(const scrapgpu_tensor_desc & s) {
    TensorDesc d;
    d.dtype  = s.dtype;
    d.layout = s.layout;
    for (int i = 0; i < kMaxDims; ++i) {
        d.ne[i] = s.ne[i];
        d.nb[i] = s.nb[i];
    }
    d.offset = s.offset;
    return d;
}

inline int64_t dim_or_one(int64_t ne) {
    return ne == 0 ? 1 : ne;
}

// a * b saturated to the int64 range instead of wrapping (audit M-4: a wrapped product passes the
// size limits — fits_int32, the kernel launch margin — or makes nbytes smaller than the tensor)
inline int64_t mul_sat_i64(int64_t a, int64_t b) {
    const int64_t lo = std::numeric_limits<int64_t>::min();
    const int64_t hi = std::numeric_limits<int64_t>::max();
    if (a > 0) {
        if (b > 0) {
            return a > hi / b ? hi : a * b;
        }
        if (b < 0) {
            return b < lo / a ? lo : a * b;
        }
        return 0;
    }
    if (a < 0) {
        if (b < 0) {
            return a < hi / b ? hi : a * b;
        }
        if (b > 0) {
            return a < lo / b ? lo : a * b;
        }
    }
    return 0;
}

inline uint64_t mul_sat_u64(uint64_t a, uint64_t b) {
    return a != 0 && b > std::numeric_limits<uint64_t>::max() / a ? std::numeric_limits<uint64_t>::max() : a * b;
}

inline uint64_t add_sat_u64(uint64_t a, uint64_t b) {
    return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max() : a + b;
}

// contiguous desc of `ne` elements (a dim <= 0 counts as 1): nb[0] = type size, nb[1] = the bytes of
// a row (whole blocks), nb[i] = nb[i-1] * ne[i-1]
inline TensorDesc make_contiguous(scrapgpu_dtype dtype, const int64_t ne[4], scrapgpu_layout layout = SCRAPGPU_LAYOUT_NATURAL) {
    TensorDesc d;
    d.dtype  = dtype;
    d.layout = layout;
    d.nb[0]  = dtype_type_size(dtype);
    for (int i = 0; i < kMaxDims; ++i) {
        d.ne[i] = ne[i];
    }
    d.nb[1] = row_size(dtype, dim_or_one(ne[0]));
    d.nb[2] = mul_sat_u64(d.nb[1], (uint64_t) dim_or_one(ne[1]));
    d.nb[3] = mul_sat_u64(d.nb[2], (uint64_t) dim_or_one(ne[2]));
    return d;
}

// the product of the dims, saturated instead of wrapping (a dim of 0 gives 0)
inline int64_t nelements(const TensorDesc & d) {
    int64_t n = 1;
    for (int i = 0; i < kMaxDims; ++i) {
        n = mul_sat_i64(n, d.ne[i]);
    }
    return n;
}

// the number of bytes a tensor touches in its storage (the offset of the last byte + 1): the ggml
// definition, so a view reports the bytes of the view, not of the allocation. 0 when a dim is <= 0.
inline uint64_t nbytes(const TensorDesc & d) {
    for (int i = 0; i < kMaxDims; ++i) {
        if (d.ne[i] <= 0) {
            return 0;
        }
    }
    const size_t block_size = dtype_block_size(d.dtype);
    const size_t type_size  = dtype_type_size(d.dtype);
    if (block_size == 0 || type_size == 0) {
        return 0;
    }
    uint64_t n;
    if (block_size == 1) {
        n = type_size;
        for (int i = 0; i < kMaxDims; ++i) {
            n = add_sat_u64(n, mul_sat_u64((uint64_t) (d.ne[i] - 1), d.nb[i]));
        }
    } else {
        n = mul_sat_u64((uint64_t) d.ne[0], d.nb[0]) / block_size;
        for (int i = 1; i < kMaxDims; ++i) {
            n = add_sat_u64(n, mul_sat_u64((uint64_t) (d.ne[i] - 1), d.nb[i]));
        }
    }
    return n;
}

// the ggml definition of contiguity: a dim of size 1 and the stride of a single-block row are not
// constrained (ggml_is_contiguous, the m = 0 variant)
inline bool is_contiguous(const TensorDesc & d) {
    const size_t block_size = dtype_block_size(d.dtype);
    uint64_t next = dtype_type_size(d.dtype);
    if (block_size == 0 || next == 0) {
        return false;
    }
    if (d.ne[0] != (int64_t) block_size && d.nb[0] != next) {
        return false;
    }
    // the expected stride product must not wrap: a wrapped `next` could equal a crafted nb
    bool overflow = next != 0 && (uint64_t) (d.ne[0] / (int64_t) block_size) > std::numeric_limits<uint64_t>::max() / next;
    next *= (uint64_t) (d.ne[0] / (int64_t) block_size);
    for (int i = 1; i < kMaxDims; ++i) {
        if (d.ne[i] != 1 && d.nb[i] != next) {
            return false;
        }
        overflow |= next != 0 && (uint64_t) d.ne[i] > std::numeric_limits<uint64_t>::max() / next;
        next *= (uint64_t) d.ne[i];
    }
    return !overflow;
}

// the strict definition of the public API: every stride is exactly the canonical contiguous stride
// (a dim of size 0 counts as 1); the contract of scrapgpu_tensor_is_contiguous
inline bool is_contiguous_strict(const TensorDesc & d) {
    const size_t row = row_size(d.dtype, dim_or_one(d.ne[0]));
    if (row == 0 || d.nb[0] != dtype_type_size(d.dtype)) {
        return false;
    }
    uint64_t expected = row;
    for (int i = 1; i < kMaxDims; ++i) {
        const int64_t dim = dim_or_one(d.ne[i]);
        if (d.nb[i] != expected) {
            return false;
        }
        if (dim > 0 && expected > std::numeric_limits<uint64_t>::max() / (uint64_t) dim) {
            return false;
        }
        expected *= (uint64_t) dim;
    }
    return true;
}

// nb[0] > nb[1]: the elements of dim 0 are not contiguous (ggml_is_transposed)
inline bool is_transposed(const TensorDesc & d) {
    return d.nb[0] > d.nb[1];
}

// the strides are not sorted ascending (ggml_is_permuted)
inline bool is_permuted(const TensorDesc & d) {
    return d.nb[0] > d.nb[1] || d.nb[1] > d.nb[2] || d.nb[2] > d.nb[3];
}

inline bool same_shape(const TensorDesc & a, const TensorDesc & b) {
    return a.ne[0] == b.ne[0] && a.ne[1] == b.ne[1] && a.ne[2] == b.ne[2] && a.ne[3] == b.ne[3];
}

// a view of `parent`: same dtype and layout, byte offset and ne / nb of the view
inline TensorDesc view(const TensorDesc & parent, uint64_t byte_offset, const int64_t ne[4], const uint64_t nb[4]) {
    TensorDesc d = parent;
    d.offset = parent.offset + byte_offset;
    for (int i = 0; i < kMaxDims; ++i) {
        d.ne[i] = ne[i];
        d.nb[i] = nb[i];
    }
    return d;
}

// the descriptor of `d` with the dims reordered: out.ne[i] = d.ne[a[i]]
inline TensorDesc permute(const TensorDesc & d, int a0, int a1, int a2, int a3) {
    const int a[4] = { a0, a1, a2, a3 };
    TensorDesc r = d;
    for (int i = 0; i < kMaxDims; ++i) {
        r.ne[i] = d.ne[a[i]];
        r.nb[i] = d.nb[a[i]];
    }
    return r;
}

inline TensorDesc transpose(const TensorDesc & d) {
    return permute(d, 1, 0, 2, 3);
}

// `to` repeats `from` in every dim: from.ne[i] > 0 and to.ne[i] = r[i] * from.ne[i]; returns the
// repeat factors r[i] (the MUL_MAT kernels divide by r[2] / r[3] to fold the batch dims)
inline bool broadcast_factors(const TensorDesc & from, const TensorDesc & to, int64_t r[4]) {
    for (int i = 0; i < kMaxDims; ++i) {
        if (from.ne[i] <= 0 || to.ne[i] % from.ne[i] != 0) {
            return false;
        }
        r[i] = to.ne[i] / from.ne[i];
    }
    return true;
}

// the offsets and strides are passed to the kernels as 32-bit ints (with a margin: they compute
// group_index * work_group_size + local_id in 32 bits)
inline bool fits_int32(const TensorDesc & d) {
    for (int i = 0; i < kMaxDims; ++i) {
        if (d.ne[i] > INT32_MAX || d.nb[i] > (uint64_t) INT32_MAX) {
            return false;
        }
    }
    return nelements(d) <= INT32_MAX - 65536 && nbytes(d) <= (uint64_t) INT32_MAX - 65536;
}

// ---- fast division ----------------------------------------------------------

// magic numbers m, l such that n / d == (mulhi(n, m) + n) >> l for 0 <= n < 2^31 (FDIV in
// src/backends/opencl11/kernels/common.cl: the integer division of 32-bit ints is very slow on the
// old GPUs)
inline void fastdiv_init(uint32_t d, uint32_t & m, uint32_t & l) {
    m = 0;
    l = 0;
    if (d == 0) {
        return;
    }
    if (d == 1) {
        m = 1;
        return;
    }
#if defined(__GNUC__) || defined(__clang__)
    // The launch descriptors repeatedly use the same small/power-of-two
    // dimensions. ceil(log2(d)) needs no loop or newer x86 instruction set.
    l = 32u - static_cast<uint32_t>(__builtin_clz(d - 1));
#else
    while (l < 32 && ((uint64_t) 1 << l) < d) {
        ++l;
    }
#endif
    if ((d & (d - 1)) == 0) {
        m = 1;
        return;
    }
    m = (uint32_t) ((((uint64_t) 1 << 32) * (((uint64_t) 1 << l) - d)) / d + 1);
}

// n / d for 0 <= n < 2^31, given the magic numbers of d: FDIV_RAW of the kernels. l >= 32 (d > 2^31,
// outside of the range the descriptors allow) means n / d == 0.
inline int32_t fastdiv_apply(uint32_t n, uint32_t m, uint32_t l) {
    if (l >= 32) {
        return 0;
    }
    const uint32_t hi = (uint32_t) (((uint64_t) n * m) >> 32);
    return (int32_t) ((hi + n) >> l);
}

// ---- R32: the repacked layout of the quantized dtypes -----------------------

// the tensor can be stored in the R32 layout. Note: a view of an R32 tensor cannot be addressed (the
// kernels and the transfers know the layout of the whole tensor only); whether a descriptor is a
// view is a property of the caller.
inline bool r32_eligible(const TensorDesc & d) {
    if (!dtype_is_repackable(d.dtype) || !is_contiguous(d)) {
        return false;
    }
    // the row must hold whole blocks: a partial-block row gives a geometry with nb == 0 and a
    // division by zero in the transfer (audit H-5)
    const size_t block_size = dtype_block_size(d.dtype);
    if (d.ne[0] <= 0 || (uint64_t) d.ne[0] % block_size != 0) {
        return false;
    }
    // the slices must keep the 16-byte alignment of the 128-bit loads (a single slice has no
    // slice strides to address)
    return (d.ne[2] == 1 && d.ne[3] == 1) || (d.nb[2] % 16 == 0 && d.nb[3] % 16 == 0);
}

// The bytes of a block of the natural layout are split into up to 3 areas, one after the other in
// the repacked slice (area k starts at M * nb * (the bytes per block of the areas before it)). Inside
// of an area of 16 * U bytes per block the 16-byte unit j of the block b of the row `lane` of the
// group g is the unit (b * U + j) * rs + lane of the group; an area of 2 bytes (a half scale) is an
// array of halves at b * rs + lane:
//   Q8_0: quants (32 bytes = 2 units, xor 0x80), d              Q4_0: nibbles (16 bytes = 1 unit), d
//   Q4_K: quants (the 128 bytes of nibbles, 8 units), header (d, dmin, 12 bytes of scales: 1 unit)
//   Q6_K: ql + qh (192 bytes, 12 units), scales (16 bytes, xor 0x80, 1 unit), d
// (see src/backends/opencl11/kernels/mul_mv_r.cl, mul_mv_k.cl, dqr_* in common.cl)

struct r32_area {
    int32_t  src; // first byte of the area in the natural block
    int32_t  len; // bytes per block (2: halves, otherwise a multiple of 16)
    uint8_t  xr;  // xor mask applied to the bytes
};

struct r32_geom {
    int32_t   nb;   // blocks per row
    int32_t   bs;   // bytes of a block in the natural layout
    int32_t   na;   // number of areas
    r32_area  area[3];
    int64_t   M;    // rows per slice
    size_t    rowb; // bytes of a row in the natural layout
};

// the R32 geometry of a repackable tensor (ne0 blocks per row, ne1 rows per slice)
r32_geom r32_geometry(const TensorDesc & d);

// bytes per block of the areas before the area k
inline size_t r32_area_before(const r32_geom & g, int k) {
    size_t n = 0;
    for (int i = 0; i < k; ++i) {
        n += (size_t) g.area[i].len;
    }
    return n;
}

// ---- reshaping --------------------------------------------------------------

// the desc of `d` reshaped to `ne`: a contiguous desc is reshaped to the canonical contiguous
// strides; fails for a non-contiguous or R32 desc, a different element count, or a quantized dtype
// whose new row would not hold whole blocks
bool reshape(const TensorDesc & d, const int64_t ne[4], TensorDesc & out);

} // namespace sgt

#endif
