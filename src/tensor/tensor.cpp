#include "tensor/tensor.h"

namespace sgt {

r32_geom r32_geometry(const TensorDesc & d) {
    r32_geom g = {};
    const size_t block_size = dtype_block_size(d.dtype);
    g.M    = d.ne[1];
    g.bs   = (int32_t) dtype_type_size(d.dtype);
    g.nb   = block_size == 0 ? 0 : (int32_t) (d.ne[0] / (int64_t) block_size);
    g.rowb = (size_t) g.nb * (size_t) g.bs;
    switch (d.dtype) {
        case SCRAPGPU_DTYPE_Q8_0: g.na = 2; g.area[0] = {  2,  32, 0x80 }; g.area[1] = {   0,   2, 0 }; break;
        case SCRAPGPU_DTYPE_Q4_0: g.na = 2; g.area[0] = {  2,  16, 0    }; g.area[1] = {   0,   2, 0 }; break;
        case SCRAPGPU_DTYPE_Q4_K: g.na = 2; g.area[0] = { 16, 128, 0    }; g.area[1] = {   0,  16, 0 }; break;
        default:                  g.na = 3; g.area[0] = {  0, 192, 0    }; g.area[1] = { 192, 16, 0x80 }; g.area[2] = { 208, 2, 0 }; break;
    }
    return g;
}

bool reshape(const TensorDesc & d, const int64_t ne[4], TensorDesc & out) {
    if (d.layout != SCRAPGPU_LAYOUT_NATURAL || !is_contiguous(d)) {
        return false;
    }
    const size_t block_size = dtype_block_size(d.dtype);
    if (block_size == 0 || (block_size > 1 && ne[0] % (int64_t) block_size != 0)) {
        return false;
    }
    int64_t n = 1;
    for (int i = 0; i < kMaxDims; ++i) {
        n *= ne[i];
    }
    if (n != nelements(d)) {
        return false;
    }
    out = make_contiguous(d.dtype, ne, d.layout);
    out.offset = d.offset;
    return true;
}

} // namespace sgt
