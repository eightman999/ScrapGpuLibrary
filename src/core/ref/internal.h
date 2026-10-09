#ifndef SCRAPGPU_CORE_REF_INTERNAL_H
#define SCRAPGPU_CORE_REF_INTERNAL_H

#include "scrapgpu/reference.h"

#include <cstdint>
#include <cstring>

namespace scrapgpu {
namespace ref {

// IEEE 754 half conversion (round to nearest even), ggml-compatible.
float half_to_float(uint16_t h);
uint16_t float_to_half(float f);

inline uint16_t load_u16(const uint8_t * p) {
    uint16_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline void store_u16(uint8_t * p, uint16_t v) {
    std::memcpy(p, &v, sizeof(v));
}

inline bool dtype_is_quantized(scrapgpu_dtype dtype) {
    switch (dtype) {
        case SCRAPGPU_DTYPE_Q4_0:
        case SCRAPGPU_DTYPE_Q8_0:
        case SCRAPGPU_DTYPE_Q4_K:
        case SCRAPGPU_DTYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

// Dequantize n elements (a multiple of the block size) of natural-layout blocks
// starting at row. The row must be contiguous (nb0 == type_size).
scrapgpu_status dequantize_row(scrapgpu_dtype dtype, const uint8_t * row, int64_t n, float * dst);

// Quantize n elements (a multiple of the block size) into natural-layout blocks.
scrapgpu_status quantize_row(scrapgpu_dtype dtype, const float * src, uint8_t * dst, int64_t n);

scrapgpu_status ref_invalid(const char * message);
scrapgpu_status ref_unsupported(const char * message);

} // namespace ref
} // namespace scrapgpu

#endif
