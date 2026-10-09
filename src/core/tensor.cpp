#include "scrapgpu/tensor.h"

#include <limits>

namespace {
constexpr int64_t kDefaultDim = 1;

bool dtype_info(scrapgpu_dtype dtype, size_t * block_size, size_t * type_size) {
    switch (dtype) {
        case SCRAPGPU_DTYPE_F32: *block_size = 1; *type_size = 4; return true;
        case SCRAPGPU_DTYPE_F16: *block_size = 1; *type_size = 2; return true;
        case SCRAPGPU_DTYPE_I32: *block_size = 1; *type_size = 4; return true;
        case SCRAPGPU_DTYPE_I64: *block_size = 1; *type_size = 8; return true;
        case SCRAPGPU_DTYPE_I16: *block_size = 1; *type_size = 2; return true;
        case SCRAPGPU_DTYPE_BF16: *block_size = 1; *type_size = 2; return true;
        case SCRAPGPU_DTYPE_Q4_0: *block_size = 32; *type_size = 18; return true;
        case SCRAPGPU_DTYPE_Q8_0: *block_size = 32; *type_size = 34; return true;
        case SCRAPGPU_DTYPE_Q4_K: *block_size = 256; *type_size = 144; return true;
        case SCRAPGPU_DTYPE_Q6_K: *block_size = 256; *type_size = 210; return true;
    }
    return false;
}

int64_t dim_or_one(const scrapgpu_tensor_desc * desc, int index) {
    return desc->ne[index] == 0 ? kDefaultDim : desc->ne[index];
}
}

extern "C" {

size_t scrapgpu_dtype_block_size(scrapgpu_dtype dtype) {
    size_t block_size = 0;
    size_t type_size = 0;
    return dtype_info(dtype, &block_size, &type_size) ? block_size : 0;
}

size_t scrapgpu_dtype_type_size(scrapgpu_dtype dtype) {
    size_t block_size = 0;
    size_t type_size = 0;
    return dtype_info(dtype, &block_size, &type_size) ? type_size : 0;
}

size_t scrapgpu_tensor_row_size(scrapgpu_dtype dtype, int64_t ne0) {
    size_t block_size = 0;
    size_t type_size = 0;
    if (ne0 < 0 || !dtype_info(dtype, &block_size, &type_size)) {
        return 0;
    }
    const uint64_t n = static_cast<uint64_t>(ne0);
    const uint64_t blocks = (n + block_size - 1u) / block_size;
    if (blocks > std::numeric_limits<size_t>::max() / type_size) {
        return 0;
    }
    return static_cast<size_t>(blocks) * type_size;
}

int scrapgpu_tensor_is_contiguous(const scrapgpu_tensor_desc * desc) {
    if (desc == nullptr) {
        return 0;
    }
    const size_t row = scrapgpu_tensor_row_size(desc->dtype, dim_or_one(desc, 0));
    if (row == 0 || desc->nb[0] != scrapgpu_dtype_type_size(desc->dtype)) {
        return 0;
    }
    uint64_t expected = static_cast<uint64_t>(row);
    for (int i = 1; i < 4; ++i) {
        const int64_t dim = dim_or_one(desc, i);
        if (desc->nb[i] != expected) {
            return 0;
        }
        if (dim > 0 && expected > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
            return 0;
        }
        expected *= static_cast<uint64_t>(dim);
    }
    return 1;
}

int scrapgpu_tensor_r32_eligible(const scrapgpu_tensor_desc * desc) {
    if (desc == nullptr) {
        return 0;
    }
    switch (desc->dtype) {
        case SCRAPGPU_DTYPE_Q4_0:
        case SCRAPGPU_DTYPE_Q8_0:
        case SCRAPGPU_DTYPE_Q4_K:
        case SCRAPGPU_DTYPE_Q6_K:
            break;
        default:
            return 0;
    }
    return scrapgpu_tensor_is_contiguous(desc) && (desc->nb[2] % 16u) == 0u;
}

}
