#include "scrapgpu/tensor.h"

#include "tensor/tensor.h"

extern "C" {

size_t scrapgpu_dtype_block_size(scrapgpu_dtype dtype) {
    return sgt::dtype_block_size(dtype);
}

size_t scrapgpu_dtype_type_size(scrapgpu_dtype dtype) {
    return sgt::dtype_type_size(dtype);
}

size_t scrapgpu_tensor_row_size(scrapgpu_dtype dtype, int64_t ne0) {
    return sgt::row_size(dtype, ne0);
}

int scrapgpu_tensor_is_contiguous(const scrapgpu_tensor_desc * desc) {
    if (desc == nullptr) {
        return 0;
    }
    return sgt::is_contiguous_strict(sgt::desc_of(*desc)) ? 1 : 0;
}

int scrapgpu_tensor_r32_eligible(const scrapgpu_tensor_desc * desc) {
    if (desc == nullptr) {
        return 0;
    }
    // the same answer as the backend rule (sgt::r32_eligible): a repackable dtype, contiguous in the
    // ggml sense, and ne2 * ne3 == 1 or 16-byte aligned slice strides
    return sgt::r32_eligible(sgt::desc_of(*desc)) ? 1 : 0;
}

}
