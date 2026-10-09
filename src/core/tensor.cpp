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
    const sgt::TensorDesc d = sgt::desc_of(*desc);
    return sgt::dtype_is_repackable(d.dtype) && sgt::is_contiguous_strict(d) && (d.nb[2] % 16u) == 0u ? 1 : 0;
}

}
