#ifndef SCRAPGPU_TENSOR_H
#define SCRAPGPU_TENSOR_H

#include "scrapgpu/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum scrapgpu_dtype {
    SCRAPGPU_DTYPE_F32 = 0,
    SCRAPGPU_DTYPE_F16 = 1,
    SCRAPGPU_DTYPE_I32 = 2,
    SCRAPGPU_DTYPE_I64 = 3,
    SCRAPGPU_DTYPE_I16 = 4,
    SCRAPGPU_DTYPE_BF16 = 5,
    SCRAPGPU_DTYPE_Q4_0 = 16,
    SCRAPGPU_DTYPE_Q8_0 = 17,
    SCRAPGPU_DTYPE_Q4_K = 32,
    SCRAPGPU_DTYPE_Q6_K = 33,
} scrapgpu_dtype;

typedef enum scrapgpu_layout {
    SCRAPGPU_LAYOUT_NATURAL = 0,
    SCRAPGPU_LAYOUT_R32 = 1,
} scrapgpu_layout;

typedef struct scrapgpu_tensor_desc {
    size_t struct_size;
    scrapgpu_dtype dtype;
    int64_t ne[4];
    uint64_t nb[4];
    scrapgpu_buffer_t * buffer;
    uint64_t offset;
    scrapgpu_layout layout;
} scrapgpu_tensor_desc;

SCRAPGPU_API size_t scrapgpu_dtype_block_size(scrapgpu_dtype dtype);
SCRAPGPU_API size_t scrapgpu_dtype_type_size(scrapgpu_dtype dtype);
SCRAPGPU_API size_t scrapgpu_tensor_row_size(scrapgpu_dtype dtype, int64_t ne0);
SCRAPGPU_API int scrapgpu_tensor_is_contiguous(const scrapgpu_tensor_desc * desc);
SCRAPGPU_API int scrapgpu_tensor_r32_eligible(const scrapgpu_tensor_desc * desc);

#ifdef __cplusplus
}
#endif

#endif
