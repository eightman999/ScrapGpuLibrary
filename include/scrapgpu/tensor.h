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

// Host <-> device transfer of the first `size` natural-layout bytes of the tensor described by
// `desc` (desc->offset locates the tensor inside its buffer). The host buffer holds the natural
// layout for both directions: a tensor whose desc->layout is SCRAPGPU_LAYOUT_R32 is packed /
// unpacked by the backend while it is copied. Transfers are enqueued on the stream; the host buffer
// may be reused / read as soon as the call returns (the transfer is performed synchronously wrt the
// host pointer). The tensor must be contiguous (sgt::is_contiguous / scrapgpu_tensor_is_contiguous);
// raw strided transfers are done with scrapgpu_buffer_write / scrapgpu_buffer_read.
SCRAPGPU_API scrapgpu_status scrapgpu_tensor_upload(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, const void * host, size_t size);
SCRAPGPU_API scrapgpu_status scrapgpu_tensor_download(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, void * host, size_t size);

#ifdef __cplusplus
}
#endif

#endif
