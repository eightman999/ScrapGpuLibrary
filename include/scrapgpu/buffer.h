#ifndef SCRAPGPU_BUFFER_H
#define SCRAPGPU_BUFFER_H

#include "scrapgpu/device.h"

#ifdef __cplusplus
extern "C" {
#endif

SCRAPGPU_API scrapgpu_status scrapgpu_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_free(scrapgpu_buffer_t * buffer);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking);
SCRAPGPU_API scrapgpu_status scrapgpu_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking);

#ifdef __cplusplus
}
#endif

#endif
