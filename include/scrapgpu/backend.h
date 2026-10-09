#ifndef SCRAPGPU_BACKEND_H
#define SCRAPGPU_BACKEND_H

#include "scrapgpu/op.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scrapgpu_plan_node {
    size_t struct_size;
    scrapgpu_op_desc op;
    uint32_t output_use_count;
} scrapgpu_plan_node;

typedef struct scrapgpu_plan_desc {
    size_t struct_size;
    const scrapgpu_plan_node * nodes;
    size_t node_count;
} scrapgpu_plan_desc;

typedef struct scrapgpu_backend_vtable {
    size_t struct_size;
    const char * name;
    void * userdata;
    size_t (*get_device_count)(void * userdata);
    scrapgpu_status (*get_device)(void * userdata, size_t index, scrapgpu_device_t ** out_device);
    scrapgpu_status (*device_get_info)(scrapgpu_device_t * device, scrapgpu_device_info * info);
    scrapgpu_status (*buffer_alloc)(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer);
    scrapgpu_status (*buffer_free)(scrapgpu_buffer_t * buffer);
    scrapgpu_status (*buffer_write)(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking);
    scrapgpu_status (*buffer_read)(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking);
    scrapgpu_status (*buffer_copy)(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking);
    scrapgpu_status (*buffer_fill)(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking);
    scrapgpu_status (*stream_sync)(scrapgpu_stream_t * stream);
    scrapgpu_status (*op_supported)(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason);
    scrapgpu_status (*plan_build)(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan);
    scrapgpu_status (*plan_execute)(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan);
    scrapgpu_status (*plan_free)(scrapgpu_plan_t * plan);
} scrapgpu_backend_vtable;

SCRAPGPU_API scrapgpu_status scrapgpu_register_backend(const scrapgpu_backend_vtable * backend);
SCRAPGPU_API void scrapgpu_unregister_all_backends(void);
SCRAPGPU_API scrapgpu_status scrapgpu_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan);
SCRAPGPU_API scrapgpu_status scrapgpu_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan);
SCRAPGPU_API scrapgpu_status scrapgpu_plan_free(scrapgpu_plan_t * plan);

#ifdef __cplusplus
}
#endif

#endif
