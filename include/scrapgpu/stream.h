#ifndef SCRAPGPU_STREAM_H
#define SCRAPGPU_STREAM_H

#include "scrapgpu/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// In-order execution queue of a device. A stream may share the device's own queue (OpenCL 1.1 has
// one queue per context anyway). Work is enqueued asynchronously; scrapgpu_stream_sync waits for
// everything submitted so far. One stream is used from one thread at a time.
SCRAPGPU_API scrapgpu_status scrapgpu_stream_create(scrapgpu_device_t * device, scrapgpu_stream_t ** out_stream);
SCRAPGPU_API scrapgpu_status scrapgpu_stream_free(scrapgpu_stream_t * stream);
SCRAPGPU_API scrapgpu_status scrapgpu_stream_sync(scrapgpu_stream_t * stream);

#ifdef __cplusplus
}
#endif

#endif
