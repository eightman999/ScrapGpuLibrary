#ifndef SCRAPGPU_OPENCL11_CONTEXT_H
#define SCRAPGPU_OPENCL11_CONTEXT_H

#include "device.h"

struct cl11_device;

const char * cl11_err_str(cl_int err);

// creates the cl_context and the in-order command queue of the device (context, queue, context_ready, failed and the
// profiling / launch log flags of dev->exec), the caller holds dev->mutex
bool cl11_ensure_runtime_context(cl11_device * dev);

#endif
