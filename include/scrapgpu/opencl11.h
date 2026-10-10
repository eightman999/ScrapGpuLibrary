#ifndef SCRAPGPU_OPENCL11_H
#define SCRAPGPU_OPENCL11_H

#include "scrapgpu/api.h"
#include "scrapgpu/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the OpenCL 1.1 backend (src/backends/opencl11): the devices of cl11_probe_device_infos,
// honouring GGML_CL11_DEVICES (SCRAPGPU_OPENCL11_DEVICES is an alias). Safe to call more than once.
SCRAPGPU_API scrapgpu_status scrapgpu_opencl11_register(void);

#ifdef __cplusplus
}
#endif

#endif
