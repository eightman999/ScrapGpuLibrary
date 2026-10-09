#ifndef SCRAPGPU_OPENCL11_CONTEXT_H
#define SCRAPGPU_OPENCL11_CONTEXT_H

#include "device.h"

const char * cl11_err_str(cl_int err);

bool cl11_ensure_runtime_context(const cl11_device_info & info,
                                 const char * name,
                                 bool & context_ready,
                                 bool & failed,
                                 cl_context & context,
                                 cl_command_queue & queue,
                                 bool & profile,
                                 bool & launch_log);

#endif
