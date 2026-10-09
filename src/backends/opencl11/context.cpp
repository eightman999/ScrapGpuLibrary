#include "context.h"
#include "device_state.h"

#include <cstdio>
#include <cstdlib>

const char * cl11_err_str(cl_int err) {
    switch (err) {
        case CL_SUCCESS:                         return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:                return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:          return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:   return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:                return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:              return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE:           return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE:                   return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE:                  return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:                 return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE:           return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT:              return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_KERNEL_ARGS:             return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_GROUP_SIZE:         return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:          return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_WORK_SIZE:        return "CL_INVALID_GLOBAL_WORK_SIZE";
        case CL_INVALID_BUFFER_SIZE:             return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_OPERATION:               return "CL_INVALID_OPERATION";
        case CL_INVALID_ARG_SIZE:                return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_ARG_VALUE:               return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_INDEX:               return "CL_INVALID_ARG_INDEX";
        default:                                 return "unknown OpenCL error";
    }
}

bool cl11_ensure_runtime_context(cl11_device * dev) {
    if (dev->context_ready) {
        return true;
    }
    if (dev->failed) {
        return false;
    }

    cl_int err = CL_SUCCESS;
    const cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) dev->info.platform, 0 };
    dev->context = clCreateContext(props, 1, &dev->info.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: clCreateContext failed for %s: %s\n", __func__, dev->name.c_str(), cl11_err_str(err));
        dev->failed = true;
        return false;
    }

    dev->exec.profile = getenv("GGML_CL11_PROFILE") != nullptr;
    dev->exec.launch_log = getenv("GGML_CL11_LAUNCH_LOG") != nullptr;
    dev->queue = clCreateCommandQueue(dev->context, dev->info.device, dev->exec.profile ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: clCreateCommandQueue failed for %s: %s\n", __func__, dev->name.c_str(), cl11_err_str(err));
        clReleaseContext(dev->context);
        dev->context = nullptr;
        dev->failed = true;
        return false;
    }

    dev->context_ready = true;
    return true;
}
