#include "context.h"

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

bool cl11_ensure_runtime_context(const cl11_device_info & info,
                                 const char * name,
                                 bool & context_ready,
                                 bool & failed,
                                 cl_context & context,
                                 cl_command_queue & queue,
                                 bool & profile,
                                 bool & launch_log) {
    if (context_ready) {
        return true;
    }
    if (failed) {
        return false;
    }

    cl_int err = CL_SUCCESS;
    const cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) info.platform, 0 };
    context = clCreateContext(props, 1, &info.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: clCreateContext failed for %s: %s\n", __func__, name, cl11_err_str(err));
        failed = true;
        return false;
    }

    profile = getenv("GGML_CL11_PROFILE") != nullptr;
    launch_log = getenv("GGML_CL11_LAUNCH_LOG") != nullptr;
    queue = clCreateCommandQueue(context, info.device, profile ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: clCreateCommandQueue failed for %s: %s\n", __func__, name, cl11_err_str(err));
        clReleaseContext(context);
        context = nullptr;
        failed = true;
        return false;
    }

    context_ready = true;
    return true;
}
