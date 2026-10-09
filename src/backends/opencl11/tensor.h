#ifndef SCRAPGPU_OPENCL11_TENSOR_H
#define SCRAPGPU_OPENCL11_TENSOR_H

#include "device.h"
#include "tensor_abi.h"

// a tensor of a cl_mem as the kernels see it: the buffer and the descriptor (built from a
// sgt::TensorDesc by cl11_make_td; the caller resolves the buffer and the byte offset)
struct cl11_tref {
    cl_mem   mem;
    cl11_td  td;
};

#endif
