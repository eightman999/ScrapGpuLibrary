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

// a tensor as the backend ops see it: the buffer, the descriptor of the natural layout (ne / nb /
// offset are always the logical ones, also for repacked tensors) and whether the storage is the
// repacked R32 layout. Built on the stack by the caller (the ggml adapter until the C API exists).
struct cl11_tensor {
    cl_mem          mem      = nullptr;
    sgt::TensorDesc d;
    bool            repacked = false;
};

// the kernel launch reference of a view (the 32-bit kernel descriptor of its natural layout)
inline cl11_tref cl11_tref_of(const cl11_tensor & t) {
    cl11_tref r;
    r.mem = t.mem;
    r.td  = cl11_make_td(t.d);
    return r;
}

#endif
