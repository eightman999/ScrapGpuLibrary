#ifndef SCRAPGPU_OPENCL11_BUFFER_H
#define SCRAPGPU_OPENCL11_BUFFER_H

#include "context.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

// largest cl_mem (and buffer) size: the kernels address a buffer with 32-bit signed byte offsets
#define CL11_MAX_BUFFER_SIZE ((size_t) 0x7ffff000)

// the tensors have to be aligned for the vector loads of the kernels (and for CL11_BASE)
#define CL11_MIN_ALIGNMENT 256

struct cl11_const_sub_entry {
    cl_mem parent;
    size_t offset;
    size_t size;
    cl_mem sub;
};

struct cl11_fill_span {
    size_t offset;
    size_t count;
};

size_t cl11_max_buffer_size(const cl11_device_info & info);
bool cl11_strict_alloc();
size_t cl11_max_alloc_size(const cl11_device_info & info, bool strict_alloc_dev);
size_t cl11_buffer_alignment(const cl11_device_info & info);
bool cl11_probe_buffer(cl_command_queue queue, cl_mem mem, size_t size);
bool cl11_alloc_buffer(cl_context context,
                       cl_command_queue queue,
                       const cl11_device_info & info,
                       const char * logical_name,
                       const char * caller_name,
                       bool strict_alloc_dev,
                       size_t requested_size,
                       cl_mem & mem,
                       size_t & alloc_size);
void cl11_release_const_subs_for_parent(std::vector<cl11_const_sub_entry> & subs, cl_mem parent);
cl_mem cl11_const_sub(std::vector<cl11_const_sub_entry> & subs, const cl11_device_info & info, cl_mem parent, size_t offset, size_t size);

inline bool cl11_fill_plan(size_t offset, size_t size, cl11_fill_span & head, cl11_fill_span & body, cl11_fill_span & tail) {
    if (size == 0) {
        head = {};
        body = {};
        tail = {};
        return false;
    }
    const size_t h = std::min(size, (4 - offset % 4) % 4);
    const size_t b = (size - h) / 4;
    const size_t t = size - h - b * 4;
    head = { offset, h };
    body = { offset + h, b };
    tail = { offset + h + b * 4, t };
    return true;
}

#endif
