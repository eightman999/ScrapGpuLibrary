#include "buffer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>

size_t cl11_max_buffer_size(const cl11_device_info & info) {
    return std::min(info.max_alloc, CL11_MAX_BUFFER_SIZE);
}

bool cl11_strict_alloc() {
    static const bool strict = [] {
        const char * env = getenv("GGML_CL11_STRICT_ALLOC");
        return env != nullptr && atoi(env) != 0;
    }();
    return strict;
}

size_t cl11_max_alloc_size(const cl11_device_info & info, bool strict_alloc_dev) {
    if (cl11_strict_alloc() || strict_alloc_dev) {
        return cl11_max_buffer_size(info);
    }
    return std::min(std::max(info.global_mem, info.max_alloc), CL11_MAX_BUFFER_SIZE);
}

size_t cl11_buffer_alignment(const cl11_device_info & info) {
    size_t align = std::max<size_t>(CL11_MIN_ALIGNMENT, info.align_bytes);
    // CL11_BASE is 4096-aligned
    align = std::min<size_t>(align, 4096);
    // power of 2
    return (size_t) cl11_pow2_floor(align);
}

bool cl11_probe_buffer(cl_command_queue queue, cl_mem mem, size_t size) {
    const uint8_t zeros[16] = {};
    if (size < sizeof(zeros)) {
        return true;
    }
    const cl_int err = clEnqueueWriteBuffer(queue, mem, CL_TRUE, size - sizeof(zeros), sizeof(zeros), zeros, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: the probe of a %zu byte buffer failed: %s\n", __func__, size, cl11_err_str(err));
    }
    return err == CL_SUCCESS;
}

bool cl11_alloc_buffer(cl_context context,
                       cl_command_queue queue,
                       const cl11_device_info & info,
                       const char * logical_name,
                       const char * caller_name,
                       bool strict_alloc_dev,
                       size_t requested_size,
                       cl_mem & mem,
                       size_t & alloc_size) {
    mem = nullptr;
    // keep a few spare bytes: the size is never zero and always a multiple of 16
    alloc_size = std::max<size_t>(16, (requested_size + 15) & ~(size_t) 15);
    const size_t max_alloc = cl11_max_alloc_size(info, strict_alloc_dev);
    if (alloc_size > max_alloc) {
        fprintf(stderr, "%s: cannot allocate %.2f MiB on %s: the limit of a single allocation is %.2f MiB%s\n", caller_name,
                requested_size / 1024.0 / 1024.0, logical_name, max_alloc / 1024.0 / 1024.0,
                cl11_strict_alloc() ? " (GGML_CL11_STRICT_ALLOC)" : "");
        return false;
    }

    cl_int err = CL_SUCCESS;
    mem = clCreateBuffer(context, CL_MEM_READ_WRITE, alloc_size, nullptr, &err);
    if (err != CL_SUCCESS || mem == nullptr) {
        fprintf(stderr, "%s: clCreateBuffer(%.2f MiB) failed on %s: %s\n", caller_name,
                alloc_size / 1024.0 / 1024.0, logical_name, cl11_err_str(err));
        mem = nullptr;
        return false;
    }

    if (alloc_size > info.max_alloc) {
        static std::once_flag warned;
        std::call_once(warned, [&]() {
            fprintf(stderr, "%s: allocating %.2f MiB on %s, more than CL_DEVICE_MAX_MEM_ALLOC_SIZE (%.2f MiB); "
                            "set GGML_CL11_STRICT_ALLOC=1 to refuse such allocations\n", "ggml_backend_cl11_buft_alloc_buffer",
                    alloc_size / 1024.0 / 1024.0, logical_name, info.max_alloc / 1024.0 / 1024.0);
        });
    }
    if (!cl11_probe_buffer(queue, mem, alloc_size)) {
        fprintf(stderr, "%s: the %.2f MiB allocation on %s is not usable: out of device memory\n", caller_name,
                alloc_size / 1024.0 / 1024.0, logical_name);
        clReleaseMemObject(mem);
        mem = nullptr;
        return false;
    }
    return true;
}

void cl11_release_const_subs_for_parent(std::vector<cl11_const_sub_entry> & subs, cl_mem parent) {
    // newest first: the NVIDIA 390 driver crashes in clReleaseMemObject when overlapping sub-buffers of one buffer are released in the
    // order of their creation (the sub-buffers of the columns of src1 overlap, as llama-perplexity -b 1 shows)
    for (size_t i = subs.size(); i-- > 0;) {
        if (subs[i].parent == parent) {
            clReleaseMemObject(subs[i].sub);
            subs.erase(subs.begin() + i);
        }
    }
}

cl_mem cl11_const_sub(std::vector<cl11_const_sub_entry> & subs, const cl11_device_info & info, cl_mem parent, size_t offset, size_t size) {
    if (info.align_bytes == 0 || offset % info.align_bytes != 0) {
        return nullptr;
    }
    for (const auto & e : subs) {
        if (e.parent == parent && e.offset == offset && e.size == size) {
            return e.sub;
        }
    }
    cl_buffer_region region = { offset, size };
    cl_int err = CL_SUCCESS;
    cl_mem sub = clCreateSubBuffer(parent, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    if (err != CL_SUCCESS || sub == nullptr) {
        return nullptr;
    }
    subs.push_back({ parent, offset, size, sub });
    return sub;
}
