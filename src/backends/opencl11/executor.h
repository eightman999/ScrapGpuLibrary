#ifndef SCRAPGPU_OPENCL11_EXECUTOR_H
#define SCRAPGPU_OPENCL11_EXECUTOR_H

#include "buffer.h"
#include "kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct cl11_args {
    cl_kernel kernel;
    cl_uint   idx = 0;
    bool      ok  = true;

    explicit cl11_args(cl_kernel k) : kernel(k) {}

    template <typename T>
    inline cl11_args & set(const T & v) {
        const cl_int err = clSetKernelArg(kernel, idx++, sizeof(T), &v);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "%s: clSetKernelArg(%u) failed: %s\n", __func__, idx - 1, cl11_err_str(err));
            ok = false;
        }
        return *this;
    }

    // a __local pointer argument of `bytes` bytes
    inline cl11_args & set_local(size_t bytes) {
        const cl_int err = clSetKernelArg(kernel, idx++, bytes, nullptr);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "%s: clSetKernelArg(%u, local %zu) failed: %s\n", __func__, idx - 1, bytes, cl11_err_str(err));
            ok = false;
        }
        return *this;
    }
};

struct cl11_prof_stat {
    std::string name;
    double      ns    = 0;
    size_t      count = 0;
    double      bytes = 0; // bytes of the weights (MUL_MAT)
};

struct cl11_prof_pending {
    cl_event event;
    int      key;
};

struct cl11_executor_state {
    bool profile = false;    // GGML_CL11_PROFILE=1: time every kernel with OpenCL events
    bool launch_log = false; // GGML_CL11_LAUNCH_LOG=1: print the number of kernel launches of every graph
    size_t n_launches = 0;
    int prof_key = -1; // key of the node that is being enqueued
    std::vector<cl11_prof_stat>    prof_stats;
    std::vector<cl11_prof_pending> prof_pending_events;
};

inline size_t cl11_div_up(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// collect the timings of the finished kernels (the queue has been finished by the caller)
inline void cl11_prof_collect(cl11_executor_state & exec) {
    for (const auto & p : exec.prof_pending_events) {
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(p.event, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(p.event, CL_PROFILING_COMMAND_END,   sizeof(t1), &t1, nullptr);
        if (p.key >= 0 && (size_t) p.key < exec.prof_stats.size()) {
            exec.prof_stats[p.key].ns += (double) (t1 - t0);
        }
        clReleaseEvent(p.event);
    }
    exec.prof_pending_events.clear();
}

inline int cl11_prof_key(cl11_executor_state & exec, const std::string & name) {
    for (size_t i = 0; i < exec.prof_stats.size(); ++i) {
        if (exec.prof_stats[i].name == name) {
            return (int) i;
        }
    }
    exec.prof_stats.emplace_back();
    exec.prof_stats.back().name = name;
    return (int) exec.prof_stats.size() - 1;
}

inline void cl11_prof_print(cl11_executor_state & exec, const char * name) {
    if (exec.prof_stats.empty()) {
        return;
    }
    std::vector<const cl11_prof_stat *> v;
    double total = 0;
    for (const auto & s : exec.prof_stats) {
        v.push_back(&s);
        total += s.ns;
    }
    std::sort(v.begin(), v.end(), [](auto * a, auto * b) { return a->ns > b->ns; });
    fprintf(stderr, "cl11 profile of %s: total kernel time %.3f ms\n", name, total / 1e6);
    for (auto * s : v) {
        char gbs[64] = "";
        if (s->bytes > 0 && s->ns > 0) {
            snprintf(gbs, sizeof(gbs), "  weights %.2f GB/s", s->bytes / s->ns);
        }
        fprintf(stderr, "  %-44s %6zu calls %10.3f ms %8.1f us/call %5.1f%%%s\n", s->name.c_str(), s->count,
                s->ns / 1e6, s->ns / 1e3 / std::max<size_t>(1, s->count), 100.0 * s->ns / std::max(1.0, total), gbs);
    }
    exec.prof_stats.clear();
}

// enqueue `n_groups` work-groups of `local` work-items. The launch is 2-D so that no dimension
// exceeds 32768 groups; the kernels linearize the group index themselves (LGID()).
inline bool cl11_launch(cl_command_queue queue, cl11_executor_state & exec, const cl11_args & args, size_t local, size_t n_groups) {
    if (!args.ok) {
        return false;
    }
    if (n_groups == 0) {
        return true;
    }
    ++exec.n_launches;
    const size_t gx = std::min<size_t>(n_groups, 32768);
    const size_t gy = (n_groups + gx - 1) / gx;
    const size_t global_size[2] = { gx * local, gy };
    const size_t local_size[2]  = { local, 1 };
    cl_event event = nullptr;
    const cl_int err = clEnqueueNDRangeKernel(queue, args.kernel, 2, nullptr, global_size, local_size, 0, nullptr, exec.profile ? &event : nullptr);
    if (event != nullptr) {
        exec.prof_pending_events.push_back({ event, exec.prof_key });
    }
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: clEnqueueNDRangeKernel failed: %s (groups %zu x %zu, local %zu)\n", __func__, cl11_err_str(err), gx, gy, local);
        return false;
    }
    return true;
}

// memset of a range of a buffer with a kernel (OpenCL 1.2 clEnqueueFillBuffer is not allowed)
inline bool cl11_fill(cl_command_queue queue, cl11_executor_state & exec, cl_kernel fill_u32, cl_kernel fill_u8,
                      const cl11_config & cfg, cl_mem mem, size_t offset, size_t size, uint8_t value) {
    if (size == 0) {
        return true;
    }

    const uint32_t pattern = 0x01010101u * (uint32_t) value;
    cl11_fill_span head = {};
    cl11_fill_span body = {};
    cl11_fill_span tail = {};
    cl11_fill_plan(offset, size, head, body, tail);

    auto fill_bytes = [&](size_t off, size_t n) -> bool {
        if (n == 0) {
            return true;
        }
        cl11_args a(fill_u8);
        a.set(mem).set((int32_t) off).set((int32_t) n).set((cl_uint) value);
        return cl11_launch(queue, exec, a, cfg.wg_ew, cl11_div_up(n, cfg.wg_ew));
    };

    if (!fill_bytes(head.offset, head.count)) {
        return false;
    }
    if (body.count > 0) {
        // the work-items address the buffer with int offsets: split very large fills
        const size_t chunk = 1u << 28; // words
        for (size_t done = 0; done < body.count; done += chunk) {
            const size_t n = std::min(chunk, body.count - done);
            cl11_args b(fill_u32);
            b.set(mem).set((int32_t) (body.offset + done * 4)).set((int32_t) n).set((cl_uint) pattern);
            if (!cl11_launch(queue, exec, b, cfg.wg_ew, cl11_div_up(n, cfg.wg_ew))) {
                return false;
            }
        }
    }
    return fill_bytes(tail.offset, tail.count);
}

#endif
