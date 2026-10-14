// The ScrapGpu C API on the host CPU: scrapgpu_cpu_register() registers a backend with a single
// device named "CPU" that keeps its buffers in aligned system memory and executes the ops
// synchronously through scrapgpu_ref_execute (src/core/ref). Everything enqueued on a stream is
// complete when the call returns; stream_sync is a no-op. The device is portable C++17: the
// reference uses SSE2 where the compiler baseline provides it and nothing more (no AVX, no
// threads, no external BLAS) is required.
//
// Only the natural layout is supported (R32 is an OpenCL repacking); quantized weights stay in
// the GGUF block layout the reference reads directly. op_supported mirrors the descriptor checks
// of the reference implementation -- it inspects the metadata only and never touches tensor data.
//
// Aliasing: the reference computes on host pointers, so an output must not overlap an input
// unless the op is elementwise (every element is read before it is written) *and* the overlapping
// tensor is the identical view: the same buffer, offset, dtype, shape and strides. Outputs must
// never overlap each other.
//
// The descriptor math below (nbytes / span / contiguity) mirrors sgt:: (src/tensor/tensor.h) and
// the reference's tensor_span_ok so that this file only needs the public headers of scrapgpu_core.

#include "scrapgpu/backend.h"
#include "scrapgpu/cpu.h"
#include "scrapgpu/reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#  include <malloc.h>
#  include <windows.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <unistd.h>
#else
#  include <unistd.h>
#endif

namespace {

// ---------------------------------------------------------------------------------------------
// objects
// ---------------------------------------------------------------------------------------------

struct cpu_api_device {
    scrapgpu_object_header hdr; // must stay the first member
    uint64_t             total_memory = 0;
    uint32_t             compute_units = 1;
};

struct cpu_api_buffer {
    scrapgpu_object_header hdr;
    cpu_api_device *       dev  = nullptr;
    uint8_t *              data = nullptr; // 256-byte aligned host allocation of `size` bytes
    size_t                 size = 0;
};

struct cpu_api_stream {
    scrapgpu_object_header hdr;
    cpu_api_device *       dev = nullptr;
};

struct cpu_api_plan_node {
    scrapgpu_op_desc                   desc;
    std::vector<scrapgpu_tensor_desc>  inputs;
    std::vector<scrapgpu_tensor_desc>  outputs;
    std::vector<uint8_t>               params;
};

struct cpu_api_plan {
    scrapgpu_object_header           hdr;
    cpu_api_device *                 dev = nullptr;
    std::vector<cpu_api_plan_node>   nodes;
    // every backend buffer the node descriptors reference: a plan keeps no ownership, but
    // buffer_free refuses while a plan points at the buffer (same contract as the other backends)
    std::vector<cpu_api_buffer *>    bufs;
};

extern const scrapgpu_backend_vtable g_cpu_vtable; // defined under "registration" below

std::vector<std::unique_ptr<cpu_api_device>> g_devices;

// how many live plans reference each buffer (the pointers of the descriptor copies in the plans)
std::mutex                                    g_plan_bufs_mutex;
std::unordered_map<cpu_api_buffer *, size_t>  g_plan_bufs;

void plan_ref_buffer(cpu_api_buffer * b) {
    std::lock_guard<std::mutex> lock(g_plan_bufs_mutex);
    ++g_plan_bufs[b];
}

void plan_unref_buffer(cpu_api_buffer * b) {
    std::lock_guard<std::mutex> lock(g_plan_bufs_mutex);
    const auto it = g_plan_bufs.find(b);
    if (it != g_plan_bufs.end() && --it->second == 0) {
        g_plan_bufs.erase(it);
    }
}

bool plan_refs_buffer(cpu_api_buffer * b) {
    std::lock_guard<std::mutex> lock(g_plan_bufs_mutex);
    return g_plan_bufs.find(b) != g_plan_bufs.end();
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

scrapgpu_status api_unsupported(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

cpu_api_device * api_device(scrapgpu_device_t * device) {
    auto * d = reinterpret_cast<cpu_api_device *>(device);
    if (d == nullptr || d->hdr.backend != &g_cpu_vtable) {
        return nullptr;
    }
    return d;
}

cpu_api_buffer * api_buffer(scrapgpu_buffer_t * buffer) {
    auto * b = reinterpret_cast<cpu_api_buffer *>(buffer);
    if (b == nullptr || b->hdr.backend != &g_cpu_vtable) {
        return nullptr;
    }
    return b;
}

cpu_api_stream * api_stream(scrapgpu_stream_t * stream) {
    auto * s = reinterpret_cast<cpu_api_stream *>(stream);
    if (s == nullptr || s->hdr.backend != &g_cpu_vtable) {
        return nullptr;
    }
    return s;
}

// the caller-visible fields of a tensor desc (a smaller struct_size keeps the defaults of the tail:
// buffer = NULL, offset = 0, layout = NATURAL)
scrapgpu_tensor_desc api_load_desc(const scrapgpu_tensor_desc & in) {
    scrapgpu_tensor_desc out = {};
    if (in.struct_size >= sizeof(size_t)) {
        std::memcpy(&out, &in, std::min(in.struct_size, sizeof(out)));
        out.struct_size = sizeof(out);
    }
    return out;
}

template <typename T>
T api_params(const scrapgpu_op_desc * desc) {
    T p = {};
    if (desc->params != nullptr) {
        std::memcpy(&p, desc->params, std::min(desc->params_size, sizeof(T)));
    }
    return p;
}

// a params enum as an integer: a value outside the declared range must not be loaded through the
// enum type itself (UB); every range check below goes through this
template <typename E>
uint32_t api_enum_u32(const E & e) {
    static_assert(sizeof(e) <= sizeof(uint32_t), "enum fits in u32");
    uint32_t v = 0;
    std::memcpy(&v, &e, sizeof(e));
    return v;
}

// no exception escapes the C ABI: allocations in the plan/desc copies and in the reference's own
// scratch vectors surface as OUT_OF_MEMORY, everything else as BACKEND_ERROR
template <typename F>
scrapgpu_status api_guard(F && fn) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc &) {
        try { scrapgpu_set_last_error("out of host memory"); } catch (...) {}
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    } catch (...) {
        try { scrapgpu_set_last_error("the CPU backend threw an exception"); } catch (...) {}
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
}

// ---------------------------------------------------------------------------------------------
// descriptor math (mirrors sgt:: of src/tensor/tensor.h and the reference's tensor_span_ok; the
// sgt helpers are internal to the core, so the CPU backend keeps its own copies)
// ---------------------------------------------------------------------------------------------

int64_t td_ne(const scrapgpu_tensor_desc & t, int i) {
    return t.ne[i] > 0 ? t.ne[i] : 1;
}

size_t td_esz(const scrapgpu_tensor_desc & t) {
    return scrapgpu_dtype_type_size(t.dtype);
}

bool td_quant(const scrapgpu_tensor_desc & t) {
    return scrapgpu_dtype_block_size(t.dtype) > 1;
}

uint64_t td_mul_sat(uint64_t a, uint64_t b) {
    return a != 0 && b > std::numeric_limits<uint64_t>::max() / a ? std::numeric_limits<uint64_t>::max() : a * b;
}

uint64_t td_add_sat(uint64_t a, uint64_t b) {
    return b > std::numeric_limits<uint64_t>::max() - a ? std::numeric_limits<uint64_t>::max() : a + b;
}

// the bytes a tensor touches in its storage (the offset of the last byte + 1), saturating on
// overflow; 0 when a dim is <= 0 or the dtype is unknown (sgt::nbytes)
uint64_t td_nbytes(const scrapgpu_tensor_desc & t) {
    for (int i = 0; i < 4; ++i) {
        if (t.ne[i] <= 0) {
            return 0;
        }
    }
    const size_t block_size = scrapgpu_dtype_block_size(t.dtype);
    const size_t type_size  = scrapgpu_dtype_type_size(t.dtype);
    if (block_size == 0 || type_size == 0) {
        return 0;
    }
    uint64_t n;
    if (block_size == 1) {
        n = type_size;
        for (int i = 0; i < 4; ++i) {
            n = td_add_sat(n, td_mul_sat((uint64_t) (t.ne[i] - 1), t.nb[i]));
        }
    } else {
        n = td_mul_sat((uint64_t) t.ne[0], t.nb[0]) / block_size;
        for (int i = 1; i < 4; ++i) {
            n = td_add_sat(n, td_mul_sat((uint64_t) (t.ne[i] - 1), t.nb[i]));
        }
    }
    return n;
}

// the reference's tensor_span_ok: offset + sum((ne[i]-1) * nb[i]) inside half of the address space
bool td_span_ok(const scrapgpu_tensor_desc & t) {
    if (t.offset > std::numeric_limits<uint64_t>::max() / 2) {
        return false;
    }
    uint64_t span = t.offset;
    for (int i = 0; i < 4; ++i) {
        const uint64_t count = (uint64_t) (td_ne(t, i) - 1);
        if (t.nb[i] != 0 && count > (std::numeric_limits<uint64_t>::max() / 2 - span) / t.nb[i]) {
            return false;
        }
        span += count * t.nb[i];
        if (span > std::numeric_limits<uint64_t>::max() / 2) {
            return false;
        }
    }
    return true;
}

// the ggml definition of contiguity (sgt::is_contiguous): a dim of size 1 and the stride of a
// single-block row are not constrained
bool td_contiguous(const scrapgpu_tensor_desc & t) {
    const size_t block_size = scrapgpu_dtype_block_size(t.dtype);
    const size_t type_size  = scrapgpu_dtype_type_size(t.dtype);
    if (block_size == 0 || type_size == 0) {
        return false;
    }
    uint64_t next = type_size;
    if (t.ne[0] != (int64_t) block_size && t.nb[0] != next) {
        return false;
    }
    bool overflow = next != 0 && (uint64_t) (t.ne[0] / (int64_t) block_size) > std::numeric_limits<uint64_t>::max() / next;
    next *= (uint64_t) (t.ne[0] / (int64_t) block_size);
    for (int i = 1; i < 4; ++i) {
        if (t.ne[i] != 1 && t.nb[i] != next) {
            return false;
        }
        overflow |= next != 0 && (uint64_t) t.ne[i] > std::numeric_limits<uint64_t>::max() / next;
        next *= (uint64_t) t.ne[i];
    }
    return !overflow;
}

bool td_same_ne(const scrapgpu_tensor_desc & a, const scrapgpu_tensor_desc & b) {
    for (int i = 0; i < 4; ++i) {
        if (td_ne(a, i) != td_ne(b, i)) {
            return false;
        }
    }
    return true;
}

// the product of the dims, false on overflow (the reference's nelements_ok)
bool td_nelements_ok(const scrapgpu_tensor_desc & t, int64_t & out) {
    int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        const int64_t d = td_ne(t, i);
        if (d > 0 && n > std::numeric_limits<int64_t>::max() / d) {
            return false;
        }
        n *= d;
    }
    out = n;
    return true;
}

// contiguous rows of the reference: nb[0] == element size and, for quantized dtypes, whole blocks
scrapgpu_status td_require_rows(const scrapgpu_tensor_desc & t, const char * what) {
    if (t.nb[0] != td_esz(t)) {
        return api_unsupported(what);
    }
    if (td_quant(t) && t.ne[0] % (int64_t) scrapgpu_dtype_block_size(t.dtype) != 0) {
        return api_unsupported("quantized row length must be a multiple of the block size");
    }
    return SCRAPGPU_STATUS_OK;
}

bool td_is_weight(scrapgpu_dtype dtype) {
    return dtype == SCRAPGPU_DTYPE_F32 || dtype == SCRAPGPU_DTYPE_F16 || dtype == SCRAPGPU_DTYPE_BF16;
}

bool td_is_mm_weight(scrapgpu_dtype dtype) {
    return td_is_weight(dtype) || scrapgpu_dtype_block_size(dtype) > 1;
}

// ---------------------------------------------------------------------------------------------
// op_supported
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_check_tensors(const scrapgpu_op_desc * desc, const char ** reason) {
    auto check_one = [&](const scrapgpu_tensor_desc & raw) -> scrapgpu_status {
        const scrapgpu_tensor_desc td = api_load_desc(raw);
        if (td_esz(td) == 0) {
            *reason = "unknown tensor dtype";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        for (int i = 0; i < 4; ++i) {
            if (td.ne[i] <= 0) {
                *reason = "non-positive tensor dim";
                return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            }
        }
        if (!td_span_ok(td)) {
            *reason = "tensor strides or offset overflow the address space";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        if (td.layout != SCRAPGPU_LAYOUT_NATURAL) {
            *reason = "the CPU backend only supports the natural layout (R32 is an OpenCL repacking)";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        return SCRAPGPU_STATUS_OK;
    };
    for (size_t i = 0; i < desc->input_count; ++i) {
        const scrapgpu_status st = check_one(desc->inputs[i]);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        const scrapgpu_status st = check_one(desc->outputs[i]);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_no(const char * why, const char ** reason) {
    if (reason != nullptr) {
        *reason = why;
    }
    scrapgpu_set_last_error(why);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

scrapgpu_status api_bad(const char * why, const char ** reason) {
    if (reason != nullptr) {
        *reason = why;
    }
    scrapgpu_set_last_error(why);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

// mirrors the descriptor checks of scrapgpu_ref_execute, without touching tensor data: a desc that
// is supported here can still fail at execute time only on index *values* (GET_ROWS / SET_ROWS /
// MUL_MAT_ID ids), which the backend cannot see
scrapgpu_status api_supports(cpu_api_device * d, const scrapgpu_op_desc * desc, const char ** reason) {
    (void) d;
    const char * reason_local = nullptr;
    if (reason == nullptr) {
        reason = &reason_local;
    }
    *reason = nullptr;
    // The generic validator reads dtype fields. Check the storage contract first.
    if (desc == nullptr || desc->struct_size < sizeof(scrapgpu_op_desc)) {
        return api_bad("op descriptor is too small", reason);
    }
    if (desc->input_count > 32 || desc->output_count > 32 ||
        (desc->input_count && !desc->inputs) || (desc->output_count && !desc->outputs)) {
        return api_bad("invalid tensor descriptor arrays", reason);
    }
    for (size_t i = 0; i < desc->input_count; ++i) {
        if (desc->inputs[i].struct_size < sizeof(scrapgpu_tensor_desc)) {
            return api_bad("input tensor descriptor is too small", reason);
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        if (desc->outputs[i].struct_size < sizeof(scrapgpu_tensor_desc)) {
            return api_bad("output tensor descriptor is too small", reason);
        }
    }
    const scrapgpu_status valid = scrapgpu_op_desc_validate(desc);
    if (valid != SCRAPGPU_STATUS_OK) {
        if (reason != nullptr) {
            *reason = scrapgpu_last_error();
        }
        return valid;
    }

    switch (desc->op) {
        case SCRAPGPU_OP_NONE:
        case SCRAPGPU_OP_RESHAPE:
        case SCRAPGPU_OP_VIEW:
        case SCRAPGPU_OP_PERMUTE:
        case SCRAPGPU_OP_TRANSPOSE:
            return SCRAPGPU_STATUS_OK;
        default:
            break;
    }

    {
        const scrapgpu_status st = api_check_tensors(desc, reason);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }

    // normalized descriptors (the op's input/output counts were enforced by the validation above)
    const auto s0 = api_load_desc(desc->inputs[0]);
    const auto s1 = desc->input_count > 1 ? api_load_desc(desc->inputs[1]) : scrapgpu_tensor_desc{};
    const auto s2 = desc->input_count > 2 ? api_load_desc(desc->inputs[2]) : scrapgpu_tensor_desc{};
    const auto dd = api_load_desc(desc->outputs[0]);
    const auto f32 = [](const scrapgpu_tensor_desc & t) { return t.dtype == SCRAPGPU_DTYPE_F32; };
    const auto f16_or_f32 = [](const scrapgpu_tensor_desc & t) {
        return t.dtype == SCRAPGPU_DTYPE_F16 || t.dtype == SCRAPGPU_DTYPE_F32;
    };
    const auto f16_bf16_f32 = [](const scrapgpu_tensor_desc & t) {
        return t.dtype == SCRAPGPU_DTYPE_F16 || t.dtype == SCRAPGPU_DTYPE_BF16 || t.dtype == SCRAPGPU_DTYPE_F32;
    };
    scrapgpu_status st = SCRAPGPU_STATUS_OK;

    switch (desc->op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_SUB:
        case SCRAPGPU_OP_MUL:
        case SCRAPGPU_OP_DIV: {
            if (!f32(s0) || !f32(s1) || !f32(dd)) {
                return api_no("binary ops need F32 tensors", reason);
            }
            if (!td_same_ne(s0, dd)) {
                return api_bad("binary op: src0 must have the shape of dst", reason);
            }
            for (int i = 0; i < 4; ++i) {
                if (td_ne(s1, i) > td_ne(dd, i)) {
                    return api_bad("binary op: src1 cannot be larger than dst", reason);
                }
            }
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_SCALE:
        case SCRAPGPU_OP_CLAMP:
        case SCRAPGPU_OP_SQR:
        case SCRAPGPU_OP_SQRT:
        case SCRAPGPU_OP_UNARY: {
            if (!f32(s0) || !f32(dd)) {
                return api_no("unary-like ops need F32 tensors", reason);
            }
            if (desc->op == SCRAPGPU_OP_UNARY) {
                const auto p = api_params<scrapgpu_unary_params>(desc);
                if (api_enum_u32(p.kind) > (uint32_t) SCRAPGPU_UNARY_TANH) {
                    return api_no("unary kind is not implemented", reason);
                }
            }
            return td_same_ne(s0, dd) ? SCRAPGPU_STATUS_OK
                                      : api_bad("unary-like op: src0 must have the shape of dst", reason);
        }

        case SCRAPGPU_OP_GLU: {
            const auto p = api_params<scrapgpu_glu_params>(desc);
            const bool split = desc->input_count == 2;
            if (split != (p.split_inputs != 0)) {
                return api_bad("GLU: split_inputs does not match the number of inputs", reason);
            }
            if (!f32(s0) || !f32(dd) || (split && !f32(s1))) {
                return api_no("GLU expects F32 tensors", reason);
            }
            if (api_enum_u32(p.kind) > (uint32_t) SCRAPGPU_GLU_REGLU) {
                return api_no("unknown GLU kind", reason);
            }
            if (split) {
                return td_same_ne(s0, dd) && td_same_ne(s1, dd)
                    ? SCRAPGPU_STATUS_OK
                    : api_bad("GLU with split inputs needs src0, src1 and dst of the same shape", reason);
            }
            for (int i = 1; i < 4; ++i) {
                if (td_ne(s0, i) != td_ne(dd, i)) {
                    return api_bad("GLU: src0 and dst differ outside of dim 0", reason);
                }
            }
            return td_ne(s0, 0) == 2 * td_ne(dd, 0)
                ? SCRAPGPU_STATUS_OK
                : api_bad("GLU: src0 dim 0 must be twice the dst dim 0", reason);
        }

        case SCRAPGPU_OP_RMS_NORM:
            if (!f32(s0) || !f32(dd)) {
                return api_no("RMS_NORM expects F32 tensors", reason);
            }
            return td_same_ne(s0, dd) ? SCRAPGPU_STATUS_OK
                                      : api_bad("RMS_NORM: src0 must have the shape of dst", reason);

        case SCRAPGPU_OP_RMS_NORM_MUL:
            if (!f32(s0) || !f32(s1) || !f32(dd)) {
                return api_no("RMS_NORM_MUL expects F32 tensors", reason);
            }
            return td_same_ne(s0, dd) && td_ne(s1, 0) == td_ne(s0, 0)
                ? SCRAPGPU_STATUS_OK
                : api_bad("RMS_NORM_MUL: dst must match src0 and the weight must be a row vector", reason);

        case SCRAPGPU_OP_ADD_RMS_NORM_MUL: {
            const auto o1 = api_load_desc(desc->outputs[1]);
            if (!f32(s0) || !f32(s1) || !f32(s2) || !f32(dd) || !f32(o1)) {
                return api_no("ADD_RMS_NORM_MUL expects F32 tensors", reason);
            }
            return td_same_ne(s0, s1) && td_same_ne(s0, dd) && td_same_ne(s0, o1) && td_ne(s2, 0) == td_ne(s0, 0)
                ? SCRAPGPU_STATUS_OK
                : api_bad("ADD_RMS_NORM_MUL: a, b, the outputs and the weight row must have matching shapes", reason);
        }

        case SCRAPGPU_OP_ROPE: {
            const auto p = api_params<scrapgpu_rope_params>(desc);
            if (api_enum_u32(p.mode) > (uint32_t) SCRAPGPU_ROPE_MODE_NEOX) {
                return api_no("unknown ROPE mode", reason);
            }
            if (desc->input_count < 2) {
                return api_bad("ROPE needs the positions tensor as inputs[1]", reason);
            }
            const auto pos = api_load_desc(desc->inputs[1]);
            if ((!f32(s0) && s0.dtype != SCRAPGPU_DTYPE_F16) || dd.dtype != s0.dtype) {
                return api_no("ROPE expects F32 or F16 tensors of the same dtype", reason);
            }
            if (pos.dtype != SCRAPGPU_DTYPE_I32) {
                return api_no("ROPE positions must be I32", reason);
            }
            if (p.n_offs != nullptr) {
                return api_no("ROPE n_offs is not supported by the reference", reason);
            }
            if (p.freq_factors != nullptr && desc->input_count != 3) {
                return api_no("ROPE params->freq_factors must be NULL or describe inputs[2]", reason);
            }
            if (!td_same_ne(s0, dd)) {
                return api_bad("ROPE: src0 must have the shape of dst", reason);
            }
            if (s0.nb[0] != td_esz(s0) || dd.nb[0] != td_esz(dd)) {
                return api_no("ROPE needs contiguous rows (nb0 == element size)", reason);
            }
            if (p.n_dims <= 0 || p.n_dims % 2 != 0 || p.n_dims > s0.ne[0] || s0.ne[0] % 2 != 0) {
                return api_bad("ROPE: n_dims must be a positive even number not above ne0 (ne0 even)", reason);
            }
            if (td_ne(pos, 0) < td_ne(dd, 2)) {
                return api_bad("ROPE: not enough positions for the sequence dimension", reason);
            }
            if (desc->input_count == 3) {
                const auto ff = api_load_desc(desc->inputs[2]);
                if (ff.dtype != SCRAPGPU_DTYPE_F32 || td_ne(ff, 0) < p.n_dims / 2) {
                    return api_bad("ROPE: freq factors must be F32 with at least n_dims/2 entries", reason);
                }
            }
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_SOFT_MAX: {
            const auto p = api_params<scrapgpu_soft_max_params>(desc);
            if (!f32(s0) || !f32(dd)) {
                return api_no("SOFT_MAX expects F32 tensors", reason);
            }
            if (!td_same_ne(s0, dd)) {
                return api_bad("SOFT_MAX: src0 must have the shape of dst", reason);
            }
            if (p.mask != nullptr && desc->input_count != 2) {
                return api_no("SOFT_MAX params->mask must be NULL or describe inputs[1]", reason);
            }
            if (desc->input_count == 2) {
                const auto mask = api_load_desc(desc->inputs[1]);
                if ((!f32(mask) && mask.dtype != SCRAPGPU_DTYPE_F16) ||
                    td_ne(mask, 0) != td_ne(s0, 0) || td_ne(mask, 1) < td_ne(s0, 1)) {
                    return api_bad("SOFT_MAX mask must be F32/F16, as long as a row and cover the rows of src0", reason);
                }
            }
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_GET_ROWS: {
            if (s1.dtype != SCRAPGPU_DTYPE_I32 || !f32(dd)) {
                return api_no("GET_ROWS expects I32 indices and an F32 destination", reason);
            }
            if (s0.dtype != SCRAPGPU_DTYPE_F32 && s0.dtype != SCRAPGPU_DTYPE_F16 && !td_quant(s0)) {
                return api_no("GET_ROWS dtype is not implemented", reason);
            }
            st = td_require_rows(s0, "GET_ROWS needs contiguous rows (nb0 == element size)");
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
            return td_ne(s1, 3) == 1 && td_ne(s1, 1) <= td_ne(s0, 2) && td_ne(s1, 2) <= td_ne(s0, 3) &&
                   td_ne(dd, 0) == td_ne(s0, 0) && td_ne(dd, 1) == td_ne(s1, 0) &&
                   td_ne(dd, 2) == td_ne(s1, 1) && td_ne(dd, 3) == td_ne(s1, 2)
                ? SCRAPGPU_STATUS_OK
                : api_bad("GET_ROWS: dst must be [ne00, ids] over the slices of src0", reason);
        }

        case SCRAPGPU_OP_SET_ROWS: {
            if (!f16_or_f32(s0) || !f16_or_f32(dd)) {
                return api_no("SET_ROWS expects F32/F16 values and destination", reason);
            }
            if (s1.dtype != SCRAPGPU_DTYPE_I32 && s1.dtype != SCRAPGPU_DTYPE_I64) {
                return api_no("SET_ROWS indices must be I32 or I64", reason);
            }
            st = td_require_rows(s0, "SET_ROWS needs contiguous rows (nb0 == element size)");
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
            return td_ne(s0, 0) == td_ne(dd, 0) && td_ne(s0, 2) == td_ne(dd, 2) &&
                   td_ne(s0, 3) == td_ne(dd, 3) && td_ne(dd, 1) >= 1 &&
                   td_ne(s1, 0) >= td_ne(s0, 1) && td_ne(s1, 1) >= 1 && td_ne(s1, 2) >= 1 &&
                   td_ne(s0, 2) % td_ne(s1, 1) == 0 && td_ne(s0, 3) % td_ne(s1, 2) == 0
                ? SCRAPGPU_STATUS_OK
                : api_bad("SET_ROWS: dst must match src0 except along dim 1 and the index tensor must cover its rows", reason);
        }

        case SCRAPGPU_OP_CPY:
        case SCRAPGPU_OP_DUP:
        case SCRAPGPU_OP_CONT: {
            if (!f16_bf16_f32(s0) || !f16_bf16_f32(dd)) {
                return api_no("CPY/DUP/CONT support F32, F16 and BF16 only", reason);
            }
            int64_t ns = 0, nd = 0;
            if (!td_nelements_ok(s0, ns) || !td_nelements_ok(dd, nd) || ns != nd) {
                return api_bad("CPY/DUP/CONT: src0 and dst must have the same number of elements", reason);
            }
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_MUL_MAT: {
            const auto p = api_params<scrapgpu_mul_mat_params>(desc);
            if (!f32(s1) || !f32(dd)) {
                return api_no("MUL_MAT expects an F32 src1 and destination", reason);
            }
            if (!td_is_mm_weight(s0.dtype)) {
                return api_no("MUL_MAT dtype is not implemented", reason);
            }
            st = td_require_rows(s0, "MUL_MAT needs contiguous rows of src0 (nb0 == element size)");
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
            if (s1.nb[0] != td_esz(s1) || dd.nb[0] != td_esz(dd)) {
                return api_no("MUL_MAT needs contiguous rows of src1 and dst (nb0 == element size)", reason);
            }
            if (td_ne(s0, 0) != td_ne(s1, 0)) {
                return api_bad("MUL_MAT: the inner dimensions of src0 and src1 differ", reason);
            }
            if (td_ne(s1, 2) % td_ne(s0, 2) != 0 || td_ne(s1, 3) % td_ne(s0, 3) != 0) {
                return api_bad("MUL_MAT: src1 dims 2 and 3 must be multiples of the src0 dims", reason);
            }
            const int64_t r2 = td_ne(s1, 2) / td_ne(s0, 2);
            const int64_t r3 = td_ne(s1, 3) / td_ne(s0, 3);
            if (p.allow_broadcast == 0 && (r2 != 1 || r3 != 1)) {
                return api_no("MUL_MAT: broadcasting over dims 2/3 is not allowed", reason);
            }
            return td_ne(dd, 0) == td_ne(s0, 1) && td_ne(dd, 1) == td_ne(s1, 1) &&
                   td_ne(dd, 2) == td_ne(s1, 2) && td_ne(dd, 3) == td_ne(s1, 3)
                ? SCRAPGPU_STATUS_OK
                : api_bad("MUL_MAT: dst must be [ne01, ne11, ne12, ne13]", reason);
        }

        case SCRAPGPU_OP_MUL_MAT_ID: {
            const auto ids = api_load_desc(desc->inputs[2]);
            if (!f32(s1) || !f32(dd) || ids.dtype != SCRAPGPU_DTYPE_I32) {
                return api_no("MUL_MAT_ID expects F32 src1/dst and I32 ids", reason);
            }
            if (s0.dtype != SCRAPGPU_DTYPE_F32 && s0.dtype != SCRAPGPU_DTYPE_F16 && !td_quant(s0)) {
                return api_no("MUL_MAT_ID dtype is not implemented", reason);
            }
            st = td_require_rows(s0, "MUL_MAT_ID needs contiguous rows of src0 (nb0 == element size)");
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
            if (td_ne(s0, 3) != 1 || td_ne(s1, 3) != 1 || td_ne(ids, 2) != 1 || td_ne(ids, 3) != 1) {
                return api_bad("MUL_MAT_ID: the tensors must be 3-dimensional at most", reason);
            }
            return td_ne(s0, 0) == td_ne(s1, 0) && (td_ne(s1, 1) == 1 || td_ne(s1, 1) == td_ne(ids, 0)) &&
                   td_ne(s1, 2) == td_ne(ids, 1) && td_ne(dd, 0) == td_ne(s0, 1) &&
                   td_ne(dd, 1) == td_ne(ids, 0) && td_ne(dd, 2) == td_ne(ids, 1) && td_ne(dd, 3) == 1
                ? SCRAPGPU_STATUS_OK
                : api_bad("MUL_MAT_ID: shapes do not match the contract", reason);
        }

        case SCRAPGPU_OP_ADD_ID: {
            const auto ids = api_load_desc(desc->inputs[2]);
            if (!f32(s0) || !f32(s1) || !f32(dd)) {
                return api_no("ADD_ID expects F32 tensors", reason);
            }
            if (ids.dtype != SCRAPGPU_DTYPE_I32) {
                return api_no("ADD_ID ids must be I32", reason);
            }
            return td_ne(s0, 3) == 1 && td_ne(s1, 2) == 1 && td_ne(s1, 3) == 1 &&
                   td_ne(s0, 0) == td_ne(s1, 0) && td_ne(ids, 0) == td_ne(s0, 1) &&
                   td_ne(ids, 1) == td_ne(s0, 2) && td_ne(ids, 2) == 1 && td_same_ne(s0, dd)
                ? SCRAPGPU_STATUS_OK
                : api_bad("ADD_ID: shapes do not match the contract", reason);
        }

        case SCRAPGPU_OP_SSM_CONV: {
            const auto p = api_params<scrapgpu_ssm_conv_params>(desc);
            if (!f32(s0) || !f32(s1) || !f32(dd)) {
                return api_no("SSM_CONV expects F32 tensors", reason);
            }
            if (p.d_conv != 0 && (int64_t) p.d_conv != td_ne(s1, 0)) {
                return api_bad("SSM_CONV: d_conv does not match the length of the filter", reason);
            }
            return td_ne(s0, 3) == 1 && td_ne(s1, 2) == 1 && td_ne(s1, 3) == 1 &&
                   td_ne(s0, 1) == td_ne(s1, 1) && td_ne(s0, 1) == td_ne(dd, 0) &&
                   td_ne(dd, 1) == td_ne(s0, 0) - td_ne(s1, 0) + 1 &&
                   td_ne(dd, 2) == td_ne(s0, 2) && td_ne(dd, 3) == 1
                ? SCRAPGPU_STATUS_OK
                : api_bad("SSM_CONV: shapes do not match the contract", reason);
        }

        case SCRAPGPU_OP_CONCAT: {
            const auto p = api_params<scrapgpu_concat_params>(desc);
            if (p.dim >= 4) {
                return api_bad("CONCAT dim must be 0..3", reason);
            }
            if (s0.dtype != s1.dtype || s0.dtype != dd.dtype) {
                return api_bad("CONCAT: all tensors must have the same dtype", reason);
            }
            if (td_esz(s0) != 2 && td_esz(s0) != 4) {
                return api_no("CONCAT supports 2-byte and 4-byte element types only", reason);
            }
            for (int i = 0; i < 4; ++i) {
                if (i == (int) p.dim) {
                    if (td_ne(dd, i) != td_ne(s0, i) + td_ne(s1, i)) {
                        return api_bad("CONCAT: dst dim does not match the concatenation", reason);
                    }
                } else if (td_ne(s0, i) != td_ne(s1, i) || td_ne(s0, i) != td_ne(dd, i)) {
                    return api_bad("CONCAT: dims other than the concat dim must match", reason);
                }
            }
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_PAD: {
            const auto p = api_params<scrapgpu_pad_params>(desc);
            if (!f32(s0) || !f32(dd)) {
                return api_no("PAD expects F32 tensors", reason);
            }
            for (int i = 0; i < 4; ++i) {
                if (p.before[i] < 0 || p.after[i] < 0) {
                    return api_bad("PAD: negative padding", reason);
                }
                if (td_ne(dd, i) != td_ne(s0, i) + p.before[i] + p.after[i]) {
                    return api_bad("PAD: dst dim does not match src plus the padding", reason);
                }
            }
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_SUM_ROWS:
            if (!f32(s0) || !f32(dd)) {
                return api_no("SUM_ROWS expects F32 tensors", reason);
            }
            return td_ne(dd, 0) == 1 && td_ne(dd, 1) == td_ne(s0, 1) &&
                   td_ne(dd, 2) == td_ne(s0, 2) && td_ne(dd, 3) == td_ne(s0, 3)
                ? SCRAPGPU_STATUS_OK
                : api_bad("SUM_ROWS: dst must be [1, ne1, ne2, ne3] of src0", reason);

        case SCRAPGPU_OP_ARGSORT:
        case SCRAPGPU_OP_TOP_K: {
            const auto p = api_params<scrapgpu_argsort_params>(desc);
            if (api_enum_u32(p.order) > (uint32_t) SCRAPGPU_ARGSORT_DESC) {
                return api_no("unknown ARGSORT order", reason);
            }
            if (!f32(s0) || dd.dtype != SCRAPGPU_DTYPE_I32) {
                return api_no("ARGSORT/TOP_K expect F32 keys and I32 indices", reason);
            }
            if (td_ne(dd, 1) != td_ne(s0, 1) || td_ne(dd, 2) != td_ne(s0, 2) || td_ne(dd, 3) != td_ne(s0, 3)) {
                return api_bad("ARGSORT/TOP_K: dst must have the rows of src0", reason);
            }
            const int64_t k = td_ne(dd, 0);
            if (k < 1 || k > td_ne(s0, 0)) {
                return api_bad("ARGSORT/TOP_K: dst ne0 must be within the row", reason);
            }
            if (desc->op == SCRAPGPU_OP_TOP_K) {
                return p.top_k == 0 || (int64_t) p.top_k == k
                    ? SCRAPGPU_STATUS_OK
                    : api_bad("TOP_K: params top_k does not match dst ne0", reason);
            }
            return k == td_ne(s0, 0)
                ? SCRAPGPU_STATUS_OK
                : api_bad("ARGSORT: dst ne0 must match the row length", reason);
        }


        case SCRAPGPU_OP_GDN_STEP: {
            const auto state = s0, q = s1, k = s2;
            const auto v = api_load_desc(desc->inputs[3]);
            const auto gate = api_load_desc(desc->inputs[4]);
            const auto beta = api_load_desc(desc->inputs[5]);
            const auto out = api_load_desc(desc->outputs[1]);
            for (size_t i = 0; i < desc->input_count; ++i) {
                if (desc->inputs[i].dtype != SCRAPGPU_DTYPE_F32) return api_no("GDN_STEP needs F32 inputs", reason);
            }
            if (!f32(dd) || !f32(out)) return api_no("GDN_STEP needs F32 outputs", reason);
            const int64_t S = td_ne(state, 0), Hv = td_ne(state, 1), Hk = td_ne(q, 1);
            const auto shape = [&](const scrapgpu_tensor_desc & t, int64_t n0, int64_t n1, int64_t n2) {
                return td_ne(t, 0) == n0 && td_ne(t, 1) == n1 && td_ne(t, 2) == n2 && td_ne(t, 3) == 1;
            };
            if (S <= 0 || Hv <= 0 || Hk <= 0 || Hv % Hk != 0 || !shape(state, S, Hv, S) ||
                !shape(q, S, Hk, 1) || !shape(k, S, Hk, 1) || !shape(v, S, Hv, 1) ||
                !shape(gate, Hv, 1, 1) || !shape(beta, Hv, 1, 1) ||
                !td_same_ne(state, dd) || !shape(out, S, Hv, 1))
                return api_bad("GDN_STEP expects state[S,Hv,S], q/k[S,Hk], v/out[S,Hv], gate/beta[Hv] and Hv divisible by Hk", reason);
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_QSA_STEP: {
            const auto p = api_params<scrapgpu_qsa_step_params>(desc);
            if (s0.dtype != SCRAPGPU_DTYPE_I32 || dd.dtype != SCRAPGPU_DTYPE_I32)
                return api_no("QSA_STEP uses I32 tensors", reason);
            if (p.idx_block == 0 || p.idx_top_k == 0 ||
                (uint64_t) p.idx_top_k + p.idx_block - 1 > (uint64_t) INT32_MAX)
                return api_bad("QSA_STEP requires positive idx_block and idx_top_k with an I32 selection width", reason);
            for (int i = 1; i < 4; ++i) {
                if (td_ne(s0, i) != 1 || td_ne(dd, i) != 1)
                    return api_bad("QSA_STEP expects 1D input and output", reason);
            }
            return td_ne(s0, 0) == 1 && td_ne(dd, 0) == 4 ? SCRAPGPU_STATUS_OK
                : api_bad("QSA_STEP needs an I32 position and a four-element I32 output", reason);
        }

        case SCRAPGPU_OP_GDN_L2_NORM: {
            const auto p = api_params<scrapgpu_rms_norm_params>(desc);
            if (!f32(s0) || !f32(dd)) return api_no("GDN_L2_NORM expects F32 tensors", reason);
            if (!std::isfinite(p.eps) || p.eps <= 0.0f)
                return api_bad("GDN_L2_NORM eps must be positive and finite", reason);
            return td_same_ne(s0, dd) ? SCRAPGPU_STATUS_OK
                : api_bad("GDN_L2_NORM output shape must match input", reason);
        }

        case SCRAPGPU_OP_GDN_CONV_STEP: {
            const auto weights = s2;
            const auto next_state = dd;
            const auto out = api_load_desc(desc->outputs[1]);
            if (!f32(s0) || !f32(s1) || !f32(weights) || !f32(next_state) || !f32(out))
                return api_no("GDN_CONV_STEP expects F32 tensors", reason);
            const int64_t C = td_ne(s1, 0), dc = td_ne(weights, 0);
            const auto shape = [&](const scrapgpu_tensor_desc & t, int64_t n0, int64_t n1) {
                return td_ne(t, 0) == n0 && td_ne(t, 1) == n1 &&
                       td_ne(t, 2) == 1 && td_ne(t, 3) == 1;
            };
            if (dc < 2 || C <= 0 || !shape(s0, dc - 1, C) ||
                !shape(s1, C, 1) || !shape(weights, dc, C) ||
                !td_same_ne(s0, next_state) || !shape(out, C, 1))
                return api_bad("GDN_CONV_STEP expects state[d_conv-1,C], x[C], weights[d_conv,C] and d_conv >= 2", reason);
            return SCRAPGPU_STATUS_OK;
        }

        default:
            return api_no("op is not implemented by the CPU backend", reason);
    }
}

// ---------------------------------------------------------------------------------------------
// tensor resolution and aliasing (op_execute / plan_execute)
// ---------------------------------------------------------------------------------------------

struct cpu_tref {
    cpu_api_buffer *     b = nullptr;
    scrapgpu_tensor_desc d = {}; // normalized (api_load_desc)
};

scrapgpu_status api_resolve(cpu_api_stream * s, const scrapgpu_tensor_desc & raw, cpu_tref & out) {
    const scrapgpu_tensor_desc td = api_load_desc(raw);
    cpu_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr) {
        return api_invalid("tensor desc has no CPU buffer");
    }
    if (b->dev != s->dev) {
        return api_invalid("tensor buffer and stream belong to different devices");
    }
    const uint64_t nb = td_nbytes(td);
    if (td.offset > (uint64_t) b->size || nb > (uint64_t) b->size - td.offset) {
        return api_invalid("tensor exceeds its buffer");
    }
    out.b = b;
    out.d = td;
    return SCRAPGPU_STATUS_OK;
}

bool api_overlap(const cpu_tref & a, const cpu_tref & b) {
    if (a.b != b.b) {
        return false;
    }
    const uint64_t a0 = a.d.offset, a1 = a0 + td_nbytes(a.d);
    const uint64_t b0 = b.d.offset, b1 = b0 + td_nbytes(b.d);
    return a0 < b1 && b0 < a1;
}

// the identical view (same buffer, offset, dtype, shape and strides): the only aliasing the
// elementwise-safe ops allow
bool api_same_view(const cpu_tref & a, const cpu_tref & b) {
    if (a.b != b.b || a.d.offset != b.d.offset || a.d.dtype != b.d.dtype) {
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (a.d.ne[i] != b.d.ne[i] || a.d.nb[i] != b.d.nb[i]) {
            return false;
        }
    }
    return true;
}

// ops whose reference implementation reads every element before writing it back at the same
// address: an identical-view in-place is sound for them; every other op rejects any overlap
bool api_op_alias_safe(scrapgpu_op op) {
    switch (op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_SUB:
        case SCRAPGPU_OP_MUL:
        case SCRAPGPU_OP_DIV:
        case SCRAPGPU_OP_SCALE:
        case SCRAPGPU_OP_CLAMP:
        case SCRAPGPU_OP_SQR:
        case SCRAPGPU_OP_SQRT:
        case SCRAPGPU_OP_UNARY:
        case SCRAPGPU_OP_GLU:
        case SCRAPGPU_OP_RMS_NORM:
        case SCRAPGPU_OP_GDN_L2_NORM:
        case SCRAPGPU_OP_SOFT_MAX:
        case SCRAPGPU_OP_RMS_NORM_MUL:
        case SCRAPGPU_OP_ADD_RMS_NORM_MUL:
        case SCRAPGPU_OP_CPY:
        case SCRAPGPU_OP_DUP:
        case SCRAPGPU_OP_CONT:
            return true;
        default:
            return false;
    }
}

scrapgpu_status api_check_aliasing(scrapgpu_op op, const cpu_tref * in, size_t nin,
                                   const cpu_tref * out, size_t nout) {
    for (size_t i = 0; i < nout; ++i) {
        for (size_t j = i + 1; j < nout; ++j) {
            if (api_overlap(out[i], out[j])) {
                return api_invalid("op outputs overlap each other");
            }
        }
    }
    const bool elementwise = api_op_alias_safe(op);
    for (size_t i = 0; i < nout; ++i) {
        for (size_t j = 0; j < nin; ++j) {
            if (!api_overlap(out[i], in[j])) {
                continue;
            }
            // XeStrata's recurrence and convolution update independent state columns,
            // making only the exact state input/output pair safe for in-place execution.
            if ((op == SCRAPGPU_OP_GDN_STEP || op == SCRAPGPU_OP_GDN_CONV_STEP) &&
                i == 0 && j == 0 && api_same_view(out[i], in[j])) {
                continue;
            }
            if (!elementwise) {
                return api_invalid("the output overlaps an input and the reference is not in-place safe for this op");
            }
            if (!api_same_view(out[i], in[j])) {
                return api_invalid("a partially overlapping output/input view is not in-place safe");
            }
            // Norm weights reuse the first row across rows; overwriting any weight view
            // changes the weights used by later rows, even when the views are identical.
            if ((op == SCRAPGPU_OP_RMS_NORM_MUL && j == 1) ||
                (op == SCRAPGPU_OP_ADD_RMS_NORM_MUL && j == 2)) {
                return api_invalid("normalization outputs must not alias the weight input");
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// op dispatch
// ---------------------------------------------------------------------------------------------

// takes an int: the name is also used in plan-build errors for ops that failed the validation
const char * api_op_name(int op) {
    switch (op) {
        case SCRAPGPU_OP_ADD: return "ADD";
        case SCRAPGPU_OP_SUB: return "SUB";
        case SCRAPGPU_OP_MUL: return "MUL";
        case SCRAPGPU_OP_DIV: return "DIV";
        case SCRAPGPU_OP_SCALE: return "SCALE";
        case SCRAPGPU_OP_CLAMP: return "CLAMP";
        case SCRAPGPU_OP_SQR: return "SQR";
        case SCRAPGPU_OP_SQRT: return "SQRT";
        case SCRAPGPU_OP_UNARY: return "UNARY";
        case SCRAPGPU_OP_GLU: return "GLU";
        case SCRAPGPU_OP_RMS_NORM: return "RMS_NORM";
        case SCRAPGPU_OP_ROPE: return "ROPE";
        case SCRAPGPU_OP_SOFT_MAX: return "SOFT_MAX";
        case SCRAPGPU_OP_GET_ROWS: return "GET_ROWS";
        case SCRAPGPU_OP_SET_ROWS: return "SET_ROWS";
        case SCRAPGPU_OP_CPY: return "CPY";
        case SCRAPGPU_OP_DUP: return "DUP";
        case SCRAPGPU_OP_CONT: return "CONT";
        case SCRAPGPU_OP_MUL_MAT: return "MUL_MAT";
        case SCRAPGPU_OP_MUL_MAT_ID: return "MUL_MAT_ID";
        case SCRAPGPU_OP_ADD_ID: return "ADD_ID";
        case SCRAPGPU_OP_SSM_CONV: return "SSM_CONV";
        case SCRAPGPU_OP_CONCAT: return "CONCAT";
        case SCRAPGPU_OP_PAD: return "PAD";
        case SCRAPGPU_OP_SUM_ROWS: return "SUM_ROWS";
        case SCRAPGPU_OP_ARGSORT: return "ARGSORT";
        case SCRAPGPU_OP_TOP_K: return "TOP_K";
        case SCRAPGPU_OP_RMS_NORM_MUL: return "RMS_NORM_MUL";
        case SCRAPGPU_OP_ADD_RMS_NORM_MUL: return "ADD_RMS_NORM_MUL";
        case SCRAPGPU_OP_SET_ROWS_PAIR: return "SET_ROWS_PAIR";
        case SCRAPGPU_OP_MUL_MAT_MULTI: return "MUL_MAT_MULTI";
        case SCRAPGPU_OP_CONT_MUL_MAT: return "CONT_MUL_MAT";
        case SCRAPGPU_OP_GDN_STEP: return "GDN_STEP";
        case SCRAPGPU_OP_QSA_STEP: return "QSA_STEP";
        case SCRAPGPU_OP_GDN_L2_NORM: return "GDN_L2_NORM";
        case SCRAPGPU_OP_GDN_CONV_STEP: return "GDN_CONV_STEP";
        default: return "OP";
    }
}

scrapgpu_status api_exec_desc(cpu_api_stream * s, const scrapgpu_op_desc * desc) {
    const char * why = nullptr;
    const scrapgpu_status st = api_supports(s->dev, desc, &why);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    switch (desc->op) {
        case SCRAPGPU_OP_NONE:
        case SCRAPGPU_OP_RESHAPE:
        case SCRAPGPU_OP_VIEW:
        case SCRAPGPU_OP_PERMUTE:
        case SCRAPGPU_OP_TRANSPOSE:
            return SCRAPGPU_STATUS_OK; // metadata ops: nothing to compute
        default:
            break;
    }

    std::vector<cpu_tref> in_t(desc->input_count);
    std::vector<cpu_tref> out_t(desc->output_count);
    std::vector<scrapgpu_tensor_desc> in_d(desc->input_count);
    std::vector<scrapgpu_tensor_desc> out_d(desc->output_count);
    std::vector<void *> in_p(desc->input_count);
    std::vector<void *> out_p(desc->output_count);
    for (size_t i = 0; i < desc->input_count; ++i) {
        const scrapgpu_status r = api_resolve(s, desc->inputs[i], in_t[i]);
        if (r != SCRAPGPU_STATUS_OK) {
            return r;
        }
        in_d[i] = in_t[i].d;
        in_p[i] = in_t[i].b->data;
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        const scrapgpu_status r = api_resolve(s, desc->outputs[i], out_t[i]);
        if (r != SCRAPGPU_STATUS_OK) {
            return r;
        }
        out_d[i] = out_t[i].d;
        out_p[i] = out_t[i].b->data;
    }
    {
        const scrapgpu_status r = api_check_aliasing(desc->op, in_t.data(), in_t.size(), out_t.data(), out_t.size());
        if (r != SCRAPGPU_STATUS_OK) {
            return r;
        }
    }

    // the reference wants canonical descriptors (struct_size == sizeof) and one host base pointer
    // per tensor; desc->offset is applied by the reference itself
    scrapgpu_op_desc d2 = *desc;
    d2.struct_size = sizeof(d2);
    d2.inputs  = in_d.data();
    d2.outputs = out_d.data();
    return scrapgpu_ref_execute(&d2, in_p.data(), out_p.data());
}

// ---------------------------------------------------------------------------------------------
// devices
// ---------------------------------------------------------------------------------------------

size_t api_get_device_count(void * userdata) {
    (void) userdata;
    return g_devices.size();
}

scrapgpu_status api_get_device(void * userdata, size_t index, scrapgpu_device_t ** out_device) {
    (void) userdata;
    return api_guard([&]() -> scrapgpu_status {
        if (index >= g_devices.size()) {
            return api_invalid("device index out of range");
        }
        *out_device = reinterpret_cast<scrapgpu_device_t *>(g_devices[index].get());
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_device_get_info(scrapgpu_device_t * device, scrapgpu_device_info * info) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_device * d = api_device(device);
        if (d == nullptr) {
            return api_invalid("not a CPU device");
        }
        if (info == nullptr || info->struct_size < sizeof(size_t)) {
            return api_invalid("device info struct is too small");
        }

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
        static const char kIsa[] = "portable scalar + SSE2";
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
        static const char kIsa[] = "portable scalar + NEON autovectorization";
#else
        static const char kIsa[] = "portable scalar";
#endif

        scrapgpu_device_info full = {};
        full.struct_size            = sizeof(full);
        full.name                   = "CPU";
        full.vendor                 = "generic";
        full.platform_name          = "host";
        full.driver_version         = "scrapgpu_ref_execute";
        full.backend_name           = "cpu";
        full.reported_api_version   = "C++17";
        full.used_api_version       = kIsa;
        full.generation             = SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
        full.generation_reason      = "host processor; no SIMD requirement beyond the compiler baseline";
        full.total_memory_bytes     = d->total_memory;
        full.max_allocation_reported_bytes = d->total_memory;
        full.max_allocation_used_bytes     = d->total_memory;
        full.max_work_group_size    = 1;
        full.local_memory_bytes     = 0;
        full.compute_units          = d->compute_units;
        full.warp_or_wave_width     = 1;
        full.quirks                 = 0;
        full.probed_quirks          = 0;
        full.tuning_profile_name    = "portable";
        full.tuning_overrides       = "";

        const size_t caller_size = info->struct_size;
        std::memcpy(info, &full, std::min(caller_size, sizeof(full)));
        info->struct_size = caller_size;
        return SCRAPGPU_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------------------------

// 256-byte aligned host storage (the contract alignment of every backend)
void * api_alloc(size_t size) {
    const size_t n = size == 0 ? 1 : size;
#if defined(_WIN32)
    return _aligned_malloc(n, 256);
#else
    void * p = nullptr;
    return posix_memalign(&p, 256, n) == 0 ? p : nullptr;
#endif
}

void api_dealloc(void * p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

scrapgpu_status api_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_device * d = api_device(device);
        if (d == nullptr || out_buffer == nullptr) {
            return api_invalid("device or out_buffer is null");
        }
        *out_buffer = nullptr;
        uint8_t * data = static_cast<uint8_t *>(api_alloc(size));
        if (data == nullptr) {
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        std::unique_ptr<cpu_api_buffer> b(new (std::nothrow) cpu_api_buffer());
        if (b == nullptr) {
            api_dealloc(data);
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        b->hdr.backend = &g_cpu_vtable;
        b->dev = d;
        b->data = data;
        b->size = size;
        *out_buffer = reinterpret_cast<scrapgpu_buffer_t *>(b.release());
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_free(scrapgpu_buffer_t * buffer) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_buffer * b = api_buffer(buffer);
        if (b == nullptr) {
            return api_invalid("not a CPU buffer");
        }
        if (plan_refs_buffer(b)) {
            return api_invalid("buffer is still referenced by a live plan");
        }
        api_dealloc(b->data);
        delete b;
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking) {
    (void) blocking; // the host copy is synchronous either way
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_stream * s = api_stream(stream);
        cpu_api_buffer * b = api_buffer(buffer);
        if (s == nullptr || b == nullptr) {
            return api_invalid("stream or buffer is not a CPU object");
        }
        if (b->dev != s->dev) {
            return api_invalid("buffer and stream belong to different devices");
        }
        if (size == 0) {
            return SCRAPGPU_STATUS_OK;
        }
        if (offset > (uint64_t) b->size || size > b->size - (size_t) offset) {
            return api_invalid("write exceeds the buffer");
        }
        std::memmove(b->data + offset, data, size);
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking) {
    (void) blocking;
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_stream * s = api_stream(stream);
        cpu_api_buffer * b = api_buffer(buffer);
        if (s == nullptr || b == nullptr) {
            return api_invalid("stream or buffer is not a CPU object");
        }
        if (b->dev != s->dev) {
            return api_invalid("buffer and stream belong to different devices");
        }
        if (size == 0) {
            return SCRAPGPU_STATUS_OK;
        }
        if (offset > (uint64_t) b->size || size > b->size - (size_t) offset) {
            return api_invalid("read exceeds the buffer");
        }
        std::memmove(data, b->data + offset, size);
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking) {
    (void) blocking;
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_stream * s = api_stream(stream);
        cpu_api_buffer * db = api_buffer(dst);
        cpu_api_buffer * sb = api_buffer(src);
        if (s == nullptr || db == nullptr || sb == nullptr) {
            return api_invalid("stream or buffers are not CPU objects");
        }
        if (db->dev != s->dev || sb->dev != s->dev) {
            return api_invalid("buffers and stream belong to different devices");
        }
        if (size == 0) {
            return SCRAPGPU_STATUS_OK;
        }
        if (dst_offset > (uint64_t) db->size || size > db->size - (size_t) dst_offset ||
            src_offset > (uint64_t) sb->size || size > sb->size - (size_t) src_offset) {
            return api_invalid("copy exceeds a buffer");
        }
        // memmove: an in-place overlap inside one buffer copies as if through a temporary
        std::memmove(db->data + dst_offset, sb->data + src_offset, size);
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking) {
    (void) blocking;
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_stream * s = api_stream(stream);
        cpu_api_buffer * b = api_buffer(buffer);
        if (s == nullptr || b == nullptr) {
            return api_invalid("stream or buffer is not a CPU object");
        }
        if (b->dev != s->dev) {
            return api_invalid("buffer and stream belong to different devices");
        }
        if (size == 0) {
            return SCRAPGPU_STATUS_OK;
        }
        if (offset > (uint64_t) b->size || size > b->size - (size_t) offset) {
            return api_invalid("fill exceeds the buffer");
        }
        std::memset(b->data + offset, byte, size);
        return SCRAPGPU_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------
// streams (synchronous: stream_sync is a no-op)
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_stream_create(scrapgpu_device_t * device, scrapgpu_stream_t ** out_stream) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_device * d = api_device(device);
        if (d == nullptr || out_stream == nullptr) {
            return api_invalid("device or out_stream is null");
        }
        *out_stream = nullptr;
        std::unique_ptr<cpu_api_stream> s(new (std::nothrow) cpu_api_stream());
        if (s == nullptr) {
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        s->hdr.backend = &g_cpu_vtable;
        s->dev = d;
        *out_stream = reinterpret_cast<scrapgpu_stream_t *>(s.release());
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_stream_free(scrapgpu_stream_t * stream) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_stream * s = api_stream(stream);
        if (s == nullptr) {
            return api_invalid("not a CPU stream");
        }
        delete s;
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_stream_sync(scrapgpu_stream_t * stream) {
    cpu_api_stream * s = api_stream(stream);
    if (s == nullptr) {
        return api_invalid("not a CPU stream");
    }
    return SCRAPGPU_STATUS_OK; // all work is synchronous on this backend
}

// the device a stream runs on: the core uses the identity for the same-device fast path of
// scrapgpu_tensor_copy
scrapgpu_device_t * api_stream_device(scrapgpu_stream_t * stream) {
    cpu_api_stream * s = api_stream(stream);
    return s == nullptr ? nullptr : reinterpret_cast<scrapgpu_device_t *>(s->dev);
}

// ---------------------------------------------------------------------------------------------
// tensor transfers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_tensor_transfer(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * raw, uint8_t * host, size_t size, bool to_dev) {
    cpu_api_stream * s = api_stream(stream);
    if (s == nullptr || raw == nullptr) {
        return api_invalid("stream or tensor desc is null");
    }
    if (raw->struct_size < offsetof(scrapgpu_tensor_desc, offset) + sizeof(raw->offset)) {
        return api_invalid("tensor desc is too small");
    }
    const scrapgpu_tensor_desc td = api_load_desc(*raw);
    cpu_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr) {
        return api_invalid("tensor desc has no CPU buffer");
    }
    if (b->dev != s->dev) {
        return api_invalid("tensor buffer and stream belong to different devices");
    }
    if (td.offset > (uint64_t) b->size || size > b->size - (size_t) td.offset) {
        return api_invalid("transfer exceeds the buffer");
    }
    if (size > td_nbytes(td)) {
        return api_invalid("transfer size exceeds the tensor");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (td.layout != SCRAPGPU_LAYOUT_NATURAL) {
        return api_unsupported("the CPU backend only supports the natural layout (R32 is an OpenCL repacking)");
    }
    if (!td_contiguous(td)) {
        return api_unsupported("tensor is not contiguous; use buffer_read / buffer_write for raw strided transfers");
    }
    if (to_dev) {
        std::memmove(b->data + td.offset, host, size);
    } else {
        std::memmove(host, b->data + td.offset, size);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_tensor_upload(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, const void * host, size_t size) {
    return api_guard([&]() -> scrapgpu_status {
        return api_tensor_transfer(stream, desc, (uint8_t *) const_cast<void *>(host), size, true);
    });
}

scrapgpu_status api_tensor_download(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, void * host, size_t size) {
    return api_guard([&]() -> scrapgpu_status {
        return api_tensor_transfer(stream, desc, (uint8_t *) host, size, false);
    });
}

// ---------------------------------------------------------------------------------------------
// ops and plans
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_op_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason) {
    cpu_api_device * d = api_device(device);
    if (d == nullptr) {
        if (reason != nullptr) {
            *reason = "not a CPU device";
        }
        return api_invalid("not a CPU device");
    }
    return api_guard([&]() -> scrapgpu_status {
        return api_supports(d, desc, reason);
    });
}

scrapgpu_status api_op_execute(scrapgpu_stream_t * stream, const scrapgpu_op_desc * desc) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_stream * s = api_stream(stream);
        if (s == nullptr) {
            return api_invalid("not a CPU stream");
        }
        if (desc == nullptr) {
            return api_invalid("op desc is null");
        }
        return api_exec_desc(s, desc);
    });
}

scrapgpu_status api_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_device * d = api_device(device);
        if (d == nullptr || desc == nullptr || out_plan == nullptr ||
            desc->struct_size < offsetof(scrapgpu_plan_desc, node_count) + sizeof(desc->node_count) ||
            desc->node_count > 4096 || (desc->node_count > 0 && desc->nodes == nullptr)) {
            return api_invalid("device, plan desc or node list is invalid");
        }
        *out_plan = nullptr;

        std::unique_ptr<cpu_api_plan> plan(new (std::nothrow) cpu_api_plan());
        if (plan == nullptr) {
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        plan->hdr.backend = &g_cpu_vtable;
        plan->dev = d;
        plan->nodes.reserve(desc->node_count);
        for (size_t i = 0; i < desc->node_count; ++i) {
            const scrapgpu_plan_node & src = desc->nodes[i];
            if (src.struct_size < offsetof(scrapgpu_plan_node, op) + sizeof(src.op)) {
                return api_invalid("plan node is too small");
            }
            // every node must be supported: the plan fails to build instead of failing midway
            const char * why = nullptr;
            const scrapgpu_status st = api_supports(d, &src.op, &why);
            if (st != SCRAPGPU_STATUS_OK) {
                char buf[256];
                snprintf(buf, sizeof(buf), "plan node %zu (%s): %s", i, api_op_name((int) src.op.op), why != nullptr ? why : "not supported");
                scrapgpu_set_last_error(buf);
                return st;
            }
            cpu_api_plan_node n;
            n.desc = src.op;
            n.desc.struct_size = sizeof(n.desc);
            n.inputs.reserve(src.op.input_count);
            for (size_t j = 0; j < src.op.input_count; ++j) {
                n.inputs.push_back(api_load_desc(src.op.inputs[j]));
            }
            n.outputs.reserve(src.op.output_count);
            for (size_t j = 0; j < src.op.output_count; ++j) {
                n.outputs.push_back(api_load_desc(src.op.outputs[j]));
            }
            if (src.op.params != nullptr && src.op.params_size > 0) {
                const uint8_t * p = static_cast<const uint8_t *>(src.op.params);
                n.params.assign(p, p + src.op.params_size);
            }
            n.desc.inputs  = n.inputs.data();
            n.desc.outputs = n.outputs.data();
            n.desc.params  = n.params.empty() ? nullptr : n.params.data();
            // the desc pointers inside the params now reference the plan's copies of inputs[]
            if (src.op.op == SCRAPGPU_OP_ROPE && n.params.size() >= sizeof(scrapgpu_rope_params)) {
                auto * p = reinterpret_cast<scrapgpu_rope_params *>(n.params.data());
                p->freq_factors = p->freq_factors != nullptr && n.inputs.size() > 2 ? &n.inputs[2] : nullptr;
                p->n_offs       = p->n_offs != nullptr && n.inputs.size() > 3 ? &n.inputs[3] : nullptr;
            }
            if (src.op.op == SCRAPGPU_OP_SOFT_MAX && n.params.size() >= sizeof(scrapgpu_soft_max_params)) {
                auto * p = reinterpret_cast<scrapgpu_soft_max_params *>(n.params.data());
                p->mask = p->mask != nullptr && n.inputs.size() > 1 ? &n.inputs[1] : nullptr;
            }
            // the buffers the node references: buffer_free must refuse while this plan is alive
            for (const scrapgpu_tensor_desc & t : n.inputs) {
                if (cpu_api_buffer * rb = api_buffer(t.buffer)) {
                    plan->bufs.push_back(rb);
                }
            }
            for (const scrapgpu_tensor_desc & t : n.outputs) {
                if (cpu_api_buffer * rb = api_buffer(t.buffer)) {
                    plan->bufs.push_back(rb);
                }
            }
            plan->nodes.push_back(std::move(n));
        }
        for (cpu_api_buffer * rb : plan->bufs) {
            plan_ref_buffer(rb);
        }
        *out_plan = reinterpret_cast<scrapgpu_plan_t *>(plan.release());
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan_) {
    return api_guard([&]() -> scrapgpu_status {
        cpu_api_stream * s = api_stream(stream);
        auto * plan = reinterpret_cast<cpu_api_plan *>(plan_);
        if (s == nullptr || plan == nullptr || plan->hdr.backend != &g_cpu_vtable) {
            return api_invalid("stream or plan is not a CPU object");
        }
        if (plan->dev != s->dev) {
            return api_invalid("plan and stream belong to different devices");
        }
        for (auto & node : plan->nodes) {
            const scrapgpu_status st = api_exec_desc(s, &node.desc);
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
        }
        return SCRAPGPU_STATUS_OK;
    });
}

scrapgpu_status api_plan_free(scrapgpu_plan_t * plan_) {
    return api_guard([&]() -> scrapgpu_status {
        auto * plan = reinterpret_cast<cpu_api_plan *>(plan_);
        if (plan == nullptr || plan->hdr.backend != &g_cpu_vtable) {
            return api_invalid("not a CPU plan");
        }
        for (cpu_api_buffer * rb : plan->bufs) {
            plan_unref_buffer(rb);
        }
        delete plan;
        return SCRAPGPU_STATUS_OK;
    });
}

// ---------------------------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------------------------

const scrapgpu_backend_vtable g_cpu_vtable = {
    /* .struct_size        = */ sizeof(scrapgpu_backend_vtable),
    /* .name               = */ "cpu",
    /* .userdata           = */ nullptr,
    /* .get_device_count   = */ api_get_device_count,
    /* .get_device         = */ api_get_device,
    /* .device_get_info    = */ api_device_get_info,
    /* .buffer_alloc       = */ api_buffer_alloc,
    /* .buffer_free        = */ api_buffer_free,
    /* .buffer_write       = */ api_buffer_write,
    /* .buffer_read        = */ api_buffer_read,
    /* .buffer_copy        = */ api_buffer_copy,
    /* .buffer_fill        = */ api_buffer_fill,
    /* .stream_sync        = */ api_stream_sync,
    /* .op_supported       = */ api_op_supported,
    /* .plan_build         = */ api_plan_build,
    /* .plan_execute       = */ api_plan_execute,
    /* .plan_free          = */ api_plan_free,
    /* .stream_create      = */ api_stream_create,
    /* .stream_free        = */ api_stream_free,
    /* .tensor_upload      = */ api_tensor_upload,
    /* .tensor_download    = */ api_tensor_download,
    /* .op_execute         = */ api_op_execute,
    /* .stream_device      = */ api_stream_device,
};

// the physical memory of the host, best effort (0 when the platform does not report it)
uint64_t api_total_memory() {
#if defined(_WIN32)
    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    return GlobalMemoryStatusEx(&ms) ? (uint64_t) ms.ullTotalPhys : 0;
#elif defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    return sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0 ? mem : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGESIZE);
    return pages > 0 && page_size > 0 ? (uint64_t) pages * (uint64_t) page_size : 0;
#endif
}

uint32_t api_compute_units() {
#if defined(_WIN32)
    SYSTEM_INFO si = {};
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? si.dwNumberOfProcessors : 1;
#else
    const long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (uint32_t) n : 1;
#endif
}

} // namespace

scrapgpu_status scrapgpu_cpu_register(void) {
    static std::mutex reg_mutex;
    std::lock_guard<std::mutex> lock(reg_mutex);
    for (size_t i = 0; i < scrapgpu_get_device_count(); ++i) {
        scrapgpu_device_t * existing = nullptr;
        if (scrapgpu_get_device(i, &existing) == SCRAPGPU_STATUS_OK &&
            api_device(existing) != nullptr) {
            return SCRAPGPU_STATUS_OK;
        }
    }
    try {
        if (g_devices.empty()) {
            auto d = std::make_unique<cpu_api_device>();
            d->hdr.backend = &g_cpu_vtable;
            d->total_memory = api_total_memory();
            d->compute_units = api_compute_units();
            g_devices.push_back(std::move(d));
        }
        return scrapgpu_register_backend(&g_cpu_vtable);
    } catch (const std::bad_alloc &) {
        scrapgpu_set_last_error("out of host memory");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
}
