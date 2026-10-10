// The ScrapGpu C API on top of the OpenCL 1.1 backend: scrapgpu_opencl11_register() registers a
// backend whose devices come from cl11_probe_device_infos. Buffers wrap a cl_mem, a stream shares
// the device's in-order queue and op_execute maps a scrapgpu_op_desc to the cl11_op_* functions
// (the same calls the ggml adapter makes; the ggml op_params decoding lives there instead).
//
// Fused ops and their desc layout (see include/scrapgpu/op.h):
//   SCRAPGPU_OP_RMS_NORM_MUL:     inputs [x, w], outputs [dst]         (dst = rms_norm(x, eps) * w)
//   SCRAPGPU_OP_ADD_RMS_NORM_MUL: inputs [a, b, w], outputs [add, dst] (add = a + b, one row each)
//   SCRAPGPU_OP_SET_ROWS_PAIR:    inputs [src0_a, ids_a, src0_b, ids_b], outputs [dst_a, dst_b]
//   SCRAPGPU_OP_MUL_MAT_MULTI:    inputs [w_0 .. w_{n-1}, x], outputs [d_0 .. d_{n-1}], 2 <= n <= 3
//   SCRAPGPU_OP_CONT_MUL_MAT:     inputs [a, x], outputs [dst]  (a is the transposed F16 view a CONT
//                                 would have copied; one launch, k_mvt_f16)

#include "device_state.h"
#include "executor.h"
#include "ops.h"
#include "r32.h"
#include "tensor.h"

#include "scrapgpu/backend.h"
#include "scrapgpu/opencl11.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// OpenCL 1.1 device info id; old macOS headers do not define it
#ifndef CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE
#define CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE 0x1067
#endif

namespace {

// ---------------------------------------------------------------------------------------------
// objects
// ---------------------------------------------------------------------------------------------

struct cl11_api_device {
    scrapgpu_object_header hdr; // must stay the first member
    cl11_device            dev;
    // device_get_info scratch: the info strings point into these members (stable until the next call)
    std::string            generation_reason;
    std::string            tuning_overrides;
};

struct cl11_api_buffer {
    scrapgpu_object_header hdr;
    cl11_api_device *      dev  = nullptr;
    cl_mem                 mem  = nullptr;
    size_t                 size = 0; // allocated bytes
};

struct cl11_api_stream {
    scrapgpu_object_header hdr;
    cl11_api_device *      dev = nullptr;
};

struct cl11_api_plan_node {
    scrapgpu_op_desc                   desc;
    std::vector<scrapgpu_tensor_desc>  inputs;
    std::vector<scrapgpu_tensor_desc>  outputs;
    std::vector<uint8_t>               params;
};

struct cl11_api_plan {
    scrapgpu_object_header           hdr;
    cl11_api_device *                dev = nullptr;
    std::vector<cl11_api_plan_node>  nodes;
};

extern const scrapgpu_backend_vtable g_cl11_vtable; // defined under "registration" below

std::vector<std::unique_ptr<cl11_api_device>> g_devices;

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

scrapgpu_status api_cl_error(const char * what, cl_int err) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s: %s", what, cl11_err_str(err));
    scrapgpu_set_last_error(buf);
    if (err == CL_OUT_OF_RESOURCES || err == CL_MEM_OBJECT_ALLOCATION_FAILURE || err == CL_OUT_OF_HOST_MEMORY) {
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    return SCRAPGPU_STATUS_BACKEND_ERROR;
}

scrapgpu_status api_fail(const char * what) {
    scrapgpu_set_last_error(what);
    return SCRAPGPU_STATUS_BACKEND_ERROR;
}

cl11_api_device * api_device(scrapgpu_device_t * device) {
    auto * d = reinterpret_cast<cl11_api_device *>(device);
    if (d == nullptr || d->hdr.backend != &g_cl11_vtable) {
        return nullptr;
    }
    return d;
}

cl11_api_buffer * api_buffer(scrapgpu_buffer_t * buffer) {
    auto * b = reinterpret_cast<cl11_api_buffer *>(buffer);
    if (b == nullptr || b->hdr.backend != &g_cl11_vtable) {
        return nullptr;
    }
    return b;
}

cl11_api_stream * api_stream(scrapgpu_stream_t * stream) {
    auto * s = reinterpret_cast<cl11_api_stream *>(stream);
    if (s == nullptr || s->hdr.backend != &g_cl11_vtable) {
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

// the desc of a tensor as the backend sees it, for the support check (no buffer needed)
sgt::TensorDesc api_sgt_desc(const scrapgpu_tensor_desc & in) {
    return sgt::desc_of(api_load_desc(in));
}

bool api_r32(const scrapgpu_tensor_desc & in) {
    return api_load_desc(in).layout == SCRAPGPU_LAYOUT_R32;
}

template <typename T>
T api_params(const scrapgpu_op_desc * desc) {
    T p = {};
    if (desc->params != nullptr) {
        std::memcpy(&p, desc->params, std::min(desc->params_size, sizeof(T)));
    }
    return p;
}

// ---------------------------------------------------------------------------------------------
// op_supported
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_check_tensors(cl11_api_device * d, const scrapgpu_op_desc * desc, sgt::TensorDesc * in, sgt::TensorDesc * out, const char ** reason) {
    auto check_one = [&](const scrapgpu_tensor_desc & raw, sgt::TensorDesc & td) -> scrapgpu_status {
        td = api_sgt_desc(raw);
        if (sgt::dtype_type_size(td.dtype) == 0) {
            *reason = "unknown tensor dtype";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        for (int i = 0; i < sgt::kMaxDims; ++i) {
            if (td.ne[i] < 0) {
                *reason = "negative tensor dim";
                return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            }
        }
        if (!sgt::fits_int32(td)) {
            *reason = "tensor exceeds the 32-bit limits of the kernel descriptors";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        if (raw.struct_size >= offsetof(scrapgpu_tensor_desc, layout) + sizeof(raw.layout) &&
            raw.layout == SCRAPGPU_LAYOUT_R32 && !sgt::r32_eligible(td)) {
            *reason = "the R32 layout needs a repackable dtype (Q4_0 / Q8_0 / Q4_K / Q6_K) and contiguous slices with 16-byte strides";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        if (d->dev.strict_alloc_dev && sgt::nbytes(td) > cl11_max_buffer_size(d->dev.info)) {
            *reason = "tensor is larger than the driver's single-allocation limit";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        return SCRAPGPU_STATUS_OK;
    };
    for (size_t i = 0; i < desc->input_count; ++i) {
        const scrapgpu_status st = check_one(desc->inputs[i], in[i]);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        const scrapgpu_status st = check_one(desc->outputs[i], out[i]);
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

bool api_quant_needs_r32(const sgt::TensorDesc & t, bool r32, const char ** reason) {
    if (sgt::dtype_is_repackable(t.dtype) && !r32) {
        *reason = "a quantized src0 must use the R32 layout (scrapgpu_tensor_upload packs it on the way)";
        return true;
    }
    return false;
}

// the conditions of SCRAPGPU_OP_CONT_MUL_MAT (the CONT skip of the adapter: k_mvt_f16 / k_mvt_g_f16)
scrapgpu_status api_supports_cont_mul_mat(cl11_api_device * d, const sgt::TensorDesc & a, const sgt::TensorDesc & x,
                                          const sgt::TensorDesc & dst, const char ** reason) {
    // the same decision as the ggml adapter (cl11_can_skip_cont): it depends on the work-group sizes the program was
    // built with (cl11_ensure_program may reduce them), so the program is built first
    cl11_device * c = &d->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c) || !cl11_ensure_program(c)) {
        return api_no("the OpenCL context or program could not be created", reason);
    }
    const cl11_config & cfg = c->cfg;
    if (c->no_mvt || cfg.wg_row % 32 != 0) {
        return api_no("the transposed-view mat-vec is disabled (GGML_CL11_NO_MVT) or the row work-group is too small", reason);
    }
    if (a.dtype != SCRAPGPU_DTYPE_F16 || x.dtype != SCRAPGPU_DTYPE_F32 || dst.dtype != SCRAPGPU_DTYPE_F32) {
        return api_no("CONT_MUL_MAT needs an F16 view, an F32 vector and an F32 dst", reason);
    }
    if (!sgt::is_transposed(a) || a.nb[1] != sgt::dtype_type_size(SCRAPGPU_DTYPE_F16)) {
        return api_no("CONT_MUL_MAT src0 must be a transposed F16 view (dim 1 contiguous)", reason);
    }
    if (x.ne[1] != 1 || x.nb[0] != sizeof(float) || dst.nb[0] != sizeof(float) || a.ne[0] != x.ne[0] || x.ne[2] % a.ne[2] != 0) {
        return api_no("CONT_MUL_MAT needs a single F32 column x with x.ne[0] == a.ne[0] and x.ne[2] a multiple of a.ne[2]", reason);
    }
    if (dst.ne[0] != a.ne[1] || dst.ne[1] != 1 || dst.ne[2] != x.ne[2] || dst.ne[3] != x.ne[3]) {
        return api_no("CONT_MUL_MAT dst must be [a.ne1, 1, x.ne2, x.ne3]", reason);
    }
    if (cfg.mvt_groups) {
        if (x.ne[3] % a.ne[3] != 0) {
            return api_no("CONT_MUL_MAT needs x.ne[3] a multiple of a.ne[3] on this device", reason);
        }
        return SCRAPGPU_STATUS_OK;
    }
    if (x.ne[3] != a.ne[3]) {
        return api_no("CONT_MUL_MAT needs x.ne[3] == a.ne[3] on this device", reason);
    }
    if (!cl11_attn_ok(c, cl11_make_td(a), cl11_make_td(x), a.ne[1], a.ne[2], x.ne[2] / a.ne[2], a.ne[0], true)) {
        return api_no("the transposed V cache kernel needs 16-byte aligned rows and at most 8 query heads per cache head", reason);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_supports(cl11_api_device * d, const scrapgpu_op_desc * desc, const char ** reason) {
    const char * reason_local = nullptr;
    if (reason == nullptr) {
        reason = &reason_local;
    }
    *reason = nullptr;
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

    sgt::TensorDesc in[8] = {}, out[3] = {};
    {
        const scrapgpu_status st = api_check_tensors(d, desc, in, out, reason);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }

    switch (desc->op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_SUB:
        case SCRAPGPU_OP_MUL:
        case SCRAPGPU_OP_DIV:
            return cl11_supports_binary(in[0], in[1], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("binary ops need F32 tensors and dst with the shape of src0", reason);

        case SCRAPGPU_OP_SCALE:
        case SCRAPGPU_OP_CLAMP:
        case SCRAPGPU_OP_SQR:
        case SCRAPGPU_OP_SQRT:
        case SCRAPGPU_OP_UNARY: {
            int32_t code;
            switch (desc->op) {
                case SCRAPGPU_OP_SCALE: code = CL11_UC_SCALE; break;
                case SCRAPGPU_OP_CLAMP: code = CL11_UC_CLAMP; break;
                case SCRAPGPU_OP_SQR:   code = CL11_UC_SQR;   break;
                case SCRAPGPU_OP_SQRT:  code = CL11_UC_SQRT;  break;
                default: {
                    const auto p = api_params<scrapgpu_unary_params>(desc);
                    if (p.kind < SCRAPGPU_UNARY_SILU || p.kind > SCRAPGPU_UNARY_TANH) {
                        return api_no("unknown UNARY kind", reason);
                    }
                    code = CL11_UC_SILU + (int32_t) p.kind;
                }
            }
            return cl11_supports_unary(in[0], out[0], code)
                ? SCRAPGPU_STATUS_OK
                : api_no("unary-like ops need F32 tensors", reason);
        }

        case SCRAPGPU_OP_GLU: {
            const auto p = api_params<scrapgpu_glu_params>(desc);
            const bool split = desc->input_count == 2;
            if (split != (p.split_inputs != 0)) {
                return api_bad("GLU split_inputs does not match the number of inputs", reason);
            }
            return cl11_supports_glu(in[0], split ? &in[1] : nullptr, out[0], p)
                ? SCRAPGPU_STATUS_OK
                : api_no("GLU needs F32 tensors, a supported kind and matching shapes", reason);
        }

        case SCRAPGPU_OP_RMS_NORM:
            return cl11_supports_rms_norm(in[0], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("RMS_NORM needs F32 tensors with contiguous rows", reason);

        case SCRAPGPU_OP_RMS_NORM_MUL: {
            const sgt::TensorDesc & w = in[1];
            const bool ok = cl11_supports_rms_norm(in[0], out[0]) && sgt::same_shape(in[0], out[0]) &&
                            w.dtype == SCRAPGPU_DTYPE_F32 && w.ne[0] == in[0].ne[0] && sgt::nelements(w) == in[0].ne[0] &&
                            w.nb[0] == sizeof(float);
            return ok ? SCRAPGPU_STATUS_OK : api_no("RMS_NORM_MUL needs [x, w] F32 with w a single row of ne0 floats", reason);
        }

        case SCRAPGPU_OP_ADD_RMS_NORM_MUL: {
            const sgt::TensorDesc & w = in[2];
            const bool row = sgt::nelements(in[0]) == in[0].ne[0];
            const bool ok = in[0].dtype == SCRAPGPU_DTYPE_F32 && in[1].dtype == SCRAPGPU_DTYPE_F32 &&
                            out[0].dtype == SCRAPGPU_DTYPE_F32 && out[1].dtype == SCRAPGPU_DTYPE_F32 &&
                            sgt::same_shape(in[0], in[1]) && sgt::same_shape(in[0], out[0]) && sgt::same_shape(in[0], out[1]) && row &&
                            in[0].nb[0] == sizeof(float) && in[1].nb[0] == sizeof(float) &&
                            out[0].nb[0] == sizeof(float) && out[1].nb[0] == sizeof(float) &&
                            w.dtype == SCRAPGPU_DTYPE_F32 && w.ne[0] == in[0].ne[0] && sgt::nelements(w) == in[0].ne[0] && w.nb[0] == sizeof(float);
            return ok ? SCRAPGPU_STATUS_OK : api_no("ADD_RMS_NORM_MUL needs [a, b, w] -> [add, dst] F32 single rows of the same shape", reason);
        }

        case SCRAPGPU_OP_SOFT_MAX: {
            const auto p = api_params<scrapgpu_soft_max_params>(desc);
            if (p.mask != nullptr && desc->input_count != 2) {
                return api_no("SOFT_MAX params->mask must be NULL or describe inputs[1]", reason);
            }
            const cl11_soft_max_params cp = { p.scale, 0.0f };
            return cl11_supports_soft_max(in[0], desc->input_count == 2 ? &in[1] : nullptr, nullptr, out[0], cp)
                ? SCRAPGPU_STATUS_OK
                : api_no("SOFT_MAX needs F32 tensors with contiguous rows (ALiBi and sinks are not supported)", reason);
        }

        case SCRAPGPU_OP_ROPE: {
            const auto p = api_params<scrapgpu_rope_params>(desc);
            if (desc->input_count < 2) {
                return api_bad("ROPE needs the positions tensor as inputs[1]", reason);
            }
            if (p.n_offs != nullptr) {
                return api_no("ROPE n_offs is not supported", reason);
            }
            if (p.freq_factors != nullptr && desc->input_count != 3) {
                return api_no("ROPE params->freq_factors must be NULL or describe inputs[2]", reason);
            }
            cl11_rope_params cp = {};
            cp.mode       = p.mode;
            cp.n_dims     = p.n_dims;
            cp.n_ctx_orig = p.n_ctx_orig;
            cp.freq_base  = p.freq_base;
            cp.freq_scale = p.freq_scale;
            cp.ext_factor = p.ext_factor;
            cp.attn_factor = p.attn_factor;
            cp.beta_fast  = p.beta_fast;
            cp.beta_slow  = p.beta_slow;
            return cl11_supports_rope(in[0], &in[1], desc->input_count == 3 ? &in[2] : nullptr, out[0], cp)
                ? SCRAPGPU_STATUS_OK
                : api_no("ROPE shape or mode is not supported (no mrope / vision)", reason);
        }

        case SCRAPGPU_OP_GET_ROWS:
            if (api_quant_needs_r32(in[0], api_r32(desc->inputs[0]), reason)) {
                return SCRAPGPU_STATUS_UNSUPPORTED;
            }
            return cl11_supports_get_rows(in[0], api_r32(desc->inputs[0]), in[1], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("GET_ROWS needs an F32/F16/quantized src0, I32 indices and an F32 dst", reason);

        case SCRAPGPU_OP_SET_ROWS:
            return cl11_supports_set_rows(in[0], in[1], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("SET_ROWS needs F32/F16 tensors and I32/I64 indices", reason);

        case SCRAPGPU_OP_SET_ROWS_PAIR: {
            const bool ok = cl11_supports_set_rows(in[0], in[1], out[0]) && cl11_supports_set_rows(in[2], in[3], out[1]) &&
                            in[0].dtype == in[2].dtype && out[0].dtype == out[1].dtype;
            return ok ? SCRAPGPU_STATUS_OK : api_no("SET_ROWS_PAIR needs two SET_ROWS with the same dtypes", reason);
        }

        case SCRAPGPU_OP_CPY:
        case SCRAPGPU_OP_DUP:
        case SCRAPGPU_OP_CONT:
            return cl11_supports_cpy(in[0], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("CPY/DUP/CONT support F32/F16 tensors of the same element count", reason);

        case SCRAPGPU_OP_MUL_MAT:
            if (api_quant_needs_r32(in[0], api_r32(desc->inputs[0]), reason)) {
                return SCRAPGPU_STATUS_UNSUPPORTED;
            }
            return cl11_supports_mul_mat(in[0], api_r32(desc->inputs[0]), in[1], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("MUL_MAT needs an F32 src1 and dst, contiguous rows and the src0 dtype of a kernel", reason);

        case SCRAPGPU_OP_MUL_MAT_MULTI: {
            const size_t n = desc->output_count;
            for (size_t i = 0; i < n; ++i) {
                if (api_quant_needs_r32(in[i], api_r32(desc->inputs[i]), reason)) {
                    return SCRAPGPU_STATUS_UNSUPPORTED;
                }
                if (!cl11_supports_mul_mat(in[i], api_r32(desc->inputs[i]), in[n], out[i])) {
                    return api_no("MUL_MAT_MULTI: every (w_i, x, d_i) must satisfy the MUL_MAT conditions", reason);
                }
            }
            return SCRAPGPU_STATUS_OK;
        }

        case SCRAPGPU_OP_MUL_MAT_ID:
            if (api_quant_needs_r32(in[0], api_r32(desc->inputs[0]), reason)) {
                return SCRAPGPU_STATUS_UNSUPPORTED;
            }
            return cl11_supports_mul_mat_id(&d->dev, in[0], api_r32(desc->inputs[0]), in[1], &in[2], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("MUL_MAT_ID needs [K,M,E] src0, [K,n_used|1,n_tokens] src1 and I32 [n_used,n_tokens] ids", reason);

        case SCRAPGPU_OP_ADD_ID:
            return cl11_supports_add_id(in[0], in[1], &in[2], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("ADD_ID needs F32 tensors and I32 ids matching the expert dims", reason);

        case SCRAPGPU_OP_SSM_CONV:
            return cl11_supports_ssm_conv(in[0], in[1], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("SSM_CONV needs F32 [seq, ch] state and [d_conv, ch] filter tensors", reason);

        case SCRAPGPU_OP_CONCAT: {
            const auto p = api_params<scrapgpu_concat_params>(desc);
            return cl11_supports_concat(in[0], in[1], out[0], p)
                ? SCRAPGPU_STATUS_OK
                : api_no("CONCAT needs same-dtype non-quantized tensors (2-byte or 4-byte elements)", reason);
        }

        case SCRAPGPU_OP_PAD:
            return cl11_supports_pad(in[0], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("PAD needs F32 tensors with contiguous rows", reason);

        case SCRAPGPU_OP_SUM_ROWS:
            return cl11_supports_sum_rows(in[0], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("SUM_ROWS needs F32 tensors and dst.ne[0] == 1", reason);

        case SCRAPGPU_OP_ARGSORT:
        case SCRAPGPU_OP_TOP_K: {
            const auto p = api_params<scrapgpu_argsort_params>(desc);
            if (desc->op == SCRAPGPU_OP_TOP_K && (p.top_k == 0 || out[0].ne[0] > in[0].ne[0])) {
                return api_no("TOP_K needs 1 <= top_k <= ne[0]", reason);
            }
            return cl11_supports_argsort(&d->dev, in[0], out[0])
                ? SCRAPGPU_STATUS_OK
                : api_no("ARGSORT/TOP_K need an F32 row that fits the device's local memory and an I32 dst", reason);
        }

        case SCRAPGPU_OP_CONT_MUL_MAT:
            return api_supports_cont_mul_mat(d, in[0], in[1], out[0], reason);

        default:
            return api_no("op is not implemented by the CL11 backend", reason);
    }
}

// ---------------------------------------------------------------------------------------------
// tensor resolution (op_execute / plan_execute)
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_resolve(cl11_api_stream * s, const scrapgpu_tensor_desc & raw, cl11_tensor & out) {
    const scrapgpu_tensor_desc td = api_load_desc(raw);
    cl11_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr) {
        return api_invalid("tensor desc has no CL11 buffer");
    }
    if (b->dev != s->dev) {
        return api_invalid("tensor buffer and stream belong to different devices");
    }
    const sgt::TensorDesc & td_sgt = out.d = sgt::desc_of(td);
    const uint64_t nb = sgt::nbytes(td_sgt);
    if (td_sgt.offset > b->size || nb > b->size - td_sgt.offset) {
        return api_invalid("tensor exceeds its buffer");
    }
    out.mem = b->mem;
    out.repacked = td.layout == SCRAPGPU_LAYOUT_R32;
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// op dispatch
// ---------------------------------------------------------------------------------------------

void api_rope_decode(const scrapgpu_op_desc * desc, cl11_rope_params & p) {
    const auto q = api_params<scrapgpu_rope_params>(desc);
    p.mode        = q.mode;
    p.n_dims      = q.n_dims;
    p.n_ctx_orig  = q.n_ctx_orig;
    p.n_offs      = 0; // the public params carry n_offs as an optional tensor, which CL11 does not support
    p.freq_base   = q.freq_base;
    p.freq_scale  = q.freq_scale;
    p.ext_factor  = q.ext_factor;
    p.attn_factor = q.attn_factor;
    p.beta_fast   = q.beta_fast;
    p.beta_slow   = q.beta_slow;
}

// the yarn corr dims of (n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow)
// Ported from ggml (MIT License, Copyright (c) 2023-2026 The ggml authors), see THIRD_PARTY_NOTICES.md
void api_rope_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    if (n_dims <= 0 || freq_base <= 0.0f || n_ctx_orig <= 0) {
        dims[0] = 0.0f;
        dims[1] = (float) std::max(n_dims - 1, 0);
        return;
    }
    // corr_dim(n_rot) = n_dims * log(n_ctx_orig / (n_rot * 2pi)) / (2 * log(base))
    const auto corr_dim = [&](float n_rot) {
        return (float) n_dims * logf((float) n_ctx_orig / (n_rot * 2.0f * (float) M_PI)) / (2.0f * logf(freq_base));
    };
    dims[0] = std::max(0.0f, floorf(corr_dim(beta_fast)));
    dims[1] = std::min((float) n_dims - 1.0f, ceilf(corr_dim(beta_slow)));
}

const char * api_op_name(scrapgpu_op op) {
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
        default: return "OP";
    }
}

// runs one validated + supported op (the buffers were resolved by the caller); the caller holds
// dev->mutex and has built the program
scrapgpu_status api_dispatch(cl11_api_device * d, const scrapgpu_op_desc * desc, const cl11_tensor * in, cl11_tensor * out) {
    cl11_device * c = &d->dev;
    bool ok = false;

    switch (desc->op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_SUB:
        case SCRAPGPU_OP_MUL:
        case SCRAPGPU_OP_DIV: {
            const cl11_binary_kind kind = desc->op == SCRAPGPU_OP_ADD ? CL11_BINARY_ADD :
                                          desc->op == SCRAPGPU_OP_SUB ? CL11_BINARY_SUB :
                                          desc->op == SCRAPGPU_OP_MUL ? CL11_BINARY_MUL : CL11_BINARY_DIV;
            ok = cl11_op_binary(c, out[0], in[0], in[1], kind);
            break;
        }
        case SCRAPGPU_OP_SCALE:
        case SCRAPGPU_OP_CLAMP:
        case SCRAPGPU_OP_SQR:
        case SCRAPGPU_OP_SQRT:
        case SCRAPGPU_OP_UNARY: {
            int32_t code;
            float p0 = 0.0f, p1 = 0.0f;
            switch (desc->op) {
                case SCRAPGPU_OP_SCALE: {
                    const auto p = api_params<scrapgpu_scale_params>(desc);
                    code = CL11_UC_SCALE; p0 = p.scale; p1 = p.bias;
                    break;
                }
                case SCRAPGPU_OP_CLAMP: {
                    const auto p = api_params<scrapgpu_clamp_params>(desc);
                    code = CL11_UC_CLAMP; p0 = p.min_value; p1 = p.max_value;
                    break;
                }
                case SCRAPGPU_OP_SQR:  code = CL11_UC_SQR;  break;
                case SCRAPGPU_OP_SQRT: code = CL11_UC_SQRT; break;
                default: {
                    const auto p = api_params<scrapgpu_unary_params>(desc);
                    code = CL11_UC_SILU + (int32_t) p.kind;
                    break;
                }
            }
            ok = cl11_op_unary(c, out[0], in[0], code, p0, p1);
            break;
        }
        case SCRAPGPU_OP_GLU: {
            const auto p = api_params<scrapgpu_glu_params>(desc);
            ok = cl11_op_glu(c, out[0], in[0], desc->input_count == 2 ? &in[1] : nullptr, p);
            break;
        }
        case SCRAPGPU_OP_RMS_NORM: {
            const auto p = api_params<scrapgpu_rms_norm_params>(desc);
            ok = cl11_op_rms_norm(c, out[0], in[0], p);
            break;
        }
        case SCRAPGPU_OP_RMS_NORM_MUL: {
            const auto p = api_params<scrapgpu_rms_norm_params>(desc);
            ok = cl11_op_rms_norm_mul(c, out[0], in[0], in[1], p);
            break;
        }
        case SCRAPGPU_OP_ADD_RMS_NORM_MUL: {
            const auto p = api_params<scrapgpu_rms_norm_params>(desc);
            ok = cl11_op_add_rms_norm_mul(c, out[1], in[0], in[1], in[2], out[0], p);
            break;
        }
        case SCRAPGPU_OP_SOFT_MAX: {
            const auto p = api_params<scrapgpu_soft_max_params>(desc);
            ok = cl11_op_soft_max(c, out[0], in[0], desc->input_count == 2 ? &in[1] : nullptr, p.scale);
            break;
        }
        case SCRAPGPU_OP_ROPE: {
            cl11_rope_params p;
            api_rope_decode(desc, p);
            api_rope_yarn_corr_dims(p.n_dims, p.n_ctx_orig, p.freq_base, p.beta_fast, p.beta_slow, p.corr_dims);
            ok = cl11_op_rope(c, out[0], in[0], in[1], desc->input_count == 3 ? &in[2] : nullptr, p);
            break;
        }
        case SCRAPGPU_OP_GET_ROWS:
            ok = cl11_op_get_rows(c, out[0], in[0], in[1]);
            break;
        case SCRAPGPU_OP_SET_ROWS:
            ok = cl11_op_set_rows(c, out[0], in[0], in[1]);
            break;
        case SCRAPGPU_OP_SET_ROWS_PAIR: {
            // one launch when the device pairs the K / V cache writes (fuse_set_rows), two SET_ROWS otherwise
            const int kid = K_SET_ROWS2 + (in[0].d.dtype == SCRAPGPU_DTYPE_F16 ? 2 : 0) + (out[0].d.dtype == SCRAPGPU_DTYPE_F16 ? 1 : 0);
            if (c->fuse_set_rows && c->kernels[kid] != nullptr) {
                const cl11_tensor pa[3] = { in[0], in[1], out[0] };
                const cl11_tensor pb[3] = { in[2], in[3], out[1] };
                ok = cl11_op_set_rows2(c, pa, pb);
            } else {
                ok = cl11_op_set_rows(c, out[0], in[0], in[1]) && cl11_op_set_rows(c, out[1], in[2], in[3]);
            }
            break;
        }
        case SCRAPGPU_OP_CPY:
        case SCRAPGPU_OP_DUP:
        case SCRAPGPU_OP_CONT:
            ok = cl11_op_cpy(c, out[0], in[0]);
            break;
        case SCRAPGPU_OP_MUL_MAT:
            ok = cl11_op_mul_mat(c, out[0], in[0], in[1]);
            break;
        case SCRAPGPU_OP_MUL_MAT_MULTI: {
            const int n = (int) desc->output_count;
            int r = 0;
            if (c->multi_mv && !c->no_fuse && !c->mvr_const_x && c->mv_max_cols >= 1) {
                // the fused kernel needs repacked weights of one type, a single F32 column and no batch
                bool fused = in[n].d.dtype == SCRAPGPU_DTYPE_F32 && in[n].d.ne[1] == 1 && in[n].d.ne[2] == 1 && in[n].d.ne[3] == 1;
                for (int i = 0; i < n && fused; ++i) {
                    fused = in[i].repacked && in[i].d.dtype == in[0].d.dtype && in[i].d.ne[0] == in[0].d.ne[0] &&
                            in[i].d.ne[2] == 1 && in[i].d.ne[3] == 1 && out[i].d.dtype == SCRAPGPU_DTYPE_F32 &&
                            out[i].d.ne[2] == 1 && out[i].d.ne[3] == 1;
                }
                if (fused) {
                    cl11_tensor dsts[3], ws[3];
                    for (int i = 0; i < n; ++i) {
                        dsts[i] = out[i];
                        ws[i]   = in[i];
                    }
                    r = cl11_op_mul_mat_multi(c, dsts, ws, in[n], n);
                }
            }
            if (r < 0) {
                return api_fail("the multi mat-vec launch failed");
            }
            if (r > 0) {
                ok = true;
                break;
            }
            ok = true;
            for (int i = 0; i < n && ok; ++i) {
                ok = cl11_op_mul_mat(c, out[i], in[i], in[n]);
            }
            break;
        }
        case SCRAPGPU_OP_MUL_MAT_ID:
            ok = cl11_op_mul_mat_id(c, out[0], in[0], in[1], in[2]);
            break;
        case SCRAPGPU_OP_ADD_ID:
            ok = cl11_op_add_id(c, out[0], in[0], in[1], in[2]);
            break;
        case SCRAPGPU_OP_SSM_CONV:
            ok = cl11_op_ssm_conv(c, out[0], in[0], in[1]);
            break;
        case SCRAPGPU_OP_CONCAT: {
            const auto p = api_params<scrapgpu_concat_params>(desc);
            ok = cl11_op_concat(c, out[0], in[0], in[1], p);
            break;
        }
        case SCRAPGPU_OP_PAD: {
            const auto p = api_params<scrapgpu_pad_params>(desc);
            ok = cl11_op_pad(c, out[0], in[0], p);
            break;
        }
        case SCRAPGPU_OP_SUM_ROWS:
            ok = cl11_op_sum_rows(c, out[0], in[0]);
            break;
        case SCRAPGPU_OP_ARGSORT:
        case SCRAPGPU_OP_TOP_K: {
            const auto p = api_params<scrapgpu_argsort_params>(desc);
            ok = cl11_op_argsort(c, out[0], in[0], p, desc->op == SCRAPGPU_OP_TOP_K);
            break;
        }
        case SCRAPGPU_OP_CONT_MUL_MAT:
            ok = cl11_op_mul_mat_t(c, out[0], in[0], in[1]);
            break;
        default:
            return api_unsupported("op is not implemented by the CL11 backend");
    }

    if (!ok) {
        return api_fail("the kernel launch failed");
    }
    // the writes invalidate the prepared-src1 / MUL_MAT_ID ids caches
    for (size_t i = 0; i < desc->output_count; ++i) {
        cl11_xprep_note_write(c, out[i].mem, (size_t) out[i].d.offset, (size_t) sgt::nbytes(out[i].d));
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_exec_desc(cl11_api_stream * s, const scrapgpu_op_desc * desc) {
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

    const char * why = nullptr;
    const scrapgpu_status st = api_supports(s->dev, desc, &why);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }

    cl11_tensor in[8] = {}, out[3] = {};
    for (size_t i = 0; i < desc->input_count; ++i) {
        const scrapgpu_status r = api_resolve(s, desc->inputs[i], in[i]);
        if (r != SCRAPGPU_STATUS_OK) {
            return r;
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        const scrapgpu_status r = api_resolve(s, desc->outputs[i], out[i]);
        if (r != SCRAPGPU_STATUS_OK) {
            return r;
        }
    }

    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_program(c)) {
        return api_fail("the OpenCL kernels could not be built");
    }
    if (c->exec.profile) {
        c->exec.prof_key = cl11_prof_key(c->exec, api_op_name(desc->op));
        c->exec.prof_stats[c->exec.prof_key].count++;
    }
    return api_dispatch(s->dev, desc, in, out);
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
    if (index >= g_devices.size()) {
        return api_invalid("device index out of range");
    }
    *out_device = reinterpret_cast<scrapgpu_device_t *>(g_devices[index].get());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_device_generation api_generation(const cl11_device_info & info, std::string & reason) {
    char buf[256];
    if (info.nv_cc_major == 2) {
        snprintf(buf, sizeof(buf), "NVIDIA compute capability %d.%d (Fermi)", info.nv_cc_major, info.nv_cc_minor);
        reason = buf;
        return SCRAPGPU_DEVICE_GENERATION_NV_FERMI;
    }
    if (info.nv_cc_major == 3) {
        snprintf(buf, sizeof(buf), "NVIDIA compute capability %d.%d (Kepler)", info.nv_cc_major, info.nv_cc_minor);
        reason = buf;
        return SCRAPGPU_DEVICE_GENERATION_NV_KEPLER;
    }
    const std::string lvendor = cl11_lower(info.vendor);
    if (lvendor.find("apple") != std::string::npos) {
        reason = "vendor " + info.vendor;
        return SCRAPGPU_DEVICE_GENERATION_APPLE;
    }
    switch (cl11_amd_generation(info)) {
        case CL11_AMD_GCN:
            reason = "AMD GCN (" + info.name + ")";
            return SCRAPGPU_DEVICE_GENERATION_AMD_GCN;
        case CL11_AMD_RDNA:
            reason = "AMD RDNA (" + info.name + ")";
            return SCRAPGPU_DEVICE_GENERATION_AMD_RDNA;
        case CL11_AMD_UNKNOWN:
            reason = "AMD GPU, generation not determined (" + info.name + ")";
            return SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
        default:
            break;
    }
    if (info.nv_cc_major > 0) {
        snprintf(buf, sizeof(buf), "NVIDIA compute capability %d.%d", info.nv_cc_major, info.nv_cc_minor);
        reason = buf;
    } else {
        reason = "unrecognised device (" + info.name + ", " + info.vendor + ")";
    }
    return SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
}

// the GGML_CL11_* / SCRAPGPU_OPENCL11_* environment variables that are set (reported as tuning overrides)
void api_tuning_overrides(cl11_api_device * d) {
    static const char * const vars[] = {
        "SCRAPGPU_OPENCL11_DEVICES",
        "GGML_CL11_DEVICES", "GGML_CL11_FORCE_GEN", "GGML_CL11_FAKE_MAX_ALLOC", "GGML_CL11_STRICT_ALLOC",
        "GGML_CL11_MAX_WG", "GGML_CL11_MAD", "GGML_CL11_MV_MAX_COLS", "GGML_CL11_MID_MAX_TOKENS",
        "GGML_CL11_MV_TPR", "GGML_CL11_MV_STAGED", "GGML_CL11_MVS_FAMILY", "GGML_CL11_MVS_MIN_K", "GGML_CL11_MVS_NRW",
        "GGML_CL11_WG_MV", "GGML_CL11_WG_ATT", "GGML_CL11_ATT_P", "GGML_CL11_MVT_GROUPS",
        "GGML_CL11_WG_MVR", "GGML_CL11_MVR_LDG", "GGML_CL11_MVR_DN", "GGML_CL11_MVR_NRR", "GGML_CL11_MVR_CONST_X",
        "GGML_CL11_NO_CPY_T", "GGML_CL11_NO_MVT", "GGML_CL11_NO_MVQ", "GGML_CL11_NO_FUSE", "GGML_CL11_NO_ASM",
        "GGML_CL11_NO_LDG", "GGML_CL11_NO_OPS", "GGML_CL11_MM_BN", "GGML_CL11_MM_VARIANT", "GGML_CL11_MMX",
        "GGML_CL11_NV_MAXRREG", "GGML_CL11_SPLIT_PROGRAMS", "GGML_CL11_MULTI_MV", "GGML_CL11_FUSE_SET_ROWS",
        "GGML_CL11_DN_NOPROBE", "GGML_CL11_PROFILE", "GGML_CL11_LAUNCH_LOG", "GGML_CL11_DEBUG",
    };
    d->tuning_overrides.clear();
    for (const char * v : vars) {
        if (getenv(v) == nullptr) {
            continue;
        }
        if (!d->tuning_overrides.empty()) {
            d->tuning_overrides += ' ';
        }
        d->tuning_overrides += v;
    }
}

scrapgpu_status api_device_get_info(scrapgpu_device_t * device, scrapgpu_device_info * info) {
    cl11_api_device * d = api_device(device);
    if (d == nullptr) {
        return api_invalid("not a CL11 device");
    }
    if (info == nullptr || info->struct_size < sizeof(size_t)) {
        return api_invalid("device info struct is too small");
    }
    if (d->generation_reason.empty()) {
        api_generation(d->dev.info, d->generation_reason);
        api_tuning_overrides(d);
    }

    const cl11_device_info & di = d->dev.info;
    const bool nvidia = di.nv_cc_major > 0 || cl11_lower(di.platform_name).find("nvidia") != std::string::npos ||
                        cl11_lower(di.vendor).find("nvidia") != std::string::npos;

    scrapgpu_device_info full = {};
    full.struct_size            = sizeof(full);
    full.name                   = di.name.c_str();
    full.vendor                 = di.vendor.c_str();
    full.platform_name          = di.platform_name.c_str();
    full.driver_version         = di.device_version.c_str();
    full.backend_name           = "opencl11";
    full.reported_api_version   = di.device_version.c_str();
    full.used_api_version       = "OpenCL 1.1";
    full.generation_reason      = d->generation_reason.c_str();
    full.total_memory_bytes     = di.global_mem;
    full.max_allocation_reported_bytes = di.max_alloc;
    full.max_allocation_used_bytes     = cl11_max_alloc_size(di, d->dev.strict_alloc_dev);
    full.max_work_group_size    = di.max_wg;
    full.local_memory_bytes     = di.local_mem;
    full.compute_units          = (uint32_t) std::max(0, di.compute_units);
    full.warp_or_wave_width     = (uint32_t) cl11_device_val<cl_uint>(di.device, CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0);
    full.tuning_profile_name    = cl11_tuning_for(di).name;
    full.tuning_overrides       = d->tuning_overrides.c_str();
    full.generation             = api_generation(di, d->generation_reason);

    uint64_t quirks = 0;
    if (nvidia) {
        quirks |= SCRAPGPU_DEVICE_QUIRK_LAZY_ALLOC | SCRAPGPU_DEVICE_QUIRK_SUBBUFFER_RELEASE_ORDER;
        if (!d->dev.strict_alloc_dev) {
            quirks |= SCRAPGPU_DEVICE_QUIRK_ALLOC_ABOVE_MAX_OK;
        }
        if (di.nv_cc_major == 3) {
            quirks |= SCRAPGPU_DEVICE_QUIRK_CONST_SUBBUFFER_OFFSET_BROKEN;
        }
    }
    if (cl11_lower(di.platform_name).find("clover") != std::string::npos) {
        quirks |= SCRAPGPU_DEVICE_QUIRK_LAUNCH_COST_GROWS_WITH_PROGRAM;
    }
    if (cl11_amd_generation(di) == CL11_AMD_GCN && cl11_lower(di.name).find("rusticl") != std::string::npos) {
        quirks |= SCRAPGPU_DEVICE_QUIRK_FMA_EMULATED;
    }
    full.quirks = quirks;
    full.probed_quirks = 0; // the denormal-products probe runs on demand at program build

    std::memcpy(info, &full, std::min(info->struct_size, sizeof(full)));
    info->struct_size = std::min(info->struct_size, sizeof(full));
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer) {
    cl11_api_device * d = api_device(device);
    if (d == nullptr || out_buffer == nullptr) {
        return api_invalid("device or out_buffer is null");
    }
    *out_buffer = nullptr;
    cl11_device * c = &d->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c)) {
        return api_fail("cannot create the OpenCL context");
    }
    cl_mem mem = nullptr;
    size_t alloc_size = 0;
    if (!cl11_alloc_buffer(c->context, c->queue, c->info, c->name.c_str(), "scrapgpu_buffer_alloc", c->strict_alloc_dev, size, mem, alloc_size)) {
        return api_cl_error("clCreateBuffer / the allocation probe failed", CL_MEM_OBJECT_ALLOCATION_FAILURE);
    }
    std::unique_ptr<cl11_api_buffer> b(new (std::nothrow) cl11_api_buffer());
    if (b == nullptr) {
        clReleaseMemObject(mem);
        return api_cl_error("out of host memory", CL_OUT_OF_HOST_MEMORY);
    }
    b->hdr.backend = &g_cl11_vtable;
    b->dev = d;
    b->mem = mem;
    b->size = alloc_size;
    *out_buffer = reinterpret_cast<scrapgpu_buffer_t *>(b.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_free(scrapgpu_buffer_t * buffer) {
    cl11_api_buffer * b = api_buffer(buffer);
    if (b == nullptr) {
        return api_invalid("not a CL11 buffer");
    }
    cl11_device * c = &b->dev->dev;
    {
        std::lock_guard<std::mutex> lock(c->mutex);
        cl11_release_const_subs_for_parent(c->const_subs, b->mem);
    }
    if (b->mem != nullptr) {
        clReleaseMemObject(b->mem);
    }
    delete b;
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking) {
    cl11_api_stream * s = api_stream(stream);
    cl11_api_buffer * b = api_buffer(buffer);
    if (s == nullptr || b == nullptr) {
        return api_invalid("stream or buffer is not a CL11 object");
    }
    if (b->dev != s->dev) {
        return api_invalid("buffer and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (offset > b->size || size > b->size - offset) {
        return api_invalid("write exceeds the buffer");
    }
    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c)) {
        return api_fail("cannot create the OpenCL context");
    }
    const cl_int err = clEnqueueWriteBuffer(c->queue, b->mem, blocking ? CL_TRUE : CL_FALSE, (size_t) offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return api_cl_error("clEnqueueWriteBuffer failed", err);
    }
    cl11_xprep_note_write(c, b->mem, (size_t) offset, size);
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking) {
    cl11_api_stream * s = api_stream(stream);
    cl11_api_buffer * b = api_buffer(buffer);
    if (s == nullptr || b == nullptr) {
        return api_invalid("stream or buffer is not a CL11 object");
    }
    if (b->dev != s->dev) {
        return api_invalid("buffer and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (offset > b->size || size > b->size - offset) {
        return api_invalid("read exceeds the buffer");
    }
    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c)) {
        return api_fail("cannot create the OpenCL context");
    }
    const cl_int err = clEnqueueReadBuffer(c->queue, b->mem, blocking ? CL_TRUE : CL_FALSE, (size_t) offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return api_cl_error("clEnqueueReadBuffer failed", err);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking) {
    (void) blocking; // clEnqueueCopyBuffer has no blocking flag
    cl11_api_stream * s = api_stream(stream);
    cl11_api_buffer * db = api_buffer(dst);
    cl11_api_buffer * sb = api_buffer(src);
    if (s == nullptr || db == nullptr || sb == nullptr) {
        return api_invalid("stream or buffers are not CL11 objects");
    }
    if (db->dev != s->dev || sb->dev != s->dev) {
        return api_invalid("buffers and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (dst_offset > db->size || size > db->size - dst_offset || src_offset > sb->size || size > sb->size - src_offset) {
        return api_invalid("copy exceeds a buffer");
    }
    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c)) {
        return api_fail("cannot create the OpenCL context");
    }
    const cl_int err = clEnqueueCopyBuffer(c->queue, sb->mem, db->mem, (size_t) src_offset, (size_t) dst_offset, size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        return api_cl_error("clEnqueueCopyBuffer failed", err);
    }
    cl11_xprep_note_write(c, db->mem, (size_t) dst_offset, size);
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking) {
    (void) blocking; // the fill is a kernel launch on the in-order queue
    cl11_api_stream * s = api_stream(stream);
    cl11_api_buffer * b = api_buffer(buffer);
    if (s == nullptr || b == nullptr) {
        return api_invalid("stream or buffer is not a CL11 object");
    }
    if (b->dev != s->dev) {
        return api_invalid("buffer and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (offset > b->size || size > b->size - offset) {
        return api_invalid("fill exceeds the buffer");
    }
    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_program(c)) { // k_fill_u32 / k_fill_u8
        return api_fail("the OpenCL kernels could not be built");
    }
    if (!cl11_fill(c->queue, c->exec, c->kernels[K_FILL_U32], c->kernels[K_FILL_U8], c->cfg, b->mem, (size_t) offset, size, byte)) {
        return api_fail("the fill kernel launch failed");
    }
    cl11_xprep_note_write(c, b->mem, (size_t) offset, size);
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// streams
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_stream_create(scrapgpu_device_t * device, scrapgpu_stream_t ** out_stream) {
    cl11_api_device * d = api_device(device);
    if (d == nullptr || out_stream == nullptr) {
        return api_invalid("device or out_stream is null");
    }
    *out_stream = nullptr;
    std::unique_ptr<cl11_api_stream> s(new (std::nothrow) cl11_api_stream());
    if (s == nullptr) {
        return api_cl_error("out of host memory", CL_OUT_OF_HOST_MEMORY);
    }
    s->hdr.backend = &g_cl11_vtable;
    s->dev = d;
    *out_stream = reinterpret_cast<scrapgpu_stream_t *>(s.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_stream_free(scrapgpu_stream_t * stream) {
    cl11_api_stream * s = api_stream(stream);
    if (s == nullptr) {
        return api_invalid("not a CL11 stream");
    }
    delete s;
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_stream_sync(scrapgpu_stream_t * stream) {
    cl11_api_stream * s = api_stream(stream);
    if (s == nullptr) {
        return api_invalid("not a CL11 stream");
    }
    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c)) {
        return api_fail("cannot create the OpenCL context");
    }
    const cl_int err = clFinish(c->queue);
    if (c->exec.profile) {
        cl11_prof_collect(c->exec);
    }
    if (err != CL_SUCCESS) {
        return api_cl_error("clFinish failed", err);
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// tensor transfers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_tensor_transfer(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * raw, uint8_t * host, size_t size, bool to_dev) {
    cl11_api_stream * s = api_stream(stream);
    if (s == nullptr || raw == nullptr) {
        return api_invalid("stream or tensor desc is null");
    }
    if (raw->struct_size < offsetof(scrapgpu_tensor_desc, offset) + sizeof(raw->offset)) {
        return api_invalid("tensor desc is too small");
    }
    const scrapgpu_tensor_desc td = api_load_desc(*raw);
    cl11_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr) {
        return api_invalid("tensor desc has no CL11 buffer");
    }
    if (b->dev != s->dev) {
        return api_invalid("tensor buffer and stream belong to different devices");
    }
    const sgt::TensorDesc d = sgt::desc_of(td);
    if (td.offset > b->size || size > b->size - td.offset) {
        return api_invalid("transfer exceeds the buffer");
    }
    if (size > sgt::nbytes(d)) {
        return api_invalid("transfer size exceeds the tensor");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_context(c)) {
        return api_fail("cannot create the OpenCL context");
    }
    if (td.layout == SCRAPGPU_LAYOUT_R32) {
        if (!sgt::r32_eligible(d)) {
            return api_unsupported("the R32 layout needs a repackable dtype and contiguous slices with 16-byte strides");
        }
        if (!cl11_rp_transfer(c, b->mem, d, host, 0, size, to_dev)) {
            return api_fail("the R32 pack/unpack transfer failed");
        }
    } else {
        if (!sgt::is_contiguous(d)) {
            return api_unsupported("tensor is not contiguous; use buffer_read / buffer_write for raw strided transfers");
        }
        const cl_int err = to_dev
            ? clEnqueueWriteBuffer(c->queue, b->mem, CL_TRUE, (size_t) td.offset, size, host, 0, nullptr, nullptr)
            : clEnqueueReadBuffer(c->queue, b->mem, CL_TRUE, (size_t) td.offset, size, host, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            return api_cl_error(to_dev ? "clEnqueueWriteBuffer failed" : "clEnqueueReadBuffer failed", err);
        }
    }
    if (to_dev) {
        cl11_xprep_note_write(c, b->mem, (size_t) td.offset, size);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_tensor_upload(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, const void * host, size_t size) {
    return api_tensor_transfer(stream, desc, (uint8_t *) const_cast<void *>(host), size, true);
}

scrapgpu_status api_tensor_download(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, void * host, size_t size) {
    return api_tensor_transfer(stream, desc, (uint8_t *) host, size, false);
}

// ---------------------------------------------------------------------------------------------
// ops and plans
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_op_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason) {
    cl11_api_device * d = api_device(device);
    if (d == nullptr) {
        if (reason != nullptr) {
            *reason = "not a CL11 device";
        }
        return api_invalid("not a CL11 device");
    }
    return api_supports(d, desc, reason);
}

scrapgpu_status api_op_execute(scrapgpu_stream_t * stream, const scrapgpu_op_desc * desc) {
    cl11_api_stream * s = api_stream(stream);
    if (s == nullptr) {
        return api_invalid("not a CL11 stream");
    }
    if (desc == nullptr) {
        return api_invalid("op desc is null");
    }
    cl11_device * c = &s->dev->dev;
    const scrapgpu_status st = api_exec_desc(s, desc);
    {
        std::lock_guard<std::mutex> lock(c->mutex);
        if (c->context_ready) {
            clFlush(c->queue);
            if (c->exec.profile) {
                clFinish(c->queue);
                cl11_prof_collect(c->exec);
            }
        }
    }
    return st;
}

scrapgpu_status api_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan) {
    cl11_api_device * d = api_device(device);
    if (d == nullptr || desc == nullptr || out_plan == nullptr ||
        desc->struct_size < offsetof(scrapgpu_plan_desc, node_count) + sizeof(desc->node_count) ||
        (desc->node_count > 0 && desc->nodes == nullptr)) {
        return api_invalid("device, plan desc or node list is invalid");
    }
    *out_plan = nullptr;

    std::unique_ptr<cl11_api_plan> plan(new (std::nothrow) cl11_api_plan());
    if (plan == nullptr) {
        return api_cl_error("out of host memory", CL_OUT_OF_HOST_MEMORY);
    }
    plan->hdr.backend = &g_cl11_vtable;
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
            snprintf(buf, sizeof(buf), "plan node %zu (%s): %s", i, api_op_name(src.op.op), why != nullptr ? why : "not supported");
            scrapgpu_set_last_error(buf);
            return st;
        }
        cl11_api_plan_node n;
        n.desc = src.op;
        n.inputs.assign(src.op.inputs, src.op.inputs + src.op.input_count);
        n.outputs.assign(src.op.outputs, src.op.outputs + src.op.output_count);
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
        plan->nodes.push_back(std::move(n));
    }
    *out_plan = reinterpret_cast<scrapgpu_plan_t *>(plan.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan_) {
    cl11_api_stream * s = api_stream(stream);
    auto * plan = reinterpret_cast<cl11_api_plan *>(plan_);
    if (s == nullptr || plan == nullptr || plan->hdr.backend != &g_cl11_vtable) {
        return api_invalid("stream or plan is not a CL11 object");
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
    cl11_device * c = &s->dev->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->context_ready) {
        clFlush(c->queue);
        if (c->exec.profile) {
            clFinish(c->queue);
            cl11_prof_collect(c->exec);
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_plan_free(scrapgpu_plan_t * plan_) {
    auto * plan = reinterpret_cast<cl11_api_plan *>(plan_);
    if (plan == nullptr || plan->hdr.backend != &g_cl11_vtable) {
        return api_invalid("not a CL11 plan");
    }
    delete plan;
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------------------------

const scrapgpu_backend_vtable g_cl11_vtable = {
    /* .struct_size        = */ sizeof(scrapgpu_backend_vtable),
    /* .name               = */ "opencl11",
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
};

} // namespace

scrapgpu_status scrapgpu_opencl11_register(void) {
    static std::mutex reg_mutex;
    static bool registered = false;
    static scrapgpu_status reg_status = SCRAPGPU_STATUS_OK;

    std::lock_guard<std::mutex> lock(reg_mutex);
    if (registered) {
        return reg_status;
    }
    registered = true;
    cl11_probe_device_infos([](const cl11_device_info & info, bool strict_alloc) {
        auto d = std::make_unique<cl11_api_device>();
        d->hdr.backend = &g_cl11_vtable;
        d->dev.info = info;
        d->dev.strict_alloc_dev = strict_alloc;
        d->dev.index = (int) g_devices.size();
        d->dev.name = "CL11_" + std::to_string(d->dev.index);
        g_devices.push_back(std::move(d));
    });
    reg_status = scrapgpu_register_backend(&g_cl11_vtable);
    return reg_status;
}
