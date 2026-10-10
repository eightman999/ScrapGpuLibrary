#include "scrapgpu/op.h"

#include "scrapgpu/backend.h"

#include <cstddef>

namespace {
bool is_f32(const scrapgpu_tensor_desc & t) {
    return t.dtype == SCRAPGPU_DTYPE_F32;
}

bool is_f16_or_f32(const scrapgpu_tensor_desc & t) {
    return t.dtype == SCRAPGPU_DTYPE_F16 || t.dtype == SCRAPGPU_DTYPE_F32;
}

bool is_quant_or_float_weight(const scrapgpu_tensor_desc & t) {
    switch (t.dtype) {
        case SCRAPGPU_DTYPE_F32:
        case SCRAPGPU_DTYPE_F16:
        case SCRAPGPU_DTYPE_Q4_0:
        case SCRAPGPU_DTYPE_Q8_0:
        case SCRAPGPU_DTYPE_Q4_K:
        case SCRAPGPU_DTYPE_Q6_K:
            return true;
        default:
            return false;
    }
}

scrapgpu_status invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

bool has_params(const scrapgpu_op_desc * desc, size_t min_size) {
    return desc->params != nullptr && desc->params_size >= min_size;
}

scrapgpu_status validate_counts(const scrapgpu_op_desc * desc, size_t inputs, size_t outputs) {
    if (desc == nullptr || desc->struct_size < offsetof(scrapgpu_op_desc, params_size) + sizeof(desc->params_size)) {
        return invalid("op desc is null or too small");
    }
    if (desc->input_count != inputs || desc->output_count != outputs || desc->inputs == nullptr || desc->outputs == nullptr) {
        return invalid("op desc has the wrong input or output count");
    }
    return SCRAPGPU_STATUS_OK;
}

// a sanity bound on the number of tensors of one op: every op of the API takes at most 4 inputs
// (SET_ROWS_PAIR) and 2 outputs; a larger count is always a malformed descriptor and the plan
// builders copy the arrays, so they must be bounded (audit H-3)
constexpr size_t kMaxOpTensors = 32;
// the params are fixed-size structs (the largest is a few hundred bytes); the plan builders and
// the remote protocol copy params_size bytes, so a malformed huge size would read out of bounds
constexpr size_t kMaxOpParams = 4096;
}

extern "C" {

scrapgpu_status scrapgpu_op_desc_validate(const scrapgpu_op_desc * desc) {
    if (desc == nullptr) {
        return invalid("op desc is null");
    }
    // everything below reads the fields through params_size: a descriptor that does not reach that
    // far cannot be validated (desc->op / desc->params must not be touched before this check)
    if (desc->struct_size < offsetof(scrapgpu_op_desc, params_size) + sizeof(desc->params_size)) {
        return invalid("op desc is null or too small");
    }
    // structural validation of every descriptor, before the early return of the metadata ops
    if (desc->input_count > kMaxOpTensors || desc->output_count > kMaxOpTensors) {
        return invalid("op desc has an invalid input or output count");
    }
    if ((desc->input_count > 0 && desc->inputs == nullptr) ||
        (desc->output_count > 0 && desc->outputs == nullptr)) {
        return invalid("op desc has a null input or output array");
    }
    if (desc->params == nullptr && desc->params_size > 0) {
        return invalid("op desc params_size is set but params is null");
    }
    if (desc->params_size > kMaxOpParams) {
        return invalid("op desc params_size is too large");
    }
    switch (desc->op) {
        case SCRAPGPU_OP_NONE:
        case SCRAPGPU_OP_RESHAPE:
        case SCRAPGPU_OP_VIEW:
        case SCRAPGPU_OP_PERMUTE:
        case SCRAPGPU_OP_TRANSPOSE:
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_SUB:
        case SCRAPGPU_OP_MUL:
        case SCRAPGPU_OP_DIV: {
            scrapgpu_status st = validate_counts(desc, 2, 1);
            if (st != SCRAPGPU_STATUS_OK) return st;
            return is_f32(desc->inputs[0]) && is_f32(desc->inputs[1]) && is_f32(desc->outputs[0]) ? SCRAPGPU_STATUS_OK : invalid("binary op expects F32 tensors");
        }
        case SCRAPGPU_OP_SCALE:
            if (!has_params(desc, sizeof(scrapgpu_scale_params))) return invalid("SCALE params are missing");
            return validate_counts(desc, 1, 1);
        case SCRAPGPU_OP_CLAMP:
            if (!has_params(desc, sizeof(scrapgpu_clamp_params))) return invalid("CLAMP params are missing");
            return validate_counts(desc, 1, 1);
        case SCRAPGPU_OP_SQR:
        case SCRAPGPU_OP_SQRT:
        case SCRAPGPU_OP_RMS_NORM:
        case SCRAPGPU_OP_SUM_ROWS: {
            scrapgpu_status st = validate_counts(desc, 1, 1);
            if (st != SCRAPGPU_STATUS_OK) return st;
            return is_f32(desc->inputs[0]) && is_f32(desc->outputs[0]) ? SCRAPGPU_STATUS_OK : invalid("row/unary op expects F32 tensors");
        }
        case SCRAPGPU_OP_UNARY:
            if (!has_params(desc, sizeof(scrapgpu_unary_params))) return invalid("UNARY params are missing");
            return validate_counts(desc, 1, 1);
        case SCRAPGPU_OP_GLU:
            if (!has_params(desc, sizeof(scrapgpu_glu_params))) return invalid("GLU params are missing");
            return validate_counts(desc, desc->input_count == 2 ? 2 : 1, 1);
        case SCRAPGPU_OP_ROPE:
            if (!has_params(desc, sizeof(scrapgpu_rope_params))) return invalid("ROPE params are missing");
            // inputs[]: [x, positions (I32), freq_factors (F32, optional)]; params->freq_factors / n_offs must be
            // NULL or the same descriptor as the corresponding inputs[] entry (reference uses inputs[]).
            if (validate_counts(desc, desc->input_count, 1) != SCRAPGPU_STATUS_OK) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            if (desc->input_count < 1 || desc->input_count > 3) return invalid("ROPE takes 1 to 3 inputs");
            if (!is_f16_or_f32(desc->inputs[0])) return invalid("ROPE data must be F32 or F16");
            if (desc->input_count >= 2 && desc->inputs[1].dtype != SCRAPGPU_DTYPE_I32) return invalid("ROPE positions must be I32");
            if (desc->input_count >= 3 && !is_f32(desc->inputs[2])) return invalid("ROPE freq factors must be F32");
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_OP_SOFT_MAX:
            if (!has_params(desc, sizeof(scrapgpu_soft_max_params))) return invalid("SOFT_MAX params are missing");
            // inputs[]: [x, mask (F32/F16, optional)]; params->mask must be NULL or match inputs[1] (reference uses inputs[]).
            if (validate_counts(desc, desc->input_count, 1) != SCRAPGPU_STATUS_OK) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            if (desc->input_count < 1 || desc->input_count > 2) return invalid("SOFT_MAX takes 1 or 2 inputs");
            if (!is_f32(desc->inputs[0])) return invalid("SOFT_MAX data must be F32");
            if (desc->input_count == 2 && !is_f16_or_f32(desc->inputs[1])) return invalid("SOFT_MAX mask must be F32 or F16");
            return SCRAPGPU_STATUS_OK;
        case SCRAPGPU_OP_GET_ROWS:
            if (!has_params(desc, sizeof(scrapgpu_get_rows_params))) return invalid("GET_ROWS params are missing");
            if (validate_counts(desc, 2, 1) != SCRAPGPU_STATUS_OK) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            return is_quant_or_float_weight(desc->inputs[0]) && desc->inputs[1].dtype == SCRAPGPU_DTYPE_I32 && is_f32(desc->outputs[0]) ? SCRAPGPU_STATUS_OK : invalid("GET_ROWS dtype contract failed");
        case SCRAPGPU_OP_SET_ROWS:
            if (!has_params(desc, sizeof(scrapgpu_set_rows_params))) return invalid("SET_ROWS params are missing");
            if (validate_counts(desc, 2, 1) != SCRAPGPU_STATUS_OK) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            return is_f16_or_f32(desc->inputs[0]) && is_f16_or_f32(desc->outputs[0]) &&
                           (desc->inputs[1].dtype == SCRAPGPU_DTYPE_I32 || desc->inputs[1].dtype == SCRAPGPU_DTYPE_I64)
                ? SCRAPGPU_STATUS_OK
                : invalid("SET_ROWS dtype contract failed");
        case SCRAPGPU_OP_CPY:
        case SCRAPGPU_OP_DUP:
        case SCRAPGPU_OP_CONT:
            if (desc->op == SCRAPGPU_OP_CPY && !has_params(desc, sizeof(scrapgpu_cpy_params))) return invalid("CPY params are missing");
            if (validate_counts(desc, 1, 1) != SCRAPGPU_STATUS_OK) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            return is_f16_or_f32(desc->inputs[0]) && is_f16_or_f32(desc->outputs[0]) ? SCRAPGPU_STATUS_OK : invalid("copy op supports F16/F32 only");
        case SCRAPGPU_OP_MUL_MAT:
            if (!has_params(desc, sizeof(scrapgpu_mul_mat_params))) return invalid("MUL_MAT params are missing");
            if (validate_counts(desc, 2, 1) != SCRAPGPU_STATUS_OK) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            return is_quant_or_float_weight(desc->inputs[0]) && is_f32(desc->inputs[1]) && is_f32(desc->outputs[0]) ? SCRAPGPU_STATUS_OK : invalid("MUL_MAT dtype contract failed");
        case SCRAPGPU_OP_MUL_MAT_ID:
            if (!has_params(desc, sizeof(scrapgpu_mul_mat_id_params))) return invalid("MUL_MAT_ID params are missing");
            return validate_counts(desc, 3, 1);
        case SCRAPGPU_OP_ADD_ID:
            return validate_counts(desc, 3, 1);
        case SCRAPGPU_OP_SSM_CONV:
            if (!has_params(desc, sizeof(scrapgpu_ssm_conv_params))) return invalid("SSM_CONV params are missing");
            return validate_counts(desc, 2, 1);
        case SCRAPGPU_OP_CONCAT:
            if (!has_params(desc, sizeof(scrapgpu_concat_params))) return invalid("CONCAT params are missing");
            return validate_counts(desc, 2, 1);
        case SCRAPGPU_OP_PAD:
            if (!has_params(desc, sizeof(scrapgpu_pad_params))) return invalid("PAD params are missing");
            return validate_counts(desc, 1, 1);
        case SCRAPGPU_OP_ARGSORT:
        case SCRAPGPU_OP_TOP_K:
            if (!has_params(desc, sizeof(scrapgpu_argsort_params))) return invalid("ARGSORT/TOP_K params are missing");
            if (validate_counts(desc, 1, 1) != SCRAPGPU_STATUS_OK) return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            return is_f32(desc->inputs[0]) && desc->outputs[0].dtype == SCRAPGPU_DTYPE_I32 ? SCRAPGPU_STATUS_OK : invalid("ARGSORT/TOP_K dtype contract failed");
        case SCRAPGPU_OP_RMS_NORM_MUL:
            // inputs [x, w], output [dst]
            if (!has_params(desc, sizeof(scrapgpu_rms_norm_params))) return invalid("RMS_NORM_MUL params are missing");
            return validate_counts(desc, 2, 1);
        case SCRAPGPU_OP_ADD_RMS_NORM_MUL:
            // inputs [a, b, w], outputs [add, dst]
            if (!has_params(desc, sizeof(scrapgpu_rms_norm_params))) return invalid("ADD_RMS_NORM_MUL params are missing");
            return validate_counts(desc, 3, 2);
        case SCRAPGPU_OP_SET_ROWS_PAIR:
            // inputs [src0_a, ids_a, src0_b, ids_b], outputs [dst_a, dst_b]
            return validate_counts(desc, 4, 2);
        case SCRAPGPU_OP_MUL_MAT_MULTI:
            // inputs [w_0 .. w_{n-1}, x (shared)], outputs [d_0 .. d_{n-1}], 2 <= n <= 3
            if (desc->struct_size < offsetof(scrapgpu_op_desc, params_size) + sizeof(desc->params_size)) {
                return invalid("op desc is too small");
            }
            if (desc->output_count < 2 || desc->output_count > 3 || desc->input_count != desc->output_count + 1) {
                return invalid("MUL_MAT_MULTI takes n+1 inputs and n outputs (2 <= n <= 3)");
            }
            return desc->inputs == nullptr || desc->outputs == nullptr ? invalid("op desc has the wrong input or output count") : SCRAPGPU_STATUS_OK;
        case SCRAPGPU_OP_CONT_MUL_MAT:
            // inputs [a (transposed F16 view), x], output [dst]
            return validate_counts(desc, 2, 1);
    }
    return invalid("unknown op");
}

scrapgpu_status scrapgpu_op_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason) {
    const scrapgpu_status valid = scrapgpu_op_desc_validate(desc);
    if (valid != SCRAPGPU_STATUS_OK) {
        if (reason != nullptr) {
            *reason = scrapgpu_last_error();
        }
        return valid;
    }
    if (device == nullptr) {
        scrapgpu_set_last_error("no backend registered for op support query");
        if (reason != nullptr) {
            *reason = scrapgpu_last_error();
        }
        return SCRAPGPU_STATUS_UNSUPPORTED;
    }
    const scrapgpu_backend_vtable * backend = reinterpret_cast<const scrapgpu_object_header *>(device)->backend;
    if (backend == nullptr || backend->op_supported == nullptr) {
        scrapgpu_set_last_error("no backend registered for op support query");
        if (reason != nullptr) {
            *reason = scrapgpu_last_error();
        }
        return SCRAPGPU_STATUS_UNSUPPORTED;
    }
    return backend->op_supported(device, desc, reason);
}

scrapgpu_status scrapgpu_op_execute(scrapgpu_stream_t * stream, const scrapgpu_op_desc * desc) {
    if (stream == nullptr || desc == nullptr) {
        scrapgpu_set_last_error("stream or op desc is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    const scrapgpu_backend_vtable * backend = reinterpret_cast<const scrapgpu_object_header *>(stream)->backend;
    if (backend == nullptr) {
        scrapgpu_set_last_error("stream does not belong to a registered backend");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    if (backend->op_execute == nullptr) {
        scrapgpu_set_last_error("the stream backend cannot execute ops");
        return SCRAPGPU_STATUS_UNSUPPORTED;
    }
    return backend->op_execute(stream, desc);
}

}
