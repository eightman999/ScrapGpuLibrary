#include "scrapgpu/op.h"

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
}

extern "C" {

scrapgpu_status scrapgpu_op_desc_validate(const scrapgpu_op_desc * desc) {
    if (desc == nullptr) {
        return invalid("op desc is null");
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
        case SCRAPGPU_OP_ADD_RMS_NORM_MUL:
        case SCRAPGPU_OP_SET_ROWS_PAIR:
        case SCRAPGPU_OP_MUL_MAT_MULTI:
        case SCRAPGPU_OP_CONT_MUL_MAT:
            return desc->input_count > 0 && desc->output_count > 0 ? SCRAPGPU_STATUS_OK : invalid("fused op needs inputs and outputs");
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
    scrapgpu_set_last_error("no backend registered for op support query");
    if (reason != nullptr) {
        *reason = scrapgpu_last_error();
    }
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

}
