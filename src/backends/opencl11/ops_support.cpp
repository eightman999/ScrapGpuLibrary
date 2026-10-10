// The shape / type / size conditions under which the ops can run (ops.h), on the tensor
// descriptors. No ggml: the caller adds the checks a descriptor cannot carry (the 32-bit limits of
// the descriptor fields, the driver allocation limit of the tensors, the "view of a repacked tensor"
// rule and the ops disabled by name for debugging).

#include "ops.h"

#include "buffer.h"

#include <algorithm>
#include <cstdint>

// ADD / SUB / MUL / DIV: same conditions for the four kinds
bool cl11_supports_binary(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc & d) {
    return s0.dtype == SCRAPGPU_DTYPE_F32 && s1.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_F32 &&
           sgt::same_shape(s0, d);
}

// UNARY (CL11_UC_SILU .. CL11_UC_TANH) / SCALE / CLAMP / SQR / SQRT
bool cl11_supports_unary(const sgt::TensorDesc & s0, const sgt::TensorDesc & d, int32_t code) {
    if (code < CL11_UC_SILU || code > CL11_UC_CLAMP) {
        return false;
    }
    return s0.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_F32;
}

bool cl11_supports_glu(const sgt::TensorDesc & s0, const sgt::TensorDesc * s1, const sgt::TensorDesc & d, const scrapgpu_glu_params & p) {
    if (p.kind != SCRAPGPU_GLU_SWIGLU && p.kind != SCRAPGPU_GLU_REGLU) {
        return false;
    }
    if (s0.dtype != SCRAPGPU_DTYPE_F32 || d.dtype != SCRAPGPU_DTYPE_F32) {
        return false;
    }
    if (s1 != nullptr) {
        return s1->dtype == SCRAPGPU_DTYPE_F32 && sgt::same_shape(s0, *s1) && sgt::same_shape(s0, d);
    }
    return s0.ne[0] == 2 * d.ne[0];
}

bool cl11_supports_rms_norm(const sgt::TensorDesc & s0, const sgt::TensorDesc & d) {
    return s0.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_F32 &&
           s0.nb[0] == sizeof(float) && d.nb[0] == sizeof(float);
}

bool cl11_supports_soft_max(const sgt::TensorDesc & s0, const sgt::TensorDesc * mask, const sgt::TensorDesc * sinks,
                            const sgt::TensorDesc & d, const cl11_soft_max_params & p) {
    if (s0.dtype != SCRAPGPU_DTYPE_F32 || d.dtype != SCRAPGPU_DTYPE_F32 || sinks != nullptr) {
        return false; // attention sinks are not supported
    }
    if (p.max_bias != 0.0f) {
        return false; // ALiBi is not supported
    }
    if (s0.nb[0] != sizeof(float) || d.nb[0] != sizeof(float)) {
        return false;
    }
    if (mask != nullptr) {
        if (!cl11_is_float(mask->dtype) || mask->nb[0] != sgt::dtype_type_size(mask->dtype) ||
            mask->ne[0] != s0.ne[0] || mask->ne[1] < s0.ne[1]) {
            return false;
        }
    }
    return true;
}

bool cl11_supports_rope(const sgt::TensorDesc & s0, const sgt::TensorDesc * pos, const sgt::TensorDesc * ff,
                        const sgt::TensorDesc & d, const cl11_rope_params & p) {
    if (p.mode != SCRAPGPU_ROPE_MODE_NORMAL && p.mode != SCRAPGPU_ROPE_MODE_NEOX) {
        return false; // no mrope / vision
    }
    if (!cl11_is_float(s0.dtype) || d.dtype != s0.dtype || pos == nullptr || pos->dtype != SCRAPGPU_DTYPE_I32) {
        return false;
    }
    if (ff != nullptr && (ff->dtype != SCRAPGPU_DTYPE_F32 || ff->ne[0] < p.n_dims / 2 || ff->nb[0] != sizeof(float))) {
        return false;
    }
    return p.n_dims > 0 && p.n_dims % 2 == 0 && d.ne[0] % 2 == 0 && p.n_offs >= 0 && p.n_offs % 2 == 0 &&
           p.n_offs + p.n_dims <= d.ne[0] &&
           s0.nb[0] == sgt::dtype_type_size(s0.dtype) && d.nb[0] == sgt::dtype_type_size(d.dtype) &&
           pos->ne[0] >= d.ne[2];
}

bool cl11_supports_get_rows(const sgt::TensorDesc & s0, bool s0_repacked, const sgt::TensorDesc & ids, const sgt::TensorDesc & d) {
    return cl11_qt_of(s0.dtype) >= 0 && (!sgt::dtype_is_repackable(s0.dtype) || s0_repacked) &&
           ids.dtype == SCRAPGPU_DTYPE_I32 && d.dtype == SCRAPGPU_DTYPE_F32 &&
           s0.nb[0] == sgt::dtype_type_size(s0.dtype) &&
           s0.ne[0] == d.ne[0];
}

bool cl11_supports_set_rows(const sgt::TensorDesc & s0, const sgt::TensorDesc & ids, const sgt::TensorDesc & d) {
    return cl11_is_float(s0.dtype) && cl11_is_float(d.dtype) &&
           (ids.dtype == SCRAPGPU_DTYPE_I64 || ids.dtype == SCRAPGPU_DTYPE_I32) &&
           s0.nb[0] == sgt::dtype_type_size(s0.dtype) && d.nb[0] == sgt::dtype_type_size(d.dtype) &&
           s0.ne[0] == d.ne[0] && s0.ne[2] % ids.ne[1] == 0 && s0.ne[3] % ids.ne[2] == 0;
}

// CPY / DUP / CONT
bool cl11_supports_cpy(const sgt::TensorDesc & s0, const sgt::TensorDesc & d) {
    return cl11_is_float(s0.dtype) && cl11_is_float(d.dtype) && sgt::nelements(s0) == sgt::nelements(d);
}

bool cl11_supports_mul_mat(const sgt::TensorDesc & s0, bool s0_repacked, const sgt::TensorDesc & s1, const sgt::TensorDesc & d) {
    return cl11_qt_of(s0.dtype) >= 0 && (!sgt::dtype_is_repackable(s0.dtype) || s0_repacked) &&
           s1.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_F32 &&
           s0.nb[0] == sgt::dtype_type_size(s0.dtype) &&
           s1.nb[0] == sizeof(float) && d.nb[0] == sizeof(float) &&
           s0.ne[0] == s1.ne[0] &&
           s1.ne[2] % s0.ne[2] == 0 && s1.ne[3] % s0.ne[3] == 0;
}

bool cl11_supports_add_id(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc * ids, const sgt::TensorDesc & d) {
    return s0.dtype == SCRAPGPU_DTYPE_F32 && s1.dtype == SCRAPGPU_DTYPE_F32 && ids != nullptr && ids->dtype == SCRAPGPU_DTYPE_I32 && d.dtype == SCRAPGPU_DTYPE_F32 &&
           s0.nb[0] == sizeof(float) && s1.nb[0] == sizeof(float) && ids->nb[0] == sizeof(int32_t) && d.nb[0] == sizeof(float) &&
           s0.ne[0] == s1.ne[0] && s0.ne[3] == 1 && ids->ne[0] == s0.ne[1] && ids->ne[1] == s0.ne[2] && sgt::same_shape(s0, d);
}

bool cl11_supports_ssm_conv(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc & d) {
    return s0.dtype == SCRAPGPU_DTYPE_F32 && s1.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_F32 &&
           s0.nb[0] == sizeof(float) && s1.nb[0] == sizeof(float) && s0.ne[3] == 1 && s1.ne[2] == 1 && s1.ne[3] == 1 &&
           s0.ne[1] == s1.ne[1] && s0.ne[0] == s1.ne[0] - 1 + d.ne[1] && d.ne[0] == s0.ne[1];
}

bool cl11_supports_concat(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc & d, const scrapgpu_concat_params & p) {
    return s0.dtype == s1.dtype && s0.dtype == d.dtype && !sgt::dtype_is_quantized(s0.dtype) &&
           (sgt::dtype_type_size(s0.dtype) == 4 || sgt::dtype_type_size(s0.dtype) == 2) && sgt::dtype_block_size(s0.dtype) == 1 &&
           p.dim < sgt::kMaxDims;
}

bool cl11_supports_pad(const sgt::TensorDesc & s0, const sgt::TensorDesc & d) {
    return s0.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_F32 && s0.nb[0] == sizeof(float) && d.nb[0] == sizeof(float);
}

bool cl11_supports_sum_rows(const sgt::TensorDesc & s0, const sgt::TensorDesc & d) {
    return s0.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_F32 && d.ne[0] == 1;
}

// ARGSORT / TOP_K: the whole row is sorted in local memory
bool cl11_supports_argsort(const cl11_device * c, const sgt::TensorDesc & s0, const sgt::TensorDesc & d) {
    return s0.dtype == SCRAPGPU_DTYPE_F32 && d.dtype == SCRAPGPU_DTYPE_I32 && s0.nb[0] == sizeof(float) && d.nb[0] == sizeof(int32_t) &&
           cl11_sort_fits(c, s0.ne[0]);
}

// s0: experts [K, M, n_expert], s1: [K, 1 or n_used, n_tokens], ids: [n_used, n_tokens], dst [M, n_used, n_tokens]
bool cl11_supports_mul_mat_id(const cl11_device * c, const sgt::TensorDesc & s0, bool s0_repacked,
                              const sgt::TensorDesc & s1, const sgt::TensorDesc * ids, const sgt::TensorDesc & d) {
    if (ids == nullptr || ids->dtype != SCRAPGPU_DTYPE_I32 || s1.dtype != SCRAPGPU_DTYPE_F32 || d.dtype != SCRAPGPU_DTYPE_F32) {
        return false;
    }
    if (cl11_qt_of(s0.dtype) < 0 || (sgt::dtype_is_repackable(s0.dtype) && !s0_repacked)) {
        return false;
    }
    if (s0.nb[0] != sgt::dtype_type_size(s0.dtype) || s1.nb[0] != sizeof(float) || d.nb[0] != sizeof(float) || ids->nb[0] != sizeof(int32_t) ||
        s0.ne[3] != 1 || s1.ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 || s0.ne[0] != s1.ne[0]) {
        return false;
    }
    if (s1.ne[2] != ids->ne[1] || (s1.ne[1] != 1 && s1.ne[1] != ids->ne[0]) ||
        d.ne[0] != s0.ne[1] || d.ne[1] != ids->ne[0] || d.ne[2] != ids->ne[1]) {
        return false;
    }
    // the scratch buffers of the prompt processing path
    const size_t pairs = (size_t) (ids->ne[0] * ids->ne[1]);
    return pairs * (size_t) std::max(s0.ne[0], s0.ne[1]) * sizeof(float) <= cl11_max_buffer_size(c->info);
}
