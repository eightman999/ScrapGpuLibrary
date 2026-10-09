// CPU reference implementations of the ops. The semantics follow the CL11
// kernels in src/backends/opencl11/kernels (and ggml's CPU backend): shapes,
// broadcasting, strides (nb, in bytes) and the quantized natural block format
// behave like the GPU path described in docs/API_CONTRACT.md.

#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace scrapgpu {
namespace ref {

namespace {

inline float ld_f32(const uint8_t * p) {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline float ld_f16(const uint8_t * p) {
    return half_to_float(load_u16(p));
}

inline int32_t ld_i32(const uint8_t * p) {
    int32_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline int64_t ld_i64(const uint8_t * p) {
    int64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline void st_f32(uint8_t * p, float v) {
    std::memcpy(p, &v, sizeof(v));
}

inline void st_i32(uint8_t * p, int32_t v) {
    std::memcpy(p, &v, sizeof(v));
}

inline void st_f16(uint8_t * p, float v) {
    store_u16(p, float_to_half(v));
}

bool load_float(scrapgpu_dtype dtype, const uint8_t * p, float * out) {
    switch (dtype) {
        case SCRAPGPU_DTYPE_F32:
            *out = ld_f32(p);
            return true;
        case SCRAPGPU_DTYPE_F16:
            *out = ld_f16(p);
            return true;
        default:
            return false;
    }
}

bool store_float(scrapgpu_dtype dtype, uint8_t * p, float v) {
    switch (dtype) {
        case SCRAPGPU_DTYPE_F32:
            st_f32(p, v);
            return true;
        case SCRAPGPU_DTYPE_F16:
            st_f16(p, v);
            return true;
        default:
            return false;
    }
}

struct RTensor {
    const scrapgpu_tensor_desc * desc = nullptr;
    const uint8_t * base = nullptr;

    int64_t ne(int i) const {
        return desc->ne[i] > 0 ? desc->ne[i] : 1;
    }
    uint64_t nb(int i) const {
        return desc->nb[i];
    }
    scrapgpu_dtype dtype() const {
        return desc->dtype;
    }
    size_t esz() const {
        return scrapgpu_dtype_type_size(desc->dtype);
    }

    const uint8_t * ptr(int64_t i0 = 0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) const {
        return base + static_cast<uint64_t>(i0) * desc->nb[0] + static_cast<uint64_t>(i1) * desc->nb[1] +
               static_cast<uint64_t>(i2) * desc->nb[2] + static_cast<uint64_t>(i3) * desc->nb[3];
    }
};

struct WTensor {
    scrapgpu_tensor_desc * desc = nullptr;
    uint8_t * base = nullptr;

    int64_t ne(int i) const {
        return desc->ne[i] > 0 ? desc->ne[i] : 1;
    }
    uint64_t nb(int i) const {
        return desc->nb[i];
    }
    scrapgpu_dtype dtype() const {
        return desc->dtype;
    }
    size_t esz() const {
        return scrapgpu_dtype_type_size(desc->dtype);
    }

    uint8_t * ptr(int64_t i0 = 0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) const {
        return base + static_cast<uint64_t>(i0) * desc->nb[0] + static_cast<uint64_t>(i1) * desc->nb[1] +
               static_cast<uint64_t>(i2) * desc->nb[2] + static_cast<uint64_t>(i3) * desc->nb[3];
    }
};

template <typename A, typename B>
bool same_ne(const A & a, const B & b) {
    for (int i = 0; i < 4; ++i) {
        if (a.ne(i) != b.ne(i)) {
            return false;
        }
    }
    return true;
}

bool nelements_ok(const int64_t ne[4], int64_t * out) {
    int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (ne[i] > 0 && ne[i] > std::numeric_limits<int64_t>::max() / n) {
            return false;
        }
        n *= ne[i];
    }
    *out = n;
    return true;
}

bool tensor_span_ok(const scrapgpu_tensor_desc * t) {
    if (t->offset > (std::numeric_limits<uint64_t>::max)() / 2) {
        return false;
    }
    uint64_t span = t->offset;
    for (int i = 0; i < 4; ++i) {
        const int64_t n = t->ne[i] > 0 ? t->ne[i] : 1;
        const uint64_t count = static_cast<uint64_t>(n - 1);
        if (t->nb[i] != 0 && count > ((std::numeric_limits<uint64_t>::max)() / 2 - span) / t->nb[i]) {
            return false;
        }
        span += count * t->nb[i];
        if (span > (std::numeric_limits<uint64_t>::max)() / 2) {
            return false;
        }
    }
    return true;
}

scrapgpu_status check_io_desc(const scrapgpu_tensor_desc * t, const void * data, const char * what) {
    if (t == nullptr || t->struct_size < sizeof(scrapgpu_tensor_desc)) {
        return ref_invalid("tensor desc is null or too small");
    }
    if (data == nullptr) {
        return ref_invalid(what);
    }
    if (!tensor_span_ok(t)) {
        return ref_invalid("tensor strides or offset overflow the address space");
    }
    if (t->dtype != SCRAPGPU_DTYPE_F32 && t->dtype != SCRAPGPU_DTYPE_F16 && t->dtype != SCRAPGPU_DTYPE_I32 &&
        t->dtype != SCRAPGPU_DTYPE_I64 && t->dtype != SCRAPGPU_DTYPE_I16 && t->dtype != SCRAPGPU_DTYPE_BF16 &&
        !dtype_is_quantized(t->dtype)) {
        return ref_unsupported("tensor dtype is not implemented");
    }
    if (dtype_is_quantized(t->dtype) && t->layout != SCRAPGPU_LAYOUT_NATURAL) {
        return ref_unsupported("the reference only reads quantized tensors in the natural layout");
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status make_r(const scrapgpu_tensor_desc * desc, const void * data, RTensor * out, const char * what) {
    const scrapgpu_status st = check_io_desc(desc, data, what);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    out->desc = desc;
    out->base = static_cast<const uint8_t *>(data) + desc->offset;
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status make_w(scrapgpu_tensor_desc * desc, void * data, WTensor * out, const char * what) {
    const scrapgpu_status st = check_io_desc(desc, data, what);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    out->desc = desc;
    out->base = static_cast<uint8_t *>(data) + desc->offset;
    return SCRAPGPU_STATUS_OK;
}

template <typename T>
scrapgpu_status require_f32(const T & t, const char * what) {
    if (t.dtype() != SCRAPGPU_DTYPE_F32) {
        return ref_unsupported(what);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status require_contiguous_rows(const RTensor & t, const char * what) {
    if (t.nb(0) != t.esz()) {
        return ref_unsupported(what);
    }
    if (dtype_is_quantized(t.dtype()) && (t.ne(0) % static_cast<int64_t>(scrapgpu_dtype_block_size(t.dtype()))) != 0) {
        return ref_unsupported("quantized row length must be a multiple of the block size");
    }
    return SCRAPGPU_STATUS_OK;
}

// loads a whole row of src0 starting at `p` (nb0 == element size) as floats
scrapgpu_status load_row(const RTensor & t, const uint8_t * p, int64_t n, std::vector<float> * out) {
    out->resize(static_cast<size_t>(n));
    if (dtype_is_quantized(t.dtype())) {
        return dequantize_row(t.dtype(), p, n, out->data());
    }
    for (int64_t i = 0; i < n; ++i) {
        if (!load_float(t.dtype(), p + static_cast<uint64_t>(i) * t.nb(0), &(*out)[static_cast<size_t>(i)])) {
            return ref_unsupported("row dtype cannot be read as float");
        }
    }
    return SCRAPGPU_STATUS_OK;
}

int64_t wrap_index(int64_t i, int64_t n) {
    const int64_t m = i % n;
    return m < 0 ? m + n : m;
}

// ---------------------------------------------------------------------------------------------
// element-wise
// ---------------------------------------------------------------------------------------------

scrapgpu_status exec_binary(scrapgpu_op op, const RTensor & s0, const RTensor & s1, const WTensor & d) {
    if (d.dtype() != SCRAPGPU_DTYPE_F32 || s0.dtype() != SCRAPGPU_DTYPE_F32 || s1.dtype() != SCRAPGPU_DTYPE_F32) {
        return ref_unsupported("binary ops expect F32 tensors");
    }
    if (!same_ne(s0, d)) {
        return ref_invalid("binary op: src0 must have the shape of dst");
    }
    for (int i = 0; i < 4; ++i) {
        if (s1.ne(i) > d.ne(i)) {
            return ref_invalid("binary op: src1 cannot be larger than dst");
        }
    }
    for (int64_t i3 = 0; i3 < d.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < d.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < d.ne(1); ++i1) {
                for (int64_t i0 = 0; i0 < d.ne(0); ++i0) {
                    const float a = ld_f32(s0.ptr(i0, i1, i2, i3));
                    const float b = ld_f32(s1.ptr(i0 % s1.ne(0), i1 % s1.ne(1), i2 % s1.ne(2), i3 % s1.ne(3)));
                    float y = 0.0f;
                    switch (op) {
                        case SCRAPGPU_OP_ADD: y = a + b; break;
                        case SCRAPGPU_OP_SUB: y = a - b; break;
                        case SCRAPGPU_OP_MUL: y = a * b; break;
                        case SCRAPGPU_OP_DIV: y = a / b; break;
                        default: return ref_unsupported("not a binary op");
                    }
                    st_f32(d.ptr(i0, i1, i2, i3), y);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_unary(scrapgpu_op op, const RTensor & s, const WTensor & d, const void * params, size_t params_size) {
    if (s.dtype() != SCRAPGPU_DTYPE_F32 || d.dtype() != SCRAPGPU_DTYPE_F32) {
        return ref_unsupported("unary-like ops expect F32 tensors");
    }
    if (!same_ne(s, d)) {
        return ref_invalid("unary-like op: src0 must have the shape of dst");
    }
    for (int64_t idx = 0; idx < d.ne(0) * d.ne(1) * d.ne(2) * d.ne(3); ++idx) {
        int64_t r = idx;
        const int64_t i0 = r % d.ne(0);
        r /= d.ne(0);
        const int64_t i1 = r % d.ne(1);
        r /= d.ne(1);
        const int64_t i2 = r % d.ne(2);
        const int64_t i3 = r / d.ne(2);
        const float x = ld_f32(s.ptr(i0, i1, i2, i3));
        float y = 0.0f;
        switch (op) {
            case SCRAPGPU_OP_SQR:
                y = x * x;
                break;
            case SCRAPGPU_OP_SQRT:
                y = std::sqrt(x);
                break;
            case SCRAPGPU_OP_SCALE: {
                const auto * p = static_cast<const scrapgpu_scale_params *>(params);
                if (params == nullptr || params_size < sizeof(*p)) {
                    return ref_invalid("SCALE params are missing");
                }
                y = x * p->scale + p->bias;
                break;
            }
            case SCRAPGPU_OP_CLAMP: {
                const auto * p = static_cast<const scrapgpu_clamp_params *>(params);
                if (params == nullptr || params_size < sizeof(*p)) {
                    return ref_invalid("CLAMP params are missing");
                }
                y = std::min(std::max(x, p->min_value), p->max_value);
                break;
            }
            case SCRAPGPU_OP_UNARY: {
                const auto * p = static_cast<const scrapgpu_unary_params *>(params);
                if (params == nullptr || params_size < sizeof(*p)) {
                    return ref_invalid("UNARY params are missing");
                }
                switch (p->kind) {
                    case SCRAPGPU_UNARY_SILU:    y = x / (1.0f + std::exp(-x)); break;
                    case SCRAPGPU_UNARY_RELU:    y = std::max(x, 0.0f); break;
                    case SCRAPGPU_UNARY_NEG:     y = -x; break;
                    case SCRAPGPU_UNARY_ABS:     y = std::fabs(x); break;
                    case SCRAPGPU_UNARY_SIGMOID: y = 1.0f / (1.0f + std::exp(-x)); break;
                    case SCRAPGPU_UNARY_TANH:    y = std::tanh(x); break;
                    default: return ref_unsupported("unary kind is not implemented");
                }
                break;
            }
            default:
                return ref_unsupported("not a unary-like op");
        }
        st_f32(d.ptr(i0, i1, i2, i3), y);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_glu(const RTensor & s0, const RTensor & s1, bool has_s1, const WTensor & d, const scrapgpu_glu_params * p) {
    if (s0.dtype() != SCRAPGPU_DTYPE_F32 || d.dtype() != SCRAPGPU_DTYPE_F32) {
        return ref_unsupported("GLU expects F32 tensors");
    }
    if (has_s1 != (p->split_inputs != 0)) {
        return ref_invalid("GLU: split_inputs does not match the number of inputs");
    }
    if (has_s1 && (s1.dtype() != SCRAPGPU_DTYPE_F32 || !same_ne(s1, d) || !same_ne(s0, d))) {
        return ref_invalid("GLU with split inputs needs src0, src1 and dst of the same shape");
    }
    const int64_t nc = d.ne(0);
    if (!has_s1) {
        for (int i = 1; i < 4; ++i) {
            if (s0.ne(i) != d.ne(i)) {
                return ref_invalid("GLU: src0 and dst differ outside of dim 0");
            }
        }
        if (s0.ne(0) != 2 * nc) {
            return ref_invalid("GLU: src0 dim 0 must be twice the dst dim 0");
        }
    }
    const bool swiglu = p->kind == SCRAPGPU_GLU_SWIGLU;
    for (int64_t i3 = 0; i3 < d.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < d.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < d.ne(1); ++i1) {
                for (int64_t i0 = 0; i0 < nc; ++i0) {
                    float x;
                    float g;
                    if (has_s1) {
                        x = ld_f32(s0.ptr(i0, i1, i2, i3));
                        g = ld_f32(s1.ptr(i0, i1, i2, i3));
                    } else {
                        x = ld_f32(s0.ptr(i0 + (p->swapped ? nc : 0), i1, i2, i3));
                        g = ld_f32(s0.ptr(i0 + (p->swapped ? 0 : nc), i1, i2, i3));
                    }
                    const float a = swiglu ? x / (1.0f + std::exp(-x)) : std::max(x, 0.0f);
                    st_f32(d.ptr(i0, i1, i2, i3), a * g);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// norms
// ---------------------------------------------------------------------------------------------

scrapgpu_status read_eps(const void * params, size_t params_size, float * eps) {
    if (params == nullptr || params_size < sizeof(scrapgpu_rms_norm_params)) {
        return ref_invalid("RMS_NORM-family ops need scrapgpu_rms_norm_params");
    }
    const auto * p = static_cast<const scrapgpu_rms_norm_params *>(params);
    *eps = p->eps;
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_rms_norm(const RTensor & s, const WTensor & d, float eps) {
    const scrapgpu_status st = require_f32(s, "RMS_NORM expects F32 tensors");
    if (st != SCRAPGPU_STATUS_OK) return st;
    const scrapgpu_status st2 = require_f32(d, "RMS_NORM expects F32 tensors");
    if (st2 != SCRAPGPU_STATUS_OK) return st2;
    if (!same_ne(s, d)) {
        return ref_invalid("RMS_NORM: src0 must have the shape of dst");
    }
    for (int64_t i3 = 0; i3 < s.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < s.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < s.ne(1); ++i1) {
                double sum = 0.0;
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    const float v = ld_f32(s.ptr(i0, i1, i2, i3));
                    sum += static_cast<double>(v) * static_cast<double>(v);
                }
                const float scale = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(s.ne(0))) + eps);
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    st_f32(d.ptr(i0, i1, i2, i3), ld_f32(s.ptr(i0, i1, i2, i3)) * scale);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_rms_norm_mul(const RTensor & s, const RTensor & w, const WTensor & d, float eps) {
    const scrapgpu_status st = require_f32(s, "RMS_NORM_MUL expects F32 tensors");
    if (st != SCRAPGPU_STATUS_OK) return st;
    const scrapgpu_status st2 = require_f32(d, "RMS_NORM_MUL expects F32 tensors");
    if (st2 != SCRAPGPU_STATUS_OK) return st2;
    const scrapgpu_status st3 = require_f32(w, "RMS_NORM_MUL expects F32 tensors");
    if (st3 != SCRAPGPU_STATUS_OK) return st3;
    if (!same_ne(s, d) || w.ne(0) != s.ne(0)) {
        return ref_invalid("RMS_NORM_MUL: dst must match src0 and the weight must be a row vector");
    }
    for (int64_t i3 = 0; i3 < s.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < s.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < s.ne(1); ++i1) {
                double sum = 0.0;
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    const float v = ld_f32(s.ptr(i0, i1, i2, i3));
                    sum += static_cast<double>(v) * static_cast<double>(v);
                }
                const float scale = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(s.ne(0))) + eps);
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    st_f32(d.ptr(i0, i1, i2, i3), (ld_f32(s.ptr(i0, i1, i2, i3)) * scale) * ld_f32(w.ptr(i0)));
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_add_rms_norm_mul(const RTensor & a, const RTensor & b, const RTensor & w, const WTensor & s_out, const WTensor & y_out, float eps) {
    if (a.dtype() != SCRAPGPU_DTYPE_F32 || b.dtype() != SCRAPGPU_DTYPE_F32 || w.dtype() != SCRAPGPU_DTYPE_F32 ||
        s_out.dtype() != SCRAPGPU_DTYPE_F32 || y_out.dtype() != SCRAPGPU_DTYPE_F32) {
        return ref_unsupported("ADD_RMS_NORM_MUL expects F32 tensors");
    }
    if (!same_ne(a, b) || !same_ne(a, s_out) || !same_ne(a, y_out) || w.ne(0) != a.ne(0)) {
        return ref_invalid("ADD_RMS_NORM_MUL: a, b, the outputs and the weight row must have matching shapes");
    }
    for (int64_t i3 = 0; i3 < a.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < a.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < a.ne(1); ++i1) {
                double sum = 0.0;
                for (int64_t i0 = 0; i0 < a.ne(0); ++i0) {
                    const float v = ld_f32(a.ptr(i0, i1, i2, i3)) + ld_f32(b.ptr(i0, i1, i2, i3));
                    st_f32(s_out.ptr(i0, i1, i2, i3), v);
                    sum += static_cast<double>(v) * static_cast<double>(v);
                }
                const float scale = 1.0f / std::sqrt(static_cast<float>(sum / static_cast<double>(a.ne(0))) + eps);
                for (int64_t i0 = 0; i0 < a.ne(0); ++i0) {
                    st_f32(y_out.ptr(i0, i1, i2, i3), (ld_f32(s_out.ptr(i0, i1, i2, i3)) * scale) * ld_f32(w.ptr(i0)));
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// rope / soft_max
// ---------------------------------------------------------------------------------------------

float rope_ramp(float low, float high, int i0) {
    const float y = (static_cast<float>(i0 / 2) - low) / std::max(0.001f, high - low);
    return 1.0f - std::min(1.0f, std::max(0.0f, y));
}

scrapgpu_status exec_rope(const RTensor & s, const RTensor & pos, const RTensor * ff, const WTensor & d, const scrapgpu_rope_params * p) {
    if ((s.dtype() != SCRAPGPU_DTYPE_F32 && s.dtype() != SCRAPGPU_DTYPE_F16) || d.dtype() != s.dtype()) {
        return ref_unsupported("ROPE expects F32 or F16 tensors of the same dtype");
    }
    if (pos.dtype() != SCRAPGPU_DTYPE_I32) {
        return ref_unsupported("ROPE positions must be I32");
    }
    if (p->n_offs != nullptr) {
        return ref_unsupported("ROPE n_offs is not supported by the reference (the params carry no value)");
    }
    if (!same_ne(s, d)) {
        return ref_invalid("ROPE: src0 must have the shape of dst");
    }
    if (s.nb(0) != s.esz() || d.nb(0) != d.esz()) {
        return ref_unsupported("ROPE needs contiguous rows (nb0 == element size)");
    }
    if (p->n_dims <= 0 || p->n_dims % 2 != 0 || p->n_dims > s.ne(0) || s.ne(0) % 2 != 0) {
        return ref_invalid("ROPE: n_dims must be a positive even number not above ne0 (ne0 even)");
    }
    if (pos.ne(0) < d.ne(2)) {
        return ref_invalid("ROPE: not enough positions for the sequence dimension");
    }
    if (ff != nullptr && (ff->dtype() != SCRAPGPU_DTYPE_F32 || ff->ne(0) < p->n_dims / 2)) {
        return ref_invalid("ROPE: freq factors must be F32 with at least n_dims/2 entries");
    }

    const int n_dims = p->n_dims;
    const int n_offs = 0;
    const bool neox = p->mode == SCRAPGPU_ROPE_MODE_NEOX;
    const float theta_scale = std::pow(p->freq_base, -2.0f / static_cast<float>(n_dims));

    // yarn correction window (ggml_rope_yarn_corr_dims)
    float corr0 = 0.0f;
    float corr1 = static_cast<float>(n_dims - 1);
    if (p->ext_factor != 0.0f && p->freq_base > 0.0f && p->n_ctx_orig > 0) {
        const double two_log_base = 2.0 * std::log(static_cast<double>(p->freq_base));
        if (two_log_base != 0.0) {
            const auto corr_dim = [&](float n_rot) {
                return n_dims * std::log(static_cast<double>(p->n_ctx_orig) / (static_cast<double>(n_rot) * 2.0 * 3.14159265358979323846)) / two_log_base;
            };
            corr0 = std::max(0.0, std::floor(corr_dim(p->beta_fast)));
            corr1 = std::min(static_cast<double>(n_dims - 1), std::ceil(corr_dim(p->beta_slow)));
        }
    }

    const int64_t hp = s.ne(0) / 2;
    for (int64_t i3 = 0; i3 < d.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < d.ne(2); ++i2) {
            const float position = static_cast<float>(ld_i32(pos.ptr(i2)));
            for (int64_t i1 = 0; i1 < d.ne(1); ++i1) {
                const uint8_t * x = s.ptr(0, i1, i2, i3);
                uint8_t * y = d.ptr(0, i1, i2, i3);
                for (int64_t j = 0; j < hp; ++j) {
                    if (j < n_dims / 2) {
                        const int64_t pi = j;
                        float theta = position;
                        for (int k = 0; k < pi; ++k) {
                            theta *= theta_scale;
                        }
                        const float factor = ff != nullptr ? ld_f32(ff->ptr(pi)) : 1.0f;
                        const float theta_extrap = theta / factor;
                        const float theta_interp = p->freq_scale * theta_extrap;
                        float th = theta_interp;
                        float mscale = p->attn_factor;
                        if (p->ext_factor != 0.0f) {
                            const float ramp_mix = rope_ramp(corr0, corr1, static_cast<int>(2 * pi)) * p->ext_factor;
                            th = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                            mscale *= 1.0f + 0.1f * std::log(1.0f / p->freq_scale);
                        }
                        const float cs = std::cos(th) * mscale;
                        const float sn = std::sin(th) * mscale;
                        const int64_t e0 = neox ? n_offs + pi : n_offs + 2 * pi;
                        const int64_t e1 = neox ? n_offs + pi + n_dims / 2 : n_offs + 2 * pi + 1;
                        float x0 = 0.0f;
                        float x1 = 0.0f;
                        if (!load_float(s.dtype(), x + static_cast<uint64_t>(e0) * s.nb(0), &x0) ||
                            !load_float(s.dtype(), x + static_cast<uint64_t>(e1) * s.nb(0), &x1)) {
                            return ref_unsupported("ROPE dtype cannot be read");
                        }
                        if (!store_float(d.dtype(), y + static_cast<uint64_t>(e0) * d.nb(0), x0 * cs - x1 * sn) ||
                            !store_float(d.dtype(), y + static_cast<uint64_t>(e1) * d.nb(0), x0 * sn + x1 * cs)) {
                            return ref_unsupported("ROPE dtype cannot be written");
                        }
                    } else {
                        int64_t e = 2 * (j - n_dims / 2);
                        if (e >= n_offs) {
                            e += n_dims;
                        }
                        float v0 = 0.0f;
                        float v1 = 0.0f;
                        if (!load_float(s.dtype(), x + static_cast<uint64_t>(e) * s.nb(0), &v0) ||
                            !load_float(s.dtype(), x + static_cast<uint64_t>(e + 1) * s.nb(0), &v1)) {
                            return ref_unsupported("ROPE dtype cannot be read");
                        }
                        if (!store_float(d.dtype(), y + static_cast<uint64_t>(e) * d.nb(0), v0) ||
                            !store_float(d.dtype(), y + static_cast<uint64_t>(e + 1) * d.nb(0), v1)) {
                            return ref_unsupported("ROPE dtype cannot be written");
                        }
                    }
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_soft_max(const RTensor & s, const RTensor * mask, const WTensor & d, float scale) {
    const scrapgpu_status st = require_f32(s, "SOFT_MAX expects F32 tensors");
    if (st != SCRAPGPU_STATUS_OK) return st;
    const scrapgpu_status st2 = require_f32(d, "SOFT_MAX expects F32 tensors");
    if (st2 != SCRAPGPU_STATUS_OK) return st2;
    if (!same_ne(s, d)) {
        return ref_invalid("SOFT_MAX: src0 must have the shape of dst");
    }
    if (mask != nullptr) {
        if ((mask->dtype() != SCRAPGPU_DTYPE_F32 && mask->dtype() != SCRAPGPU_DTYPE_F16) || mask->ne(0) != s.ne(0) || mask->ne(1) < s.ne(1)) {
            return ref_invalid("SOFT_MAX mask must be F32/F16, as long as a row and cover the rows of src0");
        }
    }
    for (int64_t i3 = 0; i3 < s.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < s.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < s.ne(1); ++i1) {
                const uint8_t * m = mask != nullptr
                        ? mask->ptr(0, i1, i2 % mask->ne(2), i3 % mask->ne(3))
                        : nullptr;
                auto value = [&](int64_t i0) {
                    float v = ld_f32(s.ptr(i0, i1, i2, i3)) * scale;
                    if (m != nullptr) {
                        float mv = 0.0f;
                        if (!load_float(mask->dtype(), m + static_cast<uint64_t>(i0) * mask->nb(0), &mv)) {
                            return std::numeric_limits<float>::quiet_NaN();
                        }
                        v += mv;
                    }
                    return v;
                };
                float mx = -std::numeric_limits<float>::infinity();
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    mx = std::max(mx, value(i0));
                }
                float sum = 0.0f;
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    const float e = std::exp(value(i0) - mx);
                    st_f32(d.ptr(i0, i1, i2, i3), e);
                    sum += e;
                }
                const float inv = 1.0f / sum;
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    st_f32(d.ptr(i0, i1, i2, i3), ld_f32(d.ptr(i0, i1, i2, i3)) * inv);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// rows
// ---------------------------------------------------------------------------------------------

scrapgpu_status exec_get_rows(const RTensor & s0, const RTensor & ids, const WTensor & d) {
    if (ids.dtype() != SCRAPGPU_DTYPE_I32 || d.dtype() != SCRAPGPU_DTYPE_F32) {
        return ref_unsupported("GET_ROWS expects I32 indices and an F32 destination");
    }
    if (s0.dtype() != SCRAPGPU_DTYPE_F32 && s0.dtype() != SCRAPGPU_DTYPE_F16 && !dtype_is_quantized(s0.dtype())) {
        return ref_unsupported("GET_ROWS dtype is not implemented");
    }
    const scrapgpu_status st = require_contiguous_rows(s0, "GET_ROWS needs contiguous rows (nb0 == element size)");
    if (st != SCRAPGPU_STATUS_OK) return st;
    if (ids.ne(3) != 1 || ids.ne(1) > s0.ne(2) || ids.ne(2) > s0.ne(3) ||
        d.ne(0) != s0.ne(0) || d.ne(1) != ids.ne(0) || d.ne(2) != ids.ne(1) || d.ne(3) != ids.ne(2)) {
        return ref_invalid("GET_ROWS: dst must be [ne00, ids] over the slices of src0");
    }
    std::vector<float> row;
    for (int64_t i12 = 0; i12 < ids.ne(2); ++i12) {
        for (int64_t i11 = 0; i11 < ids.ne(1); ++i11) {
            for (int64_t i10 = 0; i10 < ids.ne(0); ++i10) {
                const int32_t idx = ld_i32(ids.ptr(i10, i11, i12));
                if (idx < 0 || static_cast<int64_t>(idx) >= s0.ne(1)) {
                    return ref_invalid("GET_ROWS: row index out of range");
                }
                const scrapgpu_status st2 = load_row(s0, s0.ptr(0, idx, i11, i12), s0.ne(0), &row);
                if (st2 != SCRAPGPU_STATUS_OK) {
                    return st2;
                }
                for (int64_t i0 = 0; i0 < s0.ne(0); ++i0) {
                    st_f32(d.ptr(i0, i10, i11, i12), row[static_cast<size_t>(i0)]);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_set_rows(const RTensor & s0, const RTensor & ids, const WTensor & d) {
    if ((s0.dtype() != SCRAPGPU_DTYPE_F32 && s0.dtype() != SCRAPGPU_DTYPE_F16) ||
        (d.dtype() != SCRAPGPU_DTYPE_F32 && d.dtype() != SCRAPGPU_DTYPE_F16)) {
        return ref_unsupported("SET_ROWS expects F32/F16 values and destination");
    }
    if (ids.dtype() != SCRAPGPU_DTYPE_I32 && ids.dtype() != SCRAPGPU_DTYPE_I64) {
        return ref_unsupported("SET_ROWS indices must be I32 or I64");
    }
    const scrapgpu_status st = require_contiguous_rows(s0, "SET_ROWS needs contiguous rows (nb0 == element size)");
    if (st != SCRAPGPU_STATUS_OK) return st;
    if (s0.ne(0) != d.ne(0) || s0.ne(2) != d.ne(2) || s0.ne(3) != d.ne(3) || d.ne(1) < 1) {
        return ref_invalid("SET_ROWS: dst must match src0 except along dim 1");
    }
    for (int64_t i3 = 0; i3 < s0.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < s0.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < s0.ne(1); ++i1) {
                const uint8_t * pi = ids.ptr(i1, i2 % ids.ne(1), i3 % ids.ne(2));
                const int64_t row = ids.dtype() == SCRAPGPU_DTYPE_I64 ? ld_i64(pi) : static_cast<int64_t>(ld_i32(pi));
                if (row < 0 || row >= d.ne(1)) {
                    return ref_invalid("SET_ROWS: row index out of range");
                }
                for (int64_t i0 = 0; i0 < s0.ne(0); ++i0) {
                    float v = 0.0f;
                    if (!load_float(s0.dtype(), s0.ptr(i0, i1, i2, i3), &v) ||
                        !store_float(d.dtype(), d.ptr(i0, row, i2, i3), v)) {
                        return ref_unsupported("SET_ROWS dtype cannot be converted");
                    }
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_cpy(const RTensor & s, const WTensor & d) {
    if ((s.dtype() != SCRAPGPU_DTYPE_F32 && s.dtype() != SCRAPGPU_DTYPE_F16) ||
        (d.dtype() != SCRAPGPU_DTYPE_F32 && d.dtype() != SCRAPGPU_DTYPE_F16)) {
        return ref_unsupported("CPY/DUP/CONT support F32 and F16 only");
    }
    const int64_t sne[4] = {s.ne(0), s.ne(1), s.ne(2), s.ne(3)};
    const int64_t dne[4] = {d.ne(0), d.ne(1), d.ne(2), d.ne(3)};
    int64_t ns = 0;
    int64_t nd = 0;
    if (!nelements_ok(sne, &ns) || !nelements_ok(dne, &nd) || ns != nd) {
        return ref_invalid("CPY/DUP/CONT: src0 and dst must have the same number of elements");
    }
    for (int64_t idx = 0; idx < ns; ++idx) {
        int64_t r = idx;
        const int64_t a0 = r % sne[0];
        r /= sne[0];
        const int64_t a1 = r % sne[1];
        r /= sne[1];
        const int64_t a2 = r % sne[2];
        const int64_t a3 = r / sne[2];

        r = idx;
        const int64_t b0 = r % dne[0];
        r /= dne[0];
        const int64_t b1 = r % dne[1];
        r /= dne[1];
        const int64_t b2 = r % dne[2];
        const int64_t b3 = r / dne[2];

        float v = 0.0f;
        if (!load_float(s.dtype(), s.ptr(a0, a1, a2, a3), &v) ||
            !store_float(d.dtype(), d.ptr(b0, b1, b2, b3), v)) {
            return ref_unsupported("CPY/DUP/CONT dtype cannot be converted");
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------------------------

scrapgpu_status exec_mul_mat(const RTensor & s0, const RTensor & s1, const WTensor & d, int allow_broadcast) {
    if (s1.dtype() != SCRAPGPU_DTYPE_F32 || d.dtype() != SCRAPGPU_DTYPE_F32) {
        return ref_unsupported("MUL_MAT expects an F32 src1 and destination");
    }
    if (s0.dtype() != SCRAPGPU_DTYPE_F32 && s0.dtype() != SCRAPGPU_DTYPE_F16 && !dtype_is_quantized(s0.dtype())) {
        return ref_unsupported("MUL_MAT dtype is not implemented");
    }
    const scrapgpu_status st = require_contiguous_rows(s0, "MUL_MAT needs contiguous rows of src0 (nb0 == element size)");
    if (st != SCRAPGPU_STATUS_OK) return st;
    if (s0.ne(0) != s1.ne(0)) {
        return ref_invalid("MUL_MAT: the inner dimensions of src0 and src1 differ");
    }
    if (s1.nb(0) != s1.esz() || d.nb(0) != d.esz()) {
        return ref_unsupported("MUL_MAT needs contiguous rows of src1 and dst (nb0 == element size)");
    }
    const int64_t r2 = s1.ne(2) / s0.ne(2);
    const int64_t r3 = s1.ne(3) / s0.ne(3);
    if (s1.ne(2) % s0.ne(2) != 0 || s1.ne(3) % s0.ne(3) != 0) {
        return ref_invalid("MUL_MAT: src1 dims 2 and 3 must be multiples of the src0 dims");
    }
    if (allow_broadcast == 0 && (r2 != 1 || r3 != 1)) {
        return ref_unsupported("MUL_MAT: broadcasting over dims 2/3 is not allowed");
    }
    if (d.ne(0) != s0.ne(1) || d.ne(1) != s1.ne(1) || d.ne(2) != s1.ne(2) || d.ne(3) != s1.ne(3)) {
        return ref_invalid("MUL_MAT: dst must be [ne01, ne11, ne12, ne13]");
    }
    std::vector<float> row;
    for (int64_t i13 = 0; i13 < s1.ne(3); ++i13) {
        const int64_t i03 = i13 / r3;
        for (int64_t i12 = 0; i12 < s1.ne(2); ++i12) {
            const int64_t i02 = i12 / r2;
            for (int64_t m = 0; m < s0.ne(1); ++m) {
                const scrapgpu_status st2 = load_row(s0, s0.ptr(0, m, i02, i03), s0.ne(0), &row);
                if (st2 != SCRAPGPU_STATUS_OK) {
                    return st2;
                }
                for (int64_t i11 = 0; i11 < s1.ne(1); ++i11) {
                    float acc = 0.0f;
                    for (int64_t k = 0; k < s1.ne(0); ++k) {
                        acc += row[static_cast<size_t>(k)] * ld_f32(s1.ptr(k, i11, i12, i13));
                    }
                    st_f32(d.ptr(m, i11, i12, i13), acc);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_mul_mat_id(const RTensor & s0, const RTensor & s1, const RTensor & ids, const WTensor & d) {
    if (s1.dtype() != SCRAPGPU_DTYPE_F32 || d.dtype() != SCRAPGPU_DTYPE_F32 || ids.dtype() != SCRAPGPU_DTYPE_I32) {
        return ref_unsupported("MUL_MAT_ID expects F32 src1/dst and I32 ids");
    }
    if (s0.dtype() != SCRAPGPU_DTYPE_F32 && s0.dtype() != SCRAPGPU_DTYPE_F16 && !dtype_is_quantized(s0.dtype())) {
        return ref_unsupported("MUL_MAT_ID dtype is not implemented");
    }
    const scrapgpu_status st = require_contiguous_rows(s0, "MUL_MAT_ID needs contiguous rows of src0 (nb0 == element size)");
    if (st != SCRAPGPU_STATUS_OK) return st;
    if (s0.ne(3) != 1 || s1.ne(3) != 1 || ids.ne(2) != 1 || ids.ne(3) != 1) {
        return ref_invalid("MUL_MAT_ID: the tensors must be 3-dimensional at most");
    }
    if (s0.ne(0) != s1.ne(0) || (s1.ne(1) != 1 && s1.ne(1) != ids.ne(0)) || s1.ne(2) != ids.ne(1) ||
        d.ne(0) != s0.ne(1) || d.ne(1) != ids.ne(0) || d.ne(2) != ids.ne(1) || d.ne(3) != 1) {
        return ref_invalid("MUL_MAT_ID: shapes do not match the contract");
    }
    std::vector<float> row;
    for (int64_t t = 0; t < ids.ne(1); ++t) {
        for (int64_t i = 0; i < ids.ne(0); ++i) {
            const int32_t e = ld_i32(ids.ptr(i, t));
            if (e < 0 || static_cast<int64_t>(e) >= s0.ne(2)) {
                return ref_invalid("MUL_MAT_ID: expert index out of range");
            }
            const int64_t i11 = s1.ne(1) == 1 ? 0 : i;
            for (int64_t m = 0; m < s0.ne(1); ++m) {
                const scrapgpu_status st2 = load_row(s0, s0.ptr(0, m, e), s0.ne(0), &row);
                if (st2 != SCRAPGPU_STATUS_OK) {
                    return st2;
                }
                float acc = 0.0f;
                for (int64_t k = 0; k < s1.ne(0); ++k) {
                    acc += row[static_cast<size_t>(k)] * ld_f32(s1.ptr(k, i11, t));
                }
                st_f32(d.ptr(m, i, t), acc);
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_add_id(const RTensor & s0, const RTensor & s1, const RTensor & ids, const WTensor & d) {
    const scrapgpu_status st = require_f32(s0, "ADD_ID expects F32 tensors");
    if (st != SCRAPGPU_STATUS_OK) return st;
    const scrapgpu_status st2 = require_f32(s1, "ADD_ID expects F32 tensors");
    if (st2 != SCRAPGPU_STATUS_OK) return st2;
    const scrapgpu_status st3 = require_f32(d, "ADD_ID expects F32 tensors");
    if (st3 != SCRAPGPU_STATUS_OK) return st3;
    if (ids.dtype() != SCRAPGPU_DTYPE_I32) {
        return ref_unsupported("ADD_ID ids must be I32");
    }
    if (s0.ne(3) != 1 || s1.ne(2) != 1 || s1.ne(3) != 1 || s0.ne(0) != s1.ne(0) ||
        ids.ne(0) != s0.ne(1) || ids.ne(1) != s0.ne(2) || ids.ne(2) != 1 || !same_ne(s0, d)) {
        return ref_invalid("ADD_ID: shapes do not match the contract");
    }
    for (int64_t i2 = 0; i2 < s0.ne(2); ++i2) {
        for (int64_t i1 = 0; i1 < s0.ne(1); ++i1) {
            int64_t e = ld_i32(ids.ptr(i1, i2));
            e = std::max<int64_t>(0, std::min<int64_t>(s1.ne(1) - 1, e));
            for (int64_t i0 = 0; i0 < s0.ne(0); ++i0) {
                st_f32(d.ptr(i0, i1, i2), ld_f32(s0.ptr(i0, i1, i2)) + ld_f32(s1.ptr(i0, e)));
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// misc
// ---------------------------------------------------------------------------------------------

scrapgpu_status exec_ssm_conv(const RTensor & s0, const RTensor & s1, const WTensor & d, const scrapgpu_ssm_conv_params * p) {
    const scrapgpu_status st = require_f32(s0, "SSM_CONV expects F32 tensors");
    if (st != SCRAPGPU_STATUS_OK) return st;
    const scrapgpu_status st2 = require_f32(s1, "SSM_CONV expects F32 tensors");
    if (st2 != SCRAPGPU_STATUS_OK) return st2;
    const scrapgpu_status st3 = require_f32(d, "SSM_CONV expects F32 tensors");
    if (st3 != SCRAPGPU_STATUS_OK) return st3;
    const int64_t d_conv = s1.ne(0);
    if (p->d_conv != 0 && static_cast<int64_t>(p->d_conv) != d_conv) {
        return ref_invalid("SSM_CONV: d_conv does not match the length of the filter");
    }
    if (s0.ne(3) != 1 || s1.ne(2) != 1 || s1.ne(3) != 1 || s0.ne(1) != s1.ne(1) || s0.ne(1) != d.ne(0) ||
        d.ne(1) != s0.ne(0) - d_conv + 1 || d.ne(2) != s0.ne(2) || d.ne(3) != 1) {
        return ref_invalid("SSM_CONV: shapes do not match the contract");
    }
    for (int64_t i2 = 0; i2 < d.ne(2); ++i2) {
        for (int64_t i1 = 0; i1 < d.ne(1); ++i1) {
            for (int64_t i0 = 0; i0 < d.ne(0); ++i0) {
                float sum = 0.0f;
                for (int64_t j = 0; j < d_conv; ++j) {
                    sum += ld_f32(s0.ptr(i1 + j, i0, i2)) * ld_f32(s1.ptr(j, i0));
                }
                st_f32(d.ptr(i0, i1, i2), sum);
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_concat(const RTensor & s0, const RTensor & s1, const WTensor & d, uint32_t dim) {
    if (dim >= 4) {
        return ref_invalid("CONCAT dim must be 0..3");
    }
    if (s0.dtype() != s1.dtype() || s0.dtype() != d.dtype()) {
        return ref_invalid("CONCAT: all tensors must have the same dtype");
    }
    const size_t esz = s0.esz();
    if (esz != 2 && esz != 4) {
        return ref_unsupported("CONCAT supports 2-byte and 4-byte element types only");
    }
    for (int i = 0; i < 4; ++i) {
        if (i == static_cast<int>(dim)) {
            if (d.ne(i) != s0.ne(i) + s1.ne(i)) {
                return ref_invalid("CONCAT: dst dim does not match the concatenation");
            }
        } else if (s0.ne(i) != s1.ne(i) || s0.ne(i) != d.ne(i)) {
            return ref_invalid("CONCAT: dims other than the concat dim must match");
        }
    }
    const int64_t o = s0.ne(static_cast<int>(dim));
    for (int64_t i3 = 0; i3 < d.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < d.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < d.ne(1); ++i1) {
                for (int64_t i0 = 0; i0 < d.ne(0); ++i0) {
                    const bool in0 = i0 < s0.ne(0) && i1 < s0.ne(1) && i2 < s0.ne(2) && i3 < s0.ne(3);
                    const uint8_t * src;
                    if (in0) {
                        src = s0.ptr(i0, i1, i2, i3);
                    } else {
                        const int64_t c0 = dim == 0 ? i0 - o : i0;
                        const int64_t c1 = dim == 1 ? i1 - o : i1;
                        const int64_t c2 = dim == 2 ? i2 - o : i2;
                        const int64_t c3 = dim == 3 ? i3 - o : i3;
                        src = s1.ptr(c0, c1, c2, c3);
                    }
                    uint8_t * dst = d.ptr(i0, i1, i2, i3);
                    std::memcpy(dst, src, esz);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_pad(const RTensor & s, const WTensor & d, const scrapgpu_pad_params * p) {
    const scrapgpu_status st = require_f32(s, "PAD expects F32 tensors");
    if (st != SCRAPGPU_STATUS_OK) return st;
    const scrapgpu_status st2 = require_f32(d, "PAD expects F32 tensors");
    if (st2 != SCRAPGPU_STATUS_OK) return st2;
    for (int i = 0; i < 4; ++i) {
        if (p->before[i] < 0 || p->after[i] < 0) {
            return ref_invalid("PAD: negative padding");
        }
        if (d.ne(i) != s.ne(i) + p->before[i] + p->after[i]) {
            return ref_invalid("PAD: dst dim does not match src plus the padding");
        }
    }
    for (int64_t i3 = 0; i3 < d.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < d.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < d.ne(1); ++i1) {
                for (int64_t i0 = 0; i0 < d.ne(0); ++i0) {
                    float v = 0.0f;
                    if (p->circular != 0) {
                        v = ld_f32(s.ptr(wrap_index(i0 - p->before[0], s.ne(0)),
                                         wrap_index(i1 - p->before[1], s.ne(1)),
                                         wrap_index(i2 - p->before[2], s.ne(2)),
                                         wrap_index(i3 - p->before[3], s.ne(3))));
                    } else if (i0 >= p->before[0] && i0 < d.ne(0) - p->after[0] &&
                               i1 >= p->before[1] && i1 < d.ne(1) - p->after[1] &&
                               i2 >= p->before[2] && i2 < d.ne(2) - p->after[2] &&
                               i3 >= p->before[3] && i3 < d.ne(3) - p->after[3]) {
                        v = ld_f32(s.ptr(i0 - p->before[0], i1 - p->before[1], i2 - p->before[2], i3 - p->before[3]));
                    }
                    st_f32(d.ptr(i0, i1, i2, i3), v);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_sum_rows(const RTensor & s, const WTensor & d) {
    const scrapgpu_status st = require_f32(s, "SUM_ROWS expects F32 tensors");
    if (st != SCRAPGPU_STATUS_OK) return st;
    const scrapgpu_status st2 = require_f32(d, "SUM_ROWS expects F32 tensors");
    if (st2 != SCRAPGPU_STATUS_OK) return st2;
    if (d.ne(0) != 1 || d.ne(1) != s.ne(1) || d.ne(2) != s.ne(2) || d.ne(3) != s.ne(3)) {
        return ref_invalid("SUM_ROWS: dst must be [1, ne1, ne2, ne3] of src0");
    }
    for (int64_t i3 = 0; i3 < s.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < s.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < s.ne(1); ++i1) {
                float sum = 0.0f;
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    sum += ld_f32(s.ptr(i0, i1, i2, i3));
                }
                st_f32(d.ptr(0, i1, i2, i3), sum);
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status exec_argsort(const RTensor & s, const WTensor & d, const scrapgpu_argsort_params * p, bool top_k) {
    if (s.dtype() != SCRAPGPU_DTYPE_F32 || d.dtype() != SCRAPGPU_DTYPE_I32) {
        return ref_unsupported("ARGSORT/TOP_K expect F32 keys and I32 indices");
    }
    if (d.ne(1) != s.ne(1) || d.ne(2) != s.ne(2) || d.ne(3) != s.ne(3)) {
        return ref_invalid("ARGSORT/TOP_K: dst must have the rows of src0");
    }
    const int64_t k = d.ne(0);
    if (k < 1 || k > s.ne(0)) {
        return ref_invalid("ARGSORT/TOP_K: dst ne0 must be within the row");
    }
    if (top_k) {
        if (p->top_k != 0 && static_cast<int64_t>(p->top_k) != k) {
            return ref_invalid("TOP_K: params top_k does not match dst ne0");
        }
    } else if (k != s.ne(0)) {
        return ref_invalid("ARGSORT: dst ne0 must match the row length");
    }
    const bool desc = top_k || p->order == SCRAPGPU_ARGSORT_DESC;
    std::vector<int32_t> idx(static_cast<size_t>(s.ne(0)));
    for (int64_t i3 = 0; i3 < s.ne(3); ++i3) {
        for (int64_t i2 = 0; i2 < s.ne(2); ++i2) {
            for (int64_t i1 = 0; i1 < s.ne(1); ++i1) {
                const uint8_t * row = s.ptr(0, i1, i2, i3);
                for (int64_t i0 = 0; i0 < s.ne(0); ++i0) {
                    idx[static_cast<size_t>(i0)] = static_cast<int32_t>(i0);
                }
                std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
                    const float ka = ld_f32(row + static_cast<uint64_t>(a) * s.nb(0));
                    const float kb = ld_f32(row + static_cast<uint64_t>(b) * s.nb(0));
                    return desc ? ka > kb : ka < kb;
                });
                for (int64_t i0 = 0; i0 < k; ++i0) {
                    st_i32(d.ptr(i0, i1, i2, i3), idx[static_cast<size_t>(i0)]);
                }
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

template <typename T>
const T * params_as(const void * params, size_t params_size) {
    if (params == nullptr || params_size < sizeof(T)) {
        return nullptr;
    }
    return static_cast<const T *>(params);
}

} // namespace

scrapgpu_status ref_invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

scrapgpu_status ref_unsupported(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

} // namespace ref
} // namespace scrapgpu

extern "C" {

scrapgpu_status scrapgpu_ref_execute(const scrapgpu_op_desc * desc, void * const * input_data, void * const * output_data) {
    using namespace scrapgpu::ref;

    if (desc == nullptr) {
        return ref_invalid("op desc is null");
    }
    const scrapgpu_status validated = scrapgpu_op_desc_validate(desc);
    if (validated != SCRAPGPU_STATUS_OK) {
        return validated;
    }
    if (input_data == nullptr || output_data == nullptr) {
        return ref_invalid("input_data and output_data are required");
    }
    for (size_t i = 0; i < desc->input_count; ++i) {
        RTensor t;
        const scrapgpu_status st = make_r(&desc->inputs[i], input_data[i], &t, "input data pointer is null");
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        WTensor t;
        const scrapgpu_status st = make_w(&desc->outputs[i], output_data[i], &t, "output data pointer is null");
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }

    auto input = [&](size_t i) {
        RTensor t;
        make_r(&desc->inputs[i], input_data[i], &t, "input data pointer is null");
        return t;
    };
    auto output = [&](size_t i) {
        WTensor t;
        make_w(&desc->outputs[i], output_data[i], &t, "output data pointer is null");
        return t;
    };

    switch (desc->op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_SUB:
        case SCRAPGPU_OP_MUL:
        case SCRAPGPU_OP_DIV:
            return exec_binary(desc->op, input(0), input(1), output(0));
        case SCRAPGPU_OP_SCALE:
        case SCRAPGPU_OP_CLAMP:
        case SCRAPGPU_OP_SQR:
        case SCRAPGPU_OP_SQRT:
        case SCRAPGPU_OP_UNARY:
            return exec_unary(desc->op, input(0), output(0), desc->params, desc->params_size);
        case SCRAPGPU_OP_GLU: {
            const auto * p = params_as<scrapgpu_glu_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("GLU params are missing");
            }
            return exec_glu(input(0), desc->input_count > 1 ? input(1) : RTensor(), desc->input_count > 1, output(0), p);
        }
        case SCRAPGPU_OP_RMS_NORM: {
            float eps = 0.0f;
            const scrapgpu_status st = read_eps(desc->params, desc->params_size, &eps);
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
            return exec_rms_norm(input(0), output(0), eps);
        }
        case SCRAPGPU_OP_RMS_NORM_MUL: {
            if (desc->input_count != 2 || desc->output_count != 1) {
                return ref_invalid("RMS_NORM_MUL takes 2 inputs and 1 output");
            }
            float eps = 0.0f;
            const scrapgpu_status st = read_eps(desc->params, desc->params_size, &eps);
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
            return exec_rms_norm_mul(input(0), input(1), output(0), eps);
        }
        case SCRAPGPU_OP_ADD_RMS_NORM_MUL: {
            if (desc->input_count != 3 || desc->output_count != 2) {
                return ref_invalid("ADD_RMS_NORM_MUL takes 3 inputs and 2 outputs");
            }
            float eps = 0.0f;
            const scrapgpu_status st = read_eps(desc->params, desc->params_size, &eps);
            if (st != SCRAPGPU_STATUS_OK) {
                return st;
            }
            return exec_add_rms_norm_mul(input(0), input(1), input(2), output(0), output(1), eps);
        }
        case SCRAPGPU_OP_ROPE: {
            const auto * p = params_as<scrapgpu_rope_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("ROPE params are missing");
            }
            if (desc->input_count < 2) {
                return ref_invalid("ROPE needs the positions as the second input");
            }
            RTensor ff;
            RTensor * ff_ptr = nullptr;
            if (desc->input_count > 2) {
                ff = input(2);
                ff_ptr = &ff;
            }
            return exec_rope(input(0), input(1), ff_ptr, output(0), p);
        }
        case SCRAPGPU_OP_SOFT_MAX: {
            const auto * p = params_as<scrapgpu_soft_max_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("SOFT_MAX params are missing");
            }
            RTensor mask;
            RTensor * mask_ptr = nullptr;
            if (desc->input_count > 1) {
                mask = input(1);
                mask_ptr = &mask;
            }
            return exec_soft_max(input(0), mask_ptr, output(0), p->scale);
        }
        case SCRAPGPU_OP_GET_ROWS:
            return exec_get_rows(input(0), input(1), output(0));
        case SCRAPGPU_OP_SET_ROWS:
            return exec_set_rows(input(0), input(1), output(0));
        case SCRAPGPU_OP_CPY:
        case SCRAPGPU_OP_DUP:
        case SCRAPGPU_OP_CONT:
            return exec_cpy(input(0), output(0));
        case SCRAPGPU_OP_MUL_MAT: {
            const auto * p = params_as<scrapgpu_mul_mat_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("MUL_MAT params are missing");
            }
            return exec_mul_mat(input(0), input(1), output(0), p->allow_broadcast);
        }
        case SCRAPGPU_OP_MUL_MAT_ID:
            if (desc->input_count != 3) {
                return ref_invalid("MUL_MAT_ID takes 3 inputs");
            }
            return exec_mul_mat_id(input(0), input(1), input(2), output(0));
        case SCRAPGPU_OP_ADD_ID:
            if (desc->input_count != 3) {
                return ref_invalid("ADD_ID takes 3 inputs");
            }
            return exec_add_id(input(0), input(1), input(2), output(0));
        case SCRAPGPU_OP_SSM_CONV: {
            const auto * p = params_as<scrapgpu_ssm_conv_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("SSM_CONV params are missing");
            }
            return exec_ssm_conv(input(0), input(1), output(0), p);
        }
        case SCRAPGPU_OP_CONCAT: {
            const auto * p = params_as<scrapgpu_concat_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("CONCAT params are missing");
            }
            return exec_concat(input(0), input(1), output(0), p->dim);
        }
        case SCRAPGPU_OP_PAD: {
            const auto * p = params_as<scrapgpu_pad_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("PAD params are missing");
            }
            return exec_pad(input(0), output(0), p);
        }
        case SCRAPGPU_OP_SUM_ROWS:
            return exec_sum_rows(input(0), output(0));
        case SCRAPGPU_OP_ARGSORT:
        case SCRAPGPU_OP_TOP_K: {
            const auto * p = params_as<scrapgpu_argsort_params>(desc->params, desc->params_size);
            if (p == nullptr) {
                return ref_invalid("ARGSORT/TOP_K params are missing");
            }
            return exec_argsort(input(0), output(0), p, desc->op == SCRAPGPU_OP_TOP_K);
        }
        default:
            return ref_unsupported("the reference does not implement this op");
    }
}

} // extern "C"
