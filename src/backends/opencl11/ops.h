#ifndef SCRAPGPU_OPENCL11_OPS_H
#define SCRAPGPU_OPENCL11_OPS_H

#include "device_state.h"
#include "tensor.h"

#include "scrapgpu/op.h"

#include <cstdint>

// One implementation of every op, callable without ggml: each function takes the device, the input /
// output tensor views (cl11_tensor, built on the stack by the caller) and a small struct of plain
// values. ggml never reaches the backend: the callers decode op_params and map their types / ops /
// modes to the scrapgpu enums or the kernel argument codes below.
//
// The caller holds dev->mutex and has already checked the op with the support predicates below
// (shape / type / size conditions) plus its own limits (descriptor fields must fit 32 bits).

// ---- kernel argument codes (part of the kernel ABI of elementwise.cl) ----

// the `op` argument of k_unary (UNARY, SQR, SQRT, SCALE and CLAMP share the one kernel)
enum cl11_unary_code {
    CL11_UC_SILU    = 0,
    CL11_UC_RELU    = 1,
    CL11_UC_NEG     = 2,
    CL11_UC_ABS     = 3,
    CL11_UC_SIGMOID = 4,
    CL11_UC_TANH    = 5,
    CL11_UC_SQR     = 6,
    CL11_UC_SQRT    = 7,
    CL11_UC_SCALE   = 8, // p0 = scale, p1 = bias
    CL11_UC_CLAMP   = 9, // p0 = min, p1 = max
};

// the binary elementwise ops (k_add / k_sub / k_mul / k_div)
enum cl11_binary_kind {
    CL11_BINARY_ADD,
    CL11_BINARY_SUB,
    CL11_BINARY_MUL,
    CL11_BINARY_DIV,
};

// ---- params decoded by the caller (the public scrapgpu_* structs are used where they fit) ----

// ROPE: mode and the yarn values of ggml's op_params; corr_dims are the yarn corr dims of
// (n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow), computed by the caller.
struct cl11_rope_params {
    scrapgpu_rope_mode mode;
    int32_t            n_dims;
    int32_t            n_ctx_orig;
    int32_t            n_offs;
    float              freq_base;
    float              freq_scale;
    float              ext_factor;
    float              attn_factor;
    float              beta_fast;
    float              beta_slow;
    float              corr_dims[2];
};

// SOFT_MAX: ALiBi (max_bias != 0) and attention sinks are not supported
struct cl11_soft_max_params {
    float scale;
    float max_bias;
};

// ---- dtype helpers ----

inline bool cl11_is_float(scrapgpu_dtype t) {
    return t == SCRAPGPU_DTYPE_F32 || t == SCRAPGPU_DTYPE_F16;
}

// the kernel variant index of a dtype (-1: the dtype has no kernels)
inline int cl11_qt_of(scrapgpu_dtype t) {
    switch (t) {
        case SCRAPGPU_DTYPE_F32:  return CL11_QT_F32;
        case SCRAPGPU_DTYPE_F16:  return CL11_QT_F16;
        case SCRAPGPU_DTYPE_Q4_0: return CL11_QT_Q4_0;
        case SCRAPGPU_DTYPE_Q8_0: return CL11_QT_Q8_0;
        case SCRAPGPU_DTYPE_Q4_K: return CL11_QT_Q4_K;
        case SCRAPGPU_DTYPE_Q6_K: return CL11_QT_Q6_K;
        default:                  return -1;
    }
}

// the repacked-kind index of a repackable dtype (the result for the other dtypes is meaningless)
inline int cl11_rk_of(scrapgpu_dtype t) {
    switch (t) {
        case SCRAPGPU_DTYPE_Q4_0: return CL11_RK_Q4_0;
        case SCRAPGPU_DTYPE_Q8_0: return CL11_RK_Q8_0;
        case SCRAPGPU_DTYPE_Q4_K: return CL11_RK_Q4_K;
        default:                  return CL11_RK_Q6_K;
    }
}

inline int cl11_log2_exact(int v) {
    int l = 0;
    while ((1 << l) < v) {
        ++l;
    }
    return l;
}

// ARGSORT / TOP_K: a row is sorted in local memory (2 x 4 bytes per element, padded to a power of 2)
inline int64_t cl11_sort_npad(int64_t ncols) {
    int64_t n = 1;
    while (n < ncols) {
        n <<= 1;
    }
    return n;
}

inline bool cl11_sort_fits(const cl11_device * c, int64_t ncols) {
    return ncols >= 1 && cl11_sort_npad(ncols) <= cl11_sort_nmax(c->info.local_mem);
}

// conditions of the attention kernels of attn.cl (K cache: k_mvq_f16, transposed V cache: k_mvt_f16):
// the heads of a position are contiguous in the cache (row = ne02 heads of `hd` halves), 16-byte
// aligned rows, a work-item per 16 bytes of a row, at most 8 query heads per head of the cache,
// src1 (q / x) with 16-byte aligned float4 loads
bool cl11_attn_ok(const cl11_device * c, const cl11_td & ta, const cl11_td & tb, int64_t hd, int64_t ne02, int64_t r2, int64_t k_len, bool v);

// a write to the range [offset, offset + bytes) of `mem` makes the prepared src1 columns (k_xprep
// scratch) and the cached MUL_MAT_ID ids made from it stale
void cl11_xprep_note_write(cl11_device * c, cl_mem mem, size_t offset, size_t bytes);

// ---- the ops ----

// the elementwise and row ops (ops.cpp)

bool cl11_op_binary(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, cl11_binary_kind kind);
// UNARY / SCALE / CLAMP / SQR / SQRT: `code` is a cl11_unary_code, p0 / p1 the two float params
bool cl11_op_unary(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, int32_t code, float p0, float p1);
// GLU: s1 is the second (split) input, nullptr when the gate halves are in s0; params: kind, swapped
bool cl11_op_glu(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor * s1, const scrapgpu_glu_params & p);
bool cl11_op_rms_norm(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const scrapgpu_rms_norm_params & p);
// RMS_NORM + MUL in one kernel: d = rms_norm(s0, eps) * w (the fused form of two ops)
bool cl11_op_rms_norm_mul(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & w, const scrapgpu_rms_norm_params & p);
// ADD + RMS_NORM + MUL in one kernel: add = a + b (one row), dst = rms_norm(add, eps) * w
bool cl11_op_add_rms_norm_mul(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & a, const cl11_tensor & b,
                              const cl11_tensor & w, const cl11_tensor & add, const scrapgpu_rms_norm_params & p);
// SOFT_MAX: mask is the optional second input (F32 / F16), nullptr without one
bool cl11_op_soft_max(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor * mask, float scale);
// ROPE: pos is the positions input (I32), ff the optional freq factors (F32), nullptr without them
bool cl11_op_rope(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & pos, const cl11_tensor * ff,
                  const cl11_rope_params & p);
bool cl11_op_get_rows(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & ids);
bool cl11_op_set_rows(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & ids);
// two SET_ROWS with the same types in one launch (k_set_rows2_*): a / b are { src0, ids, dst }
bool cl11_op_set_rows2(cl11_device * c, const cl11_tensor a[3], const cl11_tensor b[3]);
// CPY / DUP / CONT: same-size copy, with the transposing kernel when src0 is a transposed view
bool cl11_op_cpy(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0);
bool cl11_op_add_id(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, const cl11_tensor & ids);
bool cl11_op_ssm_conv(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1);
bool cl11_op_concat(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, const scrapgpu_concat_params & p);
bool cl11_op_pad(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const scrapgpu_pad_params & p);
bool cl11_op_sum_rows(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0);
// ARGSORT / TOP_K: params.order, top_k = whether this is a TOP_K op
bool cl11_op_argsort(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const scrapgpu_argsort_params & p, bool top_k);

// MUL_MAT and the fused mat-vec forms (ops_matmul.cpp)

bool cl11_op_mul_mat(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1);
// the kernel selection of a MUL_MAT (mat-vec / attention / staged / mat-mat); `moe`: the operands are
// slices / scratch columns of MUL_MAT_ID (no K cache kernel, no constant-memory / prepared src1)
bool cl11_mul_mat_core(cl11_device * c, const cl11_tensor & s0, const cl11_tensor & s1, const cl11_tensor & d, bool moe);
// MUL_MAT(CONT(transposed F16 view) `a`, x) with a single column x: the CONT is skipped, k_mvt_f16
bool cl11_op_mul_mat_t(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & a, const cl11_tensor & s1);
// the MUL_MATs of n repacked weight matrices with the same single-column src1 in one launch (k_mvm_*):
// 1 = computed, 0 = not applicable (compute the nodes one by one), -1 = error. The caller guarantees
// 2 <= n <= 3 and fills the first n entries of dsts / srcs0.
int cl11_op_mul_mat_multi(cl11_device * c, const cl11_tensor dsts[3], const cl11_tensor srcs0[3], const cl11_tensor & s1, int n);

// MUL_MAT_ID (ops_moe.cpp): dst[:, i, t] = s0[:, :, ids[i, t]] * s1[:, i % ne11, t]. Contains the
// host readback of the ids on the prompt path (a synchronization point).
bool cl11_op_mul_mat_id(cl11_device * c, const cl11_tensor & dst, const cl11_tensor & s0, const cl11_tensor & s1, const cl11_tensor & ids);

// ---- support predicates (ops_support.cpp) ----
//
// The shape / type / size conditions of the ops on the descriptors; the repacked flags say whether a
// repackable weight tensor is stored in the R32 layout. What they do not check (the caller's part):
// the 32-bit limits of the descriptor fields, the driver allocation limit, and every condition on
// things a descriptor does not carry (a view of a repacked tensor, the presence of extra inputs).

bool cl11_supports_binary(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc & d);
// `code` a cl11_unary_code: the UNARY kinds are CL11_UC_SILU .. CL11_UC_TANH
bool cl11_supports_unary(const sgt::TensorDesc & s0, const sgt::TensorDesc & d, int32_t code);
bool cl11_supports_glu(const sgt::TensorDesc & s0, const sgt::TensorDesc * s1, const sgt::TensorDesc & d, const scrapgpu_glu_params & p);
bool cl11_supports_rms_norm(const sgt::TensorDesc & s0, const sgt::TensorDesc & d);
bool cl11_supports_soft_max(const sgt::TensorDesc & s0, const sgt::TensorDesc * mask, const sgt::TensorDesc * sinks,
                            const sgt::TensorDesc & d, const cl11_soft_max_params & p);
bool cl11_supports_rope(const sgt::TensorDesc & s0, const sgt::TensorDesc * pos, const sgt::TensorDesc * ff,
                        const sgt::TensorDesc & d, const cl11_rope_params & p);
bool cl11_supports_get_rows(const sgt::TensorDesc & s0, bool s0_repacked, const sgt::TensorDesc & ids, const sgt::TensorDesc & d);
bool cl11_supports_set_rows(const sgt::TensorDesc & s0, const sgt::TensorDesc & ids, const sgt::TensorDesc & d);
bool cl11_supports_cpy(const sgt::TensorDesc & s0, const sgt::TensorDesc & d);
bool cl11_supports_mul_mat(const sgt::TensorDesc & s0, bool s0_repacked, const sgt::TensorDesc & s1, const sgt::TensorDesc & d);
bool cl11_supports_add_id(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc * ids, const sgt::TensorDesc & d);
bool cl11_supports_ssm_conv(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc & d);
bool cl11_supports_concat(const sgt::TensorDesc & s0, const sgt::TensorDesc & s1, const sgt::TensorDesc & d, const scrapgpu_concat_params & p);
bool cl11_supports_pad(const sgt::TensorDesc & s0, const sgt::TensorDesc & d);
bool cl11_supports_sum_rows(const sgt::TensorDesc & s0, const sgt::TensorDesc & d);
bool cl11_supports_argsort(const cl11_device * c, const sgt::TensorDesc & s0, const sgt::TensorDesc & d);
bool cl11_supports_mul_mat_id(const cl11_device * c, const sgt::TensorDesc & s0, bool s0_repacked,
                              const sgt::TensorDesc & s1, const sgt::TensorDesc * ids, const sgt::TensorDesc & d);

#endif
