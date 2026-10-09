#ifndef SCRAPGPU_OP_H
#define SCRAPGPU_OP_H

#include "scrapgpu/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum scrapgpu_op {
    SCRAPGPU_OP_NONE = 0,
    SCRAPGPU_OP_RESHAPE = 1,
    SCRAPGPU_OP_VIEW = 2,
    SCRAPGPU_OP_PERMUTE = 3,
    SCRAPGPU_OP_TRANSPOSE = 4,
    SCRAPGPU_OP_ADD = 10,
    SCRAPGPU_OP_SUB = 11,
    SCRAPGPU_OP_MUL = 12,
    SCRAPGPU_OP_DIV = 13,
    SCRAPGPU_OP_SCALE = 14,
    SCRAPGPU_OP_CLAMP = 15,
    SCRAPGPU_OP_SQR = 16,
    SCRAPGPU_OP_SQRT = 17,
    SCRAPGPU_OP_UNARY = 18,
    SCRAPGPU_OP_GLU = 19,
    SCRAPGPU_OP_RMS_NORM = 20,
    SCRAPGPU_OP_ROPE = 21,
    SCRAPGPU_OP_SOFT_MAX = 22,
    SCRAPGPU_OP_GET_ROWS = 23,
    SCRAPGPU_OP_SET_ROWS = 24,
    SCRAPGPU_OP_CPY = 25,
    SCRAPGPU_OP_DUP = 26,
    SCRAPGPU_OP_CONT = 27,
    SCRAPGPU_OP_MUL_MAT = 28,
    SCRAPGPU_OP_MUL_MAT_ID = 29,
    SCRAPGPU_OP_ADD_ID = 30,
    SCRAPGPU_OP_SSM_CONV = 31,
    SCRAPGPU_OP_CONCAT = 32,
    SCRAPGPU_OP_PAD = 33,
    SCRAPGPU_OP_SUM_ROWS = 34,
    SCRAPGPU_OP_ARGSORT = 35,
    SCRAPGPU_OP_TOP_K = 36,
    SCRAPGPU_OP_RMS_NORM_MUL = 100,
    SCRAPGPU_OP_ADD_RMS_NORM_MUL = 101,
    SCRAPGPU_OP_SET_ROWS_PAIR = 102,
    SCRAPGPU_OP_MUL_MAT_MULTI = 103,
    SCRAPGPU_OP_CONT_MUL_MAT = 104,
} scrapgpu_op;

typedef enum scrapgpu_unary_kind {
    SCRAPGPU_UNARY_SILU = 0,
    SCRAPGPU_UNARY_RELU = 1,
    SCRAPGPU_UNARY_NEG = 2,
    SCRAPGPU_UNARY_ABS = 3,
    SCRAPGPU_UNARY_SIGMOID = 4,
    SCRAPGPU_UNARY_TANH = 5,
} scrapgpu_unary_kind;

typedef enum scrapgpu_glu_kind {
    SCRAPGPU_GLU_SWIGLU = 0,
    SCRAPGPU_GLU_REGLU = 1,
} scrapgpu_glu_kind;

typedef enum scrapgpu_argsort_order {
    SCRAPGPU_ARGSORT_ASC = 0,
    SCRAPGPU_ARGSORT_DESC = 1,
} scrapgpu_argsort_order;

typedef enum scrapgpu_rope_mode {
    SCRAPGPU_ROPE_MODE_NORMAL = 0,
    SCRAPGPU_ROPE_MODE_NEOX = 1,
} scrapgpu_rope_mode;

typedef struct scrapgpu_rope_params {
    size_t struct_size;
    scrapgpu_rope_mode mode;
    int32_t n_dims;
    int32_t n_ctx_orig;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    /* Optional tensors are also passed as desc->inputs[2] (F32) and later slots; when non-NULL these must
     * describe the same tensor as the matching inputs[] entry (reference reads inputs[] only). */
    const scrapgpu_tensor_desc * freq_factors;
    const scrapgpu_tensor_desc * n_offs;
} scrapgpu_rope_params;

typedef struct scrapgpu_soft_max_params {
    size_t struct_size;
    float scale;
    /* Optional mask: also desc->inputs[1] (F32/F16); when non-NULL must match that entry (reference uses inputs[]). */
    const scrapgpu_tensor_desc * mask;
} scrapgpu_soft_max_params;

typedef struct scrapgpu_glu_params {
    size_t struct_size;
    scrapgpu_glu_kind kind;
    int split_inputs;
    int swapped;
} scrapgpu_glu_params;

typedef struct scrapgpu_scale_params {
    size_t struct_size;
    float scale;
    float bias;
} scrapgpu_scale_params;

typedef struct scrapgpu_clamp_params {
    size_t struct_size;
    float min_value;
    float max_value;
} scrapgpu_clamp_params;

typedef struct scrapgpu_unary_params {
    size_t struct_size;
    scrapgpu_unary_kind kind;
} scrapgpu_unary_params;

typedef struct scrapgpu_argsort_params {
    size_t struct_size;
    scrapgpu_argsort_order order;
    uint32_t top_k;
} scrapgpu_argsort_params;

typedef struct scrapgpu_pad_params {
    size_t struct_size;
    int64_t before[4];
    int64_t after[4];
    int circular;
} scrapgpu_pad_params;

typedef struct scrapgpu_concat_params {
    size_t struct_size;
    uint32_t dim;
} scrapgpu_concat_params;

typedef struct scrapgpu_ssm_conv_params {
    size_t struct_size;
    uint32_t d_conv;
} scrapgpu_ssm_conv_params;

typedef struct scrapgpu_get_rows_params {
    size_t struct_size;
    int indices_are_i64;
} scrapgpu_get_rows_params;

typedef struct scrapgpu_set_rows_params {
    size_t struct_size;
    int indices_are_i64;
} scrapgpu_set_rows_params;

typedef struct scrapgpu_cpy_params {
    size_t struct_size;
    int allow_transpose;
} scrapgpu_cpy_params;

typedef struct scrapgpu_rms_norm_params {
    size_t struct_size;
    float eps;
} scrapgpu_rms_norm_params;

typedef struct scrapgpu_mul_mat_params {
    size_t struct_size;
    int allow_broadcast;
} scrapgpu_mul_mat_params;

typedef struct scrapgpu_mul_mat_id_params {
    size_t struct_size;
    uint32_t n_used;
    uint32_t max_prompt_tokens;
} scrapgpu_mul_mat_id_params;

typedef struct scrapgpu_op_desc {
    size_t struct_size;
    scrapgpu_op op;
    const scrapgpu_tensor_desc * inputs;
    size_t input_count;
    scrapgpu_tensor_desc * outputs;
    size_t output_count;
    const void * params;
    size_t params_size;
} scrapgpu_op_desc;

SCRAPGPU_API scrapgpu_status scrapgpu_op_desc_validate(const scrapgpu_op_desc * desc);
SCRAPGPU_API scrapgpu_status scrapgpu_op_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason);

#ifdef __cplusplus
}
#endif

#endif
