#ifndef SCRAPGPU_OPENCL11_KERNELS_H
#define SCRAPGPU_OPENCL11_KERNELS_H

#include "context.h"

#include <string>

enum cl11_kernel_class {
    CL11_CLASS_EW,
    CL11_CLASS_ROW,
    CL11_CLASS_MV,
    CL11_CLASS_MM,
    CL11_CLASS_MVR,
    CL11_CLASS_ATT,
    CL11_CLASS_MVM,
    CL11_CLASS_MMX,
    CL11_CLASS_COUNT,
};

enum cl11_qtype {
    CL11_QT_F32,
    CL11_QT_F16,
    CL11_QT_Q4_0,
    CL11_QT_Q8_0,
    CL11_QT_Q4_K,
    CL11_QT_Q6_K,
    CL11_QT_COUNT,
};

enum cl11_kernel_id {
    K_ADD, K_SUB, K_MUL, K_DIV,
    K_UNARY,
    K_GLU,
    K_CPY_F32_F32, K_CPY_F32_F16, K_CPY_F16_F32, K_CPY_F16_F16,
    K_CPY_T_F32, K_CPY_T_F16,
    K_FILL_U32, K_FILL_U8,
    K_RMS_NORM,
    K_RMS_NORM_MUL,
    K_ADD_RMS_NORM_MUL,
    K_SOFT_MAX,
    K_ROPE_F32, K_ROPE_F16,
    K_GET_ROWS,
    K_SET_ROWS = K_GET_ROWS + CL11_QT_COUNT,
    K_MV       = K_SET_ROWS + 4,
    K_MM       = K_MV + CL11_QT_COUNT,
    K_MVX      = K_MM + CL11_QT_COUNT,
    K_MVW      = K_MVX + CL11_QT_COUNT,
    K_MVT_F16  = K_MVW + CL11_QT_COUNT,
    K_MVQ_F16,
    K_MVT_G_F16,
    K_MVR,
    K_MVRC     = K_MVR + CL11_RK_COUNT,
    K_MMX      = K_MVRC + CL11_RK_COUNT,
    K_MVM      = K_MMX + CL11_QT_COUNT,
    K_SET_ROWS2 = K_MVM + CL11_RK_COUNT,
    K_XPREP    = K_SET_ROWS2 + 4,
    K_ADD_ID,
    K_SSM_CONV,
    K_CONCAT4, K_CONCAT2,
    K_PAD,
    K_SUM_ROWS,
    K_ARGSORT,
    K_MID_GATHER, K_MID_SCATTER,
    K_MVI,
    K_MVRI     = K_MVI + CL11_QT_COUNT,
    K_COUNT    = K_MVRI + CL11_RK_COUNT,
};

struct cl11_kernel_desc {
    const char * name;
    cl11_kernel_class cls;
};

extern const cl11_kernel_desc cl11_kernel_table[K_COUNT];

bool cl11_is_k_kernel(int id);
bool cl11_is_mmx_kernel(int id);
bool cl11_is_mm_kernel(int id);
int64_t cl11_sort_nmax(size_t local_mem);

bool cl11_build_program(cl_context context,
                        cl_device_id device,
                        const char * logical_name,
                        const char * device_name,
                        size_t local_mem,
                        bool use_nv_asm,
                        bool use_nv_ldg,
                        bool multi_mv,
                        const cl11_config & cfg,
                        std::string & log,
                        int mode,
                        cl_program & prog);

#endif
