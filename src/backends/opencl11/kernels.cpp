#include "kernels.h"
#include "cl11-kernels.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#define CL11_NUM_SOURCES (sizeof(ggml_cl11_sources) / sizeof(ggml_cl11_sources[0]))

// must be in the order of enum cl11_kernel_id
const cl11_kernel_desc cl11_kernel_table[K_COUNT] = {
    { "k_add",              CL11_CLASS_EW  },
    { "k_sub",              CL11_CLASS_EW  },
    { "k_mul",              CL11_CLASS_EW  },
    { "k_div",              CL11_CLASS_EW  },
    { "k_unary",            CL11_CLASS_EW  },
    { "k_glu",              CL11_CLASS_EW  },
    { "k_cpy_f32_f32",      CL11_CLASS_EW  },
    { "k_cpy_f32_f16",      CL11_CLASS_EW  },
    { "k_cpy_f16_f32",      CL11_CLASS_EW  },
    { "k_cpy_f16_f16",      CL11_CLASS_EW  },
    { "k_cpy_t_f32_f32",    CL11_CLASS_MM  },
    { "k_cpy_t_f16_f16",    CL11_CLASS_MM  },
    { "k_fill_u32",         CL11_CLASS_EW  },
    { "k_fill_u8",          CL11_CLASS_EW  },
    { "k_rms_norm",         CL11_CLASS_ROW },
    { "k_rms_norm_mul",     CL11_CLASS_ROW },
    { "k_add_rms_norm_mul", CL11_CLASS_ROW },
    { "k_soft_max",         CL11_CLASS_ROW },
    { "k_rope_f32",         CL11_CLASS_ROW },
    { "k_rope_f16",         CL11_CLASS_ROW },
    // K_GET_ROWS + cl11_qtype
    { "k_get_rows_f32",     CL11_CLASS_ROW },
    { "k_get_rows_f16",     CL11_CLASS_ROW },
    { "k_get_rows_q4_0",    CL11_CLASS_ROW },
    { "k_get_rows_q8_0",    CL11_CLASS_ROW },
    { "k_get_rows_q4_k",    CL11_CLASS_ROW },
    { "k_get_rows_q6_k",    CL11_CLASS_ROW },
    // K_SET_ROWS + 2*(src is F16) + (dst is F16)
    { "k_set_rows_f32_f32", CL11_CLASS_EW  },
    { "k_set_rows_f32_f16", CL11_CLASS_EW  },
    { "k_set_rows_f16_f32", CL11_CLASS_EW  },
    { "k_set_rows_f16_f16", CL11_CLASS_EW  },
    // K_MV + cl11_qtype
    { "k_mv_f32",           CL11_CLASS_MV  },
    { "k_mv_f16",           CL11_CLASS_MV  },
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    // K_MM + cl11_qtype
    { "k_mm_f32",           CL11_CLASS_MM  },
    { "k_mm_f16",           CL11_CLASS_MM  },
    { "k_mm_q4_0",          CL11_CLASS_MM  },
    { "k_mm_q8_0",          CL11_CLASS_MM  },
    { "k_mm_q4_k",          CL11_CLASS_MM  },
    { "k_mm_q6_k",          CL11_CLASS_MM  },
    // K_MVX + cl11_qtype
    { "k_mvx_f32",          CL11_CLASS_MV  },
    { "k_mvx_f16",          CL11_CLASS_MV  },
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    // K_MVW + cl11_qtype
    { "k_mvw_f32",          CL11_CLASS_MV  },
    { "k_mvw_f16",          CL11_CLASS_MV  },
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvr_*
    { "k_mvt_f16",          CL11_CLASS_ATT },
    { "k_mvq_f16",          CL11_CLASS_ROW },
    { "k_mvt_g_f16",        CL11_CLASS_ROW },
    // K_MVR + cl11_rkind
    { "k_mvr_q4_0",         CL11_CLASS_MVR },
    { "k_mvr_q8_0",         CL11_CLASS_MVR },
    { "k_mvr_q4_k",         CL11_CLASS_MVR },
    { "k_mvr_q6_k",         CL11_CLASS_MVR },
    // K_MVRC + cl11_rkind
    { "k_mvrc_q4_0",        CL11_CLASS_MVR },
    { "k_mvrc_q8_0",        CL11_CLASS_MVR },
    { "k_mvrc_q4_k",        CL11_CLASS_MVR },
    { "k_mvrc_q6_k",        CL11_CLASS_MVR },
    // K_MMX + cl11_qtype
    { "k_mmx_f32",          CL11_CLASS_MMX },
    { "k_mmx_f16",          CL11_CLASS_MMX },
    { "k_mmx_q4_0",         CL11_CLASS_MMX },
    { "k_mmx_q8_0",         CL11_CLASS_MMX },
    { "k_mmx_q4_k",         CL11_CLASS_MMX },
    { "k_mmx_q6_k",         CL11_CLASS_MMX },
    // K_MVM + cl11_rkind
    { "k_mvm_q4_0",         CL11_CLASS_MVM },
    { "k_mvm_q8_0",         CL11_CLASS_MVM },
    { "k_mvm_q4_k",         CL11_CLASS_MVM },
    { "k_mvm_q6_k",         CL11_CLASS_MVM },
    // K_SET_ROWS2 + 2*(src is F16) + (dst is F16)
    { "k_set_rows2_f32_f32", CL11_CLASS_EW },
    { "k_set_rows2_f32_f16", CL11_CLASS_EW },
    { "k_set_rows2_f16_f32", CL11_CLASS_EW },
    { "k_set_rows2_f16_f16", CL11_CLASS_EW },
    { "k_xprep",            CL11_CLASS_EW  },
    { "k_add_id",           CL11_CLASS_EW  },
    { "k_ssm_conv",         CL11_CLASS_EW  },
    { "k_concat_4",         CL11_CLASS_EW  },
    { "k_concat_2",         CL11_CLASS_EW  },
    { "k_pad",              CL11_CLASS_EW  },
    { "k_sum_rows",         CL11_CLASS_ROW },
    { "k_argsort",          CL11_CLASS_ROW },
    { "k_mid_gather",       CL11_CLASS_ROW },
    { "k_mid_scatter",      CL11_CLASS_ROW },
    // K_MVI + cl11_qtype
    { "k_mvi_f32",          CL11_CLASS_MV  },
    { "k_mvi_f16",          CL11_CLASS_MV  },
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvri_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvri_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvri_*
    { nullptr,             CL11_CLASS_MV  }, // repacked type: k_mvri_*
    // K_MVRI + cl11_rkind
    { "k_mvri_q4_0",        CL11_CLASS_MVR },
    { "k_mvri_q8_0",        CL11_CLASS_MVR },
    { "k_mvki_q4_k",        CL11_CLASS_MVR },
    { "k_mvki_q6_k",        CL11_CLASS_MVR },
};
static_assert(sizeof(cl11_kernel_table) / sizeof(cl11_kernel_table[0]) == K_COUNT, "cl11_kernel_table does not match cl11_kernel_id");

bool cl11_is_k_kernel(int id) {
    return id == K_MVR + CL11_RK_Q4_K || id == K_MVR + CL11_RK_Q6_K || id == K_MVRC + CL11_RK_Q4_K || id == K_MVRC + CL11_RK_Q6_K ||
           id == K_MVM + CL11_RK_Q4_K || id == K_MVM + CL11_RK_Q6_K || id == K_MVRI + CL11_RK_Q4_K || id == K_MVRI + CL11_RK_Q6_K;
}

bool cl11_is_mmx_kernel(int id) {
    return id >= K_MMX && id < K_MMX + CL11_QT_COUNT;
}

bool cl11_is_mm_kernel(int id) {
    return (id >= K_MM && id < K_MM + CL11_QT_COUNT) || cl11_is_mmx_kernel(id);
}

int64_t cl11_sort_nmax(size_t local_mem) {
    int64_t n = 1;
    while (n < (1 << 20) && (size_t) (n * 2) * 8 + 512 <= local_mem) {
        n <<= 1;
    }
    return n;
}

// (3 and 4: cfg.split_programs, see cl11_ensure_program)
bool cl11_build_program(cl_context context, cl_device_id device, const char * logical_name, const char * device_name, size_t local_mem, bool use_nv_asm, bool use_nv_ldg, bool multi_mv, const cl11_config & cfg, std::string & log, int mode, cl_program & prog) {
    char opts[2048];
    snprintf(opts, sizeof(opts), "-cl-std=CL1.1 -DWG_EW=%d -DWG_ROW=%d -DWG_MV=%d -DGT=%d -DNRW=%d -DMMBN=%d"
            " -DWG_MVR_Q4_0=%d -DNRR_Q4_0=%d -DWG_MVR_Q8_0=%d -DNRR_Q8_0=%d -DWG_MVR_Q4_K=%d -DNRR_Q4_K=%d -DWG_MVR_Q6_K=%d -DNRR_Q6_K=%d"
            " -DLDG_Q4_0=%d -DLDG_Q8_0=%d -DLDG_Q4_K=%d -DLDG_Q6_K=%d -DDN_Q4_0=%d -DDN_Q8_0=%d -DDN_Q4_K=%d -DDN_Q6_K=%d -DATT_PL=%d -DWG_ATT=%d -DSORT_NMAX=%d%s%s %s",
            cfg.wg_ew, cfg.wg_row, cfg.wg_mv, cfg.gt, cfg.mvs_nrw, cfg.mm_bn,
            cfg.wg_mvr[CL11_RK_Q4_0], cfg.mvr_nrr[CL11_RK_Q4_0], cfg.wg_mvr[CL11_RK_Q8_0], cfg.mvr_nrr[CL11_RK_Q8_0],
            cfg.wg_mvr[CL11_RK_Q4_K], cfg.mvr_nrr[CL11_RK_Q4_K], cfg.wg_mvr[CL11_RK_Q6_K], cfg.mvr_nrr[CL11_RK_Q6_K],
            use_nv_ldg && cfg.mvr_ldg[CL11_RK_Q4_0], use_nv_ldg && cfg.mvr_ldg[CL11_RK_Q8_0],
            use_nv_ldg && cfg.mvr_ldg[CL11_RK_Q4_K], use_nv_ldg && cfg.mvr_ldg[CL11_RK_Q6_K],
            cfg.mvr_dn[CL11_RK_Q4_0], cfg.mvr_dn[CL11_RK_Q8_0], cfg.mvr_dn[CL11_RK_Q4_K], cfg.mvr_dn[CL11_RK_Q6_K], cfg.att_p, cfg.wg_att,
            (int) cl11_sort_nmax(local_mem),
            use_nv_asm ? " -DCL11_NV_ASM" : "", cfg.use_mad ? " -DCL11_MAD" : "", getenv("GGML_CL11_BUILD_OPTS") ? getenv("GGML_CL11_BUILD_OPTS") : "");
    if (use_nv_ldg) {
        strncat(opts, " -DCL11_NV_LDG", sizeof(opts) - strlen(opts) - 1);
    }
    if (!multi_mv) {
        strncat(opts, " -DCL11_NO_MVM", sizeof(opts) - strlen(opts) - 1);
    }
    if (cfg.mmx.on) {
        char mx[256];
        snprintf(mx, sizeof(mx), " -DCL11_MMX -DMX_BM=%d -DMX_BN=%d -DMX_RM=%d -DMX_CN=%d -DMX_BK=%d -DMX_KU=%d -DMX_PF=%d -DMX_LDG=%d -DMX_REQD=%d",
                cfg.mmx.bm, cfg.mmx.bn, cfg.mmx.rm, cfg.mmx.cn, cfg.mmx.bk, cfg.mmx.ku, cfg.mmx.pf, use_nv_ldg && cfg.mmx.ldg ? 1 : 0, cfg.mmx.reqd);
        strncat(opts, mx, sizeof(opts) - strlen(opts) - 1);
    }
    if (mode == 1) {
        strncat(opts, " -DCL11_NO_K", sizeof(opts) - strlen(opts) - 1);
    } else if (mode == 2) {
        char rr[96];
        if (cfg.nv_maxrreg > 0 && use_nv_asm) { // an NVIDIA option: other compilers (FORCE_GEN=kepler on a Mac) must not get it
            snprintf(rr, sizeof(rr), " -DCL11_PROG_K -cl-nv-maxrregcount=%d", cfg.nv_maxrreg);
        } else {
            snprintf(rr, sizeof(rr), " -DCL11_PROG_K");
        }
        strncat(opts, rr, sizeof(opts) - strlen(opts) - 1);
    } else if (mode == 3) {
        strncat(opts, " -DCL11_NO_K -DCL11_NO_MM", sizeof(opts) - strlen(opts) - 1);
    } else if (mode == 4) {
        strncat(opts, " -DCL11_ONLY_MM", sizeof(opts) - strlen(opts) - 1);
        if (cfg.mmx.on && cfg.mmx.maxrreg > 0 && use_nv_asm) {
            char rr[64];
            snprintf(rr, sizeof(rr), " -cl-nv-maxrregcount=%d", cfg.mmx.maxrreg);
            strncat(opts, rr, sizeof(opts) - strlen(opts) - 1);
        }
    }

    // GGML_CL11_KERNEL_DIR: read the .cl files from this directory instead of the embedded copies (development)
    std::vector<std::string> file_sources;
    std::vector<const char *> sources(ggml_cl11_sources, ggml_cl11_sources + CL11_NUM_SOURCES);
    if (const char * dir = getenv("GGML_CL11_KERNEL_DIR")) {
        file_sources.resize(CL11_NUM_SOURCES);
        for (size_t i = 0; i < CL11_NUM_SOURCES; ++i) {
            const std::string path = std::string(dir) + "/" + ggml_cl11_source_names[i];
            std::ifstream f(path, std::ios::binary);
            if (!f) {
                fprintf(stderr, "%s: cannot read %s\n", __func__, path.c_str());
                return false;
            }
            std::stringstream ss;
            ss << f.rdbuf();
            file_sources[i] = ss.str();
            sources[i] = file_sources[i].c_str();
        }
        fprintf(stderr, "%s: using the kernel sources of %s\n", __func__, dir);
    }

    cl_int err = CL_SUCCESS;
    prog = clCreateProgramWithSource(context, (cl_uint) CL11_NUM_SOURCES, sources.data(), nullptr, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: clCreateProgramWithSource failed: %s\n", __func__, cl11_err_str(err));
        return false;
    }

    if (getenv("GGML_CL11_PRINT_BUILD_OPTS") != nullptr) {
        fprintf(stderr, "%s: %s build options mode %d: %s\n", __func__, logical_name, mode, opts);
    }
    err = clBuildProgram(prog, 1, &device, opts, nullptr, nullptr);

    size_t log_size = 0;
    clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    log.assign(log_size, '\0');
    if (log_size > 0) {
        clGetProgramBuildInfo(prog, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
    }

    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s: failed to build the kernels for %s (%s) with options '%s': %s\n%s\n", __func__,
                logical_name, device_name, opts, cl11_err_str(err), log.c_str());
        clReleaseProgram(prog);
        prog = nullptr;
        return false;
    }
    if (const char * path = getenv("GGML_CL11_DUMP_BINARY")) {
        // development: write the program binary (PTX for NVIDIA) to this file
        size_t bin_size = 0;
        if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr) == CL_SUCCESS && bin_size > 0) {
            std::vector<unsigned char> bin(bin_size);
            unsigned char * ptr = bin.data();
            if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(ptr), &ptr, nullptr) == CL_SUCCESS) {
                std::ofstream f(path, std::ios::binary);
                f.write((const char *) bin.data(), (std::streamsize) bin_size);
            }
        }
    }
    return true;
}
