#include "device_state.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static bool cl11_debug() {
    static const bool debug = getenv("GGML_CL11_DEBUG") != nullptr;
    return debug;
}

//
// lazy context / program creation
//

// The "denormal products" kernels (cfg.mvr_dn) need a device that multiplies denormal floats exactly (an FFMA without flush to zero): a nibble or byte
// masked in place is used as the float n * 2^(p - 149). A one work-item kernel checks it (all the bit positions the kernels use, exact power of
// two scales); a device that flushes denormals (Apple, some Mesa drivers) gets the normal kernels.
static bool cl11_dn_probe(cl11_device * dev) {
    static const char * src =
        "kernel void k_dn_probe(global float * out, uint w, uint t, float x) {\n"
        "    float s = 0.0f;\n"
        "    s = fma(as_float(w & 0x0000000Fu), x * as_float(0x71800000u), s);\n"       // 2^100
        "    s = fma(as_float(w & 0x00000F00u), x * as_float(0x6D800000u), s);\n"       // 2^92
        "    s = fma(as_float(w & 0x00F00000u), x * as_float(0x67800000u), s);\n"       // 2^80
        "    s = fma(as_float(t & 0x0Fu), x * as_float(0x71800000u), s);\n"
        "    out[0] = s;\n"
        "}\n";
    cl_int err = CL_SUCCESS;
    cl_program prog = clCreateProgramWithSource(dev->context, 1, &src, nullptr, &err);
    if (err != CL_SUCCESS) {
        return false;
    }
    bool ok = false;
    if (clBuildProgram(prog, 1, &dev->info.device, "-cl-std=CL1.1", nullptr, nullptr) == CL_SUCCESS) {
        cl_kernel k = clCreateKernel(prog, "k_dn_probe", &err);
        cl_mem out = err == CL_SUCCESS ? clCreateBuffer(dev->context, CL_MEM_READ_WRITE, 16, nullptr, &err) : nullptr;
        if (err == CL_SUCCESS) {
            const cl_uint w = 0x00A00B05u, t = 0x0Cu; // nibbles 5 (bit 0), 11 (bit 8), 10 (bit 20), t: 12
            const cl_float x = 1.0f;
            clSetKernelArg(k, 0, sizeof(cl_mem), &out);
            clSetKernelArg(k, 1, sizeof(cl_uint), &w);
            clSetKernelArg(k, 2, sizeof(cl_uint), &t);
            clSetKernelArg(k, 3, sizeof(cl_float), &x);
            const size_t one = 1;
            float res = 0.0f;
            if (clEnqueueNDRangeKernel(dev->queue, k, 1, nullptr, &one, &one, 0, nullptr, nullptr) == CL_SUCCESS &&
                clEnqueueReadBuffer(dev->queue, out, CL_TRUE, 0, sizeof(res), &res, 0, nullptr, nullptr) == CL_SUCCESS) {
                // every product is n * 2^-49
                const double expect = (5.0 + 11.0 + 10.0 + 12.0) * std::ldexp(1.0, -49);
                ok = (double) res == expect;
            }
        }
        if (out) {
            clReleaseMemObject(out);
        }
        if (k) {
            clReleaseKernel(k);
        }
    }
    clReleaseProgram(prog);
    return ok;
}

static void cl11_release_kernels(cl11_device * dev) {
    for (int i = 0; i < K_COUNT; ++i) {
        if (dev->kernels[i]) {
            clReleaseKernel(dev->kernels[i]);
            dev->kernels[i] = nullptr;
        }
    }
    if (dev->program) {
        clReleaseProgram(dev->program);
        dev->program = nullptr;
    }
    if (dev->program_k) {
        clReleaseProgram(dev->program_k);
        dev->program_k = nullptr;
    }
    if (dev->program_mm) {
        clReleaseProgram(dev->program_mm);
        dev->program_mm = nullptr;
    }
    dev->program_k_ready = false;
}

// creates cl_context and the command queue, the caller holds dev->mutex
bool cl11_ensure_context(cl11_device * dev) {
    if (dev->context_ready) {
        return true;
    }
    if (dev->failed) {
        return false;
    }

    if (!cl11_ensure_runtime_context(dev)) {
        return false;
    }

    if (const char * env = getenv("GGML_CL11_MV_MAX_COLS")) {
        dev->mv_max_cols = std::max(0, atoi(env));
    }
    if (const char * env = getenv("GGML_CL11_MID_MAX_TOKENS")) {
        dev->mid_max_tokens = std::max(0, atoi(env));
    }
    if (const char * env = getenv("GGML_CL11_MV_TPR")) {
        dev->mv_tpr = std::max(0, atoi(env));
    }
    if (const char * env = getenv("GGML_CL11_MV_STAGED")) {
        dev->mvs_enabled = atoi(env) != 0;
    }
    dev->no_cpy_t = getenv("GGML_CL11_NO_CPY_T") != nullptr;
    dev->no_mvt   = getenv("GGML_CL11_NO_MVT") != nullptr;
    dev->no_mvq   = getenv("GGML_CL11_NO_MVQ") != nullptr;
    dev->no_fuse  = getenv("GGML_CL11_NO_FUSE") != nullptr;
    {
        char vendor[256] = {};
        clGetPlatformInfo(dev->info.platform, CL_PLATFORM_VENDOR, sizeof(vendor) - 1, vendor, nullptr);
        dev->use_nv_asm = strstr(vendor, "NVIDIA") != nullptr && getenv("GGML_CL11_NO_ASM") == nullptr;
        dev->use_nv_ldg = dev->use_nv_asm && dev->info.nv_cc_major * 10 + dev->info.nv_cc_minor >= 35 && getenv("GGML_CL11_NO_LDG") == nullptr;
    }
    {
        const cl11_tuning t = cl11_tuning_for(dev->info);
        dev->mvs_min_k  = t.mvs_min_k;
        dev->mvs_family = t.mvs_family;
        dev->mvr_const_x = t.mvr_const_x;
        dev->max_const  = (size_t) cl11_device_val<cl_ulong>(dev->info.device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    }
    if (const char * env = getenv("GGML_CL11_MVR_CONST_X")) {
        dev->mvr_const_x = atoi(env) != 0;
    }
    if (const char * env = getenv("GGML_CL11_MVS_FAMILY")) {
        dev->mvs_family = atoi(env);
    }
    if (const char * env = getenv("GGML_CL11_MVS_MIN_K")) {
        dev->mvs_min_k = std::max(0, atoi(env));
    }

    return true;
}

// builds the program and creates the kernels, the caller holds dev->mutex.
// The work-group sizes are not hard coded: they start from what the device allows and are reduced
// (and the program is rebuilt) until every kernel can be launched with them.
bool cl11_ensure_program(cl11_device * dev) {
    if (dev->program_ready) {
        return true;
    }
    if (!cl11_ensure_context(dev)) {
        return false;
    }

    const cl11_tuning tuning = cl11_tuning_for(dev->info);
    cl11_config cfg = tuning.cfg;
    {
        size_t limit_wg = std::min(dev->info.max_wg, dev->info.max_item0);
        if (const char * env = getenv("GGML_CL11_MAX_WG")) {
            // debugging / tuning: cap the work-group size
            limit_wg = std::min<size_t>(limit_wg, (size_t) std::max(1, atoi(env)));
        }
        const int cap = cl11_pow2_floor(limit_wg);
        cfg.wg_ew  = std::min(cfg.wg_ew,  cap);
        cfg.wg_row = std::min(cfg.wg_row, cap);
        if (const char * env = getenv("GGML_CL11_MAD")) {
            cfg.use_mad = atoi(env) != 0;
        }
        if (const char * env = getenv("GGML_CL11_FLUSH_SORT")) {
            cfg.flush_after_sort = atoi(env) != 0;
        }
        if (const char * env = getenv("GGML_CL11_MVS_NRW")) {
            cfg.mvs_nrw = std::max(1, std::min(8, atoi(env)));
        }
        if (const char * env = getenv("GGML_CL11_WG_MV")) {
            cfg.wg_mv = cl11_pow2_floor((size_t) std::max(1, atoi(env)));
        }
        cfg.wg_mv  = std::min(cfg.wg_mv,  cap);
        // GGML_CL11_WG_MVR / GGML_CL11_MVR_NRR set all the types, the variables with the suffix _Q4_0, _Q8_0, _Q4_K, _Q6_K one type
        static const char * const rk_names[CL11_RK_COUNT] = { "Q4_0", "Q8_0", "Q4_K", "Q6_K" };
        for (int k = 0; k < CL11_RK_COUNT; ++k) {
            const char * env = getenv((std::string("GGML_CL11_WG_MVR_") + rk_names[k]).c_str());
            if (env == nullptr) {
                env = getenv("GGML_CL11_WG_MVR");
            }
            if (env != nullptr) {
                cfg.wg_mvr[k] = cl11_pow2_floor((size_t) std::max(1, atoi(env)));
            }
            env = getenv((std::string("GGML_CL11_MVR_LDG_") + rk_names[k]).c_str());
            if (env == nullptr) {
                env = getenv("GGML_CL11_MVR_LDG");
            }
            if (env != nullptr) {
                cfg.mvr_ldg[k] = atoi(env) != 0;
            }
            env = getenv((std::string("GGML_CL11_MVR_DN_") + rk_names[k]).c_str());
            if (env == nullptr) {
                env = getenv("GGML_CL11_MVR_DN");
            }
            if (env != nullptr) {
                cfg.mvr_dn[k] = atoi(env) != 0;
            }
            env = getenv((std::string("GGML_CL11_MVR_NRR_") + rk_names[k]).c_str());
            if (env == nullptr) {
                env = getenv("GGML_CL11_MVR_NRR");
            }
            if (env != nullptr) {
                cfg.mvr_nrr[k] = std::max(1, std::min(8, atoi(env)));
            }
            cfg.wg_mvr[k] = std::min(cfg.wg_mvr[k], cap);
        }
        if (const char * env = getenv("GGML_CL11_WG_ATT")) {
            cfg.wg_att = cl11_pow2_floor((size_t) std::max(32, atoi(env)));
        }
        cfg.wg_att = std::min(cfg.wg_att, cap);
        if (const char * env = getenv("GGML_CL11_ATT_P")) {
            cfg.att_p = std::max(1, std::min(16, atoi(env)));
        }
        // the single work-group V cache kernel needs >= 512 work-items (4 slices of the positions for Qwen3-0.6B); below that the many-groups kernel is used
        cfg.mvt_groups = cfg.mvt_groups || cfg.wg_att < 512;
        if (const char * env = getenv("GGML_CL11_MVT_GROUPS")) {
            cfg.mvt_groups = atoi(env) != 0;
        }
        if (const char * env = getenv("GGML_CL11_NV_MAXRREG")) {
            cfg.nv_maxrreg = std::max(0, atoi(env));
        }
        if (const char * env = getenv("GGML_CL11_MM_BN")) {
            cfg.mm_bn = atoi(env) >= 8 ? 8 : 4;
        }
        while (cfg.gt * cfg.gt > cap && cfg.gt > 4) {
            cfg.gt /= 2;
        }
    }

    // parameterized mat-mat kernels: the tuning (cl11_tuning::mmx_variant), overridden by GGML_CL11_MM_VARIANT (a preset or "key=value,...") and
    // GGML_CL11_MMX ("key=value,..." added to it); invalid combinations are refused with a message and the default kernels are used
    {
        std::string spec = tuning.mmx_variant ? tuning.mmx_variant : "";
        bool specified = false;
        if (const char * env = getenv("GGML_CL11_MM_VARIANT")) {
            spec = env;
            specified = true;
        }
        if (const char * env = getenv("GGML_CL11_MMX")) {
            spec += std::string(spec.empty() || spec == "base" || spec == "off" ? "" : ",") + env;
            specified = true;
        }
        if (specified && (spec.find("base") == 0 || spec == "off" || spec == "0")) {
            spec.clear();
        }
        cl11_mmx mx;
        std::string why;
        if (!cl11_mmx_parse(spec, mx)) {
            mx = cl11_mmx();
        } else if (mx.on && !cl11_mmx_valid(mx, dev->info.local_mem, std::min(dev->info.max_wg, dev->info.max_item0), why)) {
            fprintf(stderr, "%s: mat-mat variant '%s' not usable on %s: %s; using the default mat-mat kernels\n", __func__, spec.c_str(), dev->name.c_str(), why.c_str());
            mx = cl11_mmx();
        }
        cfg.mmx = mx;
    }

    // split_programs: the decode kernels, the mat-mat kernels and the K-quant kernels in three programs (Mesa Clover: the launch cost grows with the
    // size of the program, see common.cl). Otherwise the Q4_K / Q6_K mat-vec kernels are apart when they run faster with a cap on the registers
    // (the others slower) in a second program. The programs of the K-quant kernels are built when needed.
    cfg.mvr_dn[CL11_RK_Q8_0] = 0; // no "denormal products" kernel for Q8_0
    // Fermi's "denormal products" (cfg.mvr_dn) and Kepler's multi mat-vec (multi_mv) are exclusive, and src1 in constant memory (mvr_const_x) decides:
    // the DN kernels are the k_mvrc_* kernels (src1 prepared by k_xprep is read from constant memory), cl11_mvm_candidate() refuses the multi mat-vec
    // where mvr_const_x is set (and graph_optimize then does not move the nodes). Fermi: const_x on, multi_mv off, DN on. Kepler: const_x off (sub-buffer
    // offsets of __constant arguments are broken in the driver), multi_mv on, DN has no kernel to run: it is turned off here. Both forced
    // (GGML_CL11_MULTI_MV=1 and GGML_CL11_MVR_DN=1) on a device with const_x: DN wins, multi_mv is unused; with GGML_CL11_MVR_CONST_X=0: multi_mv wins.
    if (!dev->mvr_const_x && (cfg.mvr_dn[CL11_RK_Q4_0] || cfg.mvr_dn[CL11_RK_Q4_K] || cfg.mvr_dn[CL11_RK_Q6_K])) {
        if (getenv("GGML_CL11_MVR_DN") != nullptr || getenv("GGML_CL11_MVR_DN_Q4_0") != nullptr || getenv("GGML_CL11_MVR_DN_Q4_K") != nullptr || getenv("GGML_CL11_MVR_DN_Q6_K") != nullptr) {
            fprintf(stderr, "%s: %s: the \"denormal products\" kernels need src1 in constant memory (GGML_CL11_MVR_CONST_X=1), disabled\n", __func__, dev->name.c_str());
        }
        for (int k = 0; k < CL11_RK_COUNT; ++k) {
            cfg.mvr_dn[k] = 0;
        }
    }
    if (cfg.mvr_dn[CL11_RK_Q4_0] || cfg.mvr_dn[CL11_RK_Q8_0] || cfg.mvr_dn[CL11_RK_Q4_K] || cfg.mvr_dn[CL11_RK_Q6_K]) {
        if (getenv("GGML_CL11_DN_NOPROBE") == nullptr && !cl11_dn_probe(dev)) {
            fprintf(stderr, "%s: %s does not multiply denormal floats exactly, the \"denormal products\" kernels are disabled\n", __func__, dev->name.c_str());
            for (int k = 0; k < CL11_RK_COUNT; ++k) {
                cfg.mvr_dn[k] = 0;
            }
        }
    }
    dev->split_programs = tuning.split_programs;
    if (const char * env = getenv("GGML_CL11_SPLIT_PROGRAMS")) {
        dev->split_programs = atoi(env) != 0;
    }
    if (cfg.mmx.on && cfg.mmx.maxrreg > 0 && dev->use_nv_asm) {
        dev->split_programs = true; // the register cap applies to the program of the mat-mat kernels only
    }
    dev->multi_mv = tuning.multi_mv;
    if (const char * env = getenv("GGML_CL11_MULTI_MV")) {
        dev->multi_mv = atoi(env) != 0;
    }
    dev->fuse_set_rows = tuning.fuse_set_rows && !dev->no_fuse;
    if (const char * env = getenv("GGML_CL11_FUSE_SET_ROWS")) {
        dev->fuse_set_rows = atoi(env) != 0;
    }
    dev->k_separate = (dev->use_nv_asm && cfg.nv_maxrreg > 0) || dev->split_programs;

    for (int attempt = 0; attempt < 8; ++attempt) {
        std::string log;
        if (!cl11_build_program(dev->context, dev->info.device, dev->name.c_str(), dev->info.name.c_str(), dev->info.local_mem,
                    dev->use_nv_asm, dev->use_nv_ldg, dev->multi_mv, cfg, log, dev->split_programs ? 3 : dev->k_separate ? 1 : 0, dev->program)) {
            dev->failed = true;
            return false;
        }
        if (dev->split_programs && !cl11_build_program(dev->context, dev->info.device, dev->name.c_str(), dev->info.name.c_str(), dev->info.local_mem,
                    dev->use_nv_asm, dev->use_nv_ldg, dev->multi_mv, cfg, log, 4, dev->program_mm)) {
            cl11_release_kernels(dev);
            dev->failed = true;
            return false;
        }

        bool ok = true;
        for (int i = 0; i < K_COUNT && ok; ++i) {
            if (cl11_kernel_table[i].name == nullptr || (dev->k_separate && cl11_is_k_kernel(i)) || (cl11_is_mmx_kernel(i) && !cfg.mmx.on) ||
                (cl11_kernel_table[i].cls == CL11_CLASS_MVM && !dev->multi_mv) || (i >= K_SET_ROWS2 && i < K_XPREP && !dev->fuse_set_rows) ||
                (i == K_XPREP && !(cfg.mvr_dn[CL11_RK_Q4_0] || cfg.mvr_dn[CL11_RK_Q4_K] || cfg.mvr_dn[CL11_RK_Q6_K]))) {
                continue;
            }
            cl_int err = CL_SUCCESS;
            dev->kernels[i] = clCreateKernel(dev->split_programs && cl11_is_mm_kernel(i) ? dev->program_mm : dev->program, cl11_kernel_table[i].name, &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "%s: clCreateKernel(%s) failed: %s\n", __func__, cl11_kernel_table[i].name, cl11_err_str(err));
                cl11_release_kernels(dev);
                dev->failed = true;
                return false;
            }
        }

        // the work-group size limit of each kernel depends on its register usage
        int limit[CL11_CLASS_COUNT];
        for (int i = 0; i < CL11_CLASS_COUNT; ++i) {
            limit[i] = INT_MAX;
        }
        for (int i = 0; i < K_COUNT; ++i) {
            if (dev->kernels[i] == nullptr) {
                continue;
            }
            size_t wgs = 0;
            if (clGetKernelWorkGroupInfo(dev->kernels[i], dev->info.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr) != CL_SUCCESS || wgs == 0) {
                wgs = 1;
            }
            const int cls = cl11_kernel_table[i].cls;
            if (cls == CL11_CLASS_MVM) {
                dev->mvm_wgs[i - K_MVM] = wgs;
            }
            limit[cls] = std::min(limit[cls], (int) std::min<size_t>(wgs, INT_MAX));
            if (cl11_debug()) {
                fprintf(stderr, "%s: kernel %-20s max work-group size %zu\n", __func__, cl11_kernel_table[i].name, wgs);
            }
        }

        cl11_config next = cfg;
        if (limit[CL11_CLASS_EW]  < cfg.wg_ew)  { next.wg_ew  = cl11_pow2_floor(limit[CL11_CLASS_EW]);  }
        if (limit[CL11_CLASS_ROW] < cfg.wg_row) { next.wg_row = cl11_pow2_floor(limit[CL11_CLASS_ROW]); }
        if (limit[CL11_CLASS_MV]  < cfg.wg_mv)  { next.wg_mv  = cl11_pow2_floor(limit[CL11_CLASS_MV]);  }
        if (limit[CL11_CLASS_ATT] < cfg.wg_att) { next.wg_att = cl11_pow2_floor(limit[CL11_CLASS_ATT]); }
        bool mvr_changed = false;
        int  wg_mvr_max  = 0;
        for (int k = 0; k < CL11_RK_COUNT; ++k) {
            wg_mvr_max = std::max(wg_mvr_max, cfg.wg_mvr[k]);
            if (limit[CL11_CLASS_MVR] < cfg.wg_mvr[k]) {
                next.wg_mvr[k] = cl11_pow2_floor(limit[CL11_CLASS_MVR]);
                mvr_changed = true;
            }
        }
        if (limit[CL11_CLASS_MM]  < cfg.gt * cfg.gt) {
            while (next.gt > 4 && next.gt * next.gt > limit[CL11_CLASS_MM]) {
                next.gt /= 2;
            }
        }
        bool mmx_dropped = false;
        if (cfg.mmx.on && limit[CL11_CLASS_MMX] < cfg.mmx.nt()) {
            fprintf(stderr, "%s: the mat-mat kernels of the variant need %d work-items but %s accepts %d: using the default mat-mat kernels\n", __func__,
                    cfg.mmx.nt(), dev->name.c_str(), limit[CL11_CLASS_MMX]);
            next.mmx = cl11_mmx();
            mmx_dropped = true;
        }

        if (next.wg_ew == cfg.wg_ew && next.wg_row == cfg.wg_row && next.wg_mv == cfg.wg_mv && next.wg_att == cfg.wg_att && !mvr_changed && next.gt == cfg.gt && !mmx_dropped) {
            if (limit[CL11_CLASS_EW] < cfg.wg_ew || limit[CL11_CLASS_ROW] < cfg.wg_row ||
                limit[CL11_CLASS_MV] < cfg.wg_mv || limit[CL11_CLASS_ATT] < cfg.wg_att || limit[CL11_CLASS_MVR] < wg_mvr_max || limit[CL11_CLASS_MM]  < cfg.gt * cfg.gt) {
                fprintf(stderr, "%s: the kernels cannot be launched on %s (work-group size limits %d/%d/%d/%d)\n", __func__,
                        dev->name.c_str(), limit[0], limit[1], limit[2], limit[3]);
                cl11_release_kernels(dev);
                dev->failed = true;
                return false;
            }
            // all kernels accept the configuration
            dev->cfg = cfg;
            dev->program_ready = true;
            if (dev->multi_mv && !dev->mvr_const_x) { // with src1 in constant memory (Fermi) cl11_mvm_candidate() refuses every node
                fprintf(stderr, "%s: %s: multi mat-vec (Q / K / V, FFN gate / up in one launch) enabled\n", __func__, dev->name.c_str());
            }
            if (cfg.mmx.on) {
                fprintf(stderr, "%s: %s: mat-mat variant: tile %d x %d, %d x %d values per work-item, BK %d, unroll %d, prefetch %d, %d work-items\n", __func__, dev->name.c_str(),
                        cfg.mmx.bm, cfg.mmx.bn, cfg.mmx.rm, cfg.mmx.cn, cfg.mmx.bk, cfg.mmx.ku, cfg.mmx.pf, cfg.mmx.nt());
            }
            fprintf(stderr, "%s: %s: %s, %s, kernels built (work-group sizes: element-wise %d, row %d, mat-vec %d, mat-mat %dx%d)\n",
                    __func__, dev->name.c_str(), dev->info.name.c_str(), dev->info.device_version.c_str(),
                    cfg.wg_ew, cfg.wg_row, cfg.wg_mv, cfg.gt, cfg.gt);
            return true;
        }

        fprintf(stderr, "%s: rebuilding the kernels with smaller work-groups\n", __func__);
        cl11_release_kernels(dev);
        cfg = next;
    }

    fprintf(stderr, "%s: could not find a working work-group configuration for %s\n", __func__, dev->name.c_str());
    dev->failed = true;
    return false;
}

// builds the program of the Q4_K / Q6_K mat-vec kernels (only when cl11_ensure_program put them apart), the caller holds dev->mutex
bool cl11_ensure_kprogram(cl11_device * dev) {
    if (!dev->k_separate || dev->program_k_ready) {
        return true;
    }
    cl11_config cfg = dev->cfg;
    for (int attempt = 0; attempt < 4; ++attempt) {
        std::string log;
        if (!cl11_build_program(dev->context, dev->info.device, dev->name.c_str(), dev->info.name.c_str(), dev->info.local_mem,
                    dev->use_nv_asm, dev->use_nv_ldg, dev->multi_mv, cfg, log, 2, dev->program_k)) {
            return false;
        }
        int limit = INT_MAX;
        bool ok = true;
        for (int i = 0; i < K_COUNT && ok; ++i) {
            if (!cl11_is_k_kernel(i) || (cl11_kernel_table[i].cls == CL11_CLASS_MVM && !dev->multi_mv)) {
                continue;
            }
            cl_int err = CL_SUCCESS;
            dev->kernels[i] = clCreateKernel(dev->program_k, cl11_kernel_table[i].name, &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "%s: clCreateKernel(%s) failed: %s\n", __func__, cl11_kernel_table[i].name, cl11_err_str(err));
                return false;
            }
            size_t wgs = 0;
            if (clGetKernelWorkGroupInfo(dev->kernels[i], dev->info.device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wgs), &wgs, nullptr) != CL_SUCCESS || wgs == 0) {
                wgs = 1;
            }
            if (cl11_kernel_table[i].cls == CL11_CLASS_MVM) {
                dev->mvm_wgs[i - K_MVM] = wgs; // only decides whether the fusion can be used with the work-group size of the type
            } else {
                limit = std::min(limit, (int) std::min<size_t>(wgs, INT_MAX));
            }
        }
        if (limit >= cfg.wg_mvr[CL11_RK_Q4_K] && limit >= cfg.wg_mvr[CL11_RK_Q6_K]) {
            dev->cfg.wg_mvr[CL11_RK_Q4_K] = cfg.wg_mvr[CL11_RK_Q4_K];
            dev->cfg.wg_mvr[CL11_RK_Q6_K] = cfg.wg_mvr[CL11_RK_Q6_K];
            dev->program_k_ready = true;
            return true;
        }
        cfg.wg_mvr[CL11_RK_Q4_K] = std::min(cfg.wg_mvr[CL11_RK_Q4_K], cl11_pow2_floor(limit));
        cfg.wg_mvr[CL11_RK_Q6_K] = std::min(cfg.wg_mvr[CL11_RK_Q6_K], cl11_pow2_floor(limit));
        for (int i = 0; i < K_COUNT; ++i) {
            if (cl11_is_k_kernel(i) && dev->kernels[i]) {
                clReleaseKernel(dev->kernels[i]);
                dev->kernels[i] = nullptr;
            }
        }
        clReleaseProgram(dev->program_k);
        dev->program_k = nullptr;
    }
    fprintf(stderr, "%s: could not find a working work-group configuration for the K-quant kernels of %s\n", __func__, dev->name.c_str());
    return false;
}

// scratch buffer of k_xprep for the pattern `slot` (one at most 64 KiB column of src1 scaled + the sums of its groups of 16), created on first use
cl_mem cl11_xprep_scratch(cl11_device * dev, int slot) {
    if (dev->xprep[slot].buf == nullptr) {
        cl_int err = CL_SUCCESS;
        dev->xprep[slot].buf = clCreateBuffer(dev->context, CL_MEM_READ_WRITE, 65536, nullptr, &err);
        if (err != CL_SUCCESS) {
            dev->xprep[slot].buf = nullptr;
        }
    }
    return dev->xprep[slot].buf;
}

bool cl11_mid_ensure(cl11_device * dev, cl11_device::mid_buffer & b, size_t bytes) {
    if (b.buf != nullptr && b.size >= bytes) {
        return true;
    }
    if (b.buf != nullptr) {
        clReleaseMemObject(b.buf);
        b.buf = nullptr;
        b.size = 0;
    }
    cl_int err = CL_SUCCESS;
    b.buf = clCreateBuffer(dev->context, CL_MEM_READ_WRITE, std::max<size_t>(bytes, 16), nullptr, &err);
    if (err != CL_SUCCESS || b.buf == nullptr) {
        fprintf(stderr, "%s: cannot allocate %zu bytes of scratch memory for MUL_MAT_ID: %s\n", __func__, bytes, cl11_err_str(err));
        b.buf = nullptr;
        return false;
    }
    if (!cl11_probe_buffer(dev->queue, b.buf, std::max<size_t>(bytes, 16))) { // lazy allocation of the driver, see alloc_buffer
        fprintf(stderr, "%s: cannot allocate %zu bytes of scratch memory for MUL_MAT_ID: out of device memory\n", __func__, bytes);
        clReleaseMemObject(b.buf);
        b.buf = nullptr;
        return false;
    }
    b.size = std::max<size_t>(bytes, 16);
    return true;
}
