#ifndef SCRAPGPU_OPENCL11_DEVICE_H
#define SCRAPGPU_OPENCL11_DEVICE_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 110
#endif

#ifdef __APPLE__
#  ifndef CL_SILENCE_DEPRECATION
#    define CL_SILENCE_DEPRECATION
#  endif
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif

#include <cstddef>
#include <string>

struct cl11_device_info {
    cl_platform_id platform = nullptr;
    cl_device_id   device   = nullptr;

    std::string name;
    std::string platform_name;
    std::string vendor;
    std::string device_version;
    std::string c_version;

    size_t max_alloc   = 0;
    size_t global_mem  = 0;
    size_t local_mem   = 0;
    size_t max_wg      = 0;
    size_t max_item0   = 0;
    size_t align_bytes = 0;
    int    compute_units = 0;

    int nv_cc_major = 0;
    int nv_cc_minor = 0;

    int amd_gfx_major = 0;
};

// AMD generation, used by the tuning (cl11_tuning_for) and the device info of the C API
enum cl11_amd_gen {
    CL11_AMD_NONE,    // not an AMD GPU
    CL11_AMD_GCN,     // GCN (gfx6 - gfx9: Southern Islands .. Vega); CDNA is GCN-based as well
    CL11_AMD_RDNA,    // RDNA 1 - 4 (gfx10 and later)
    CL11_AMD_UNKNOWN, // AMD, generation not determined
};

cl11_amd_gen cl11_amd_generation(const cl11_device_info & info);

// Known driver/device combinations that must not create a compute context.
// Returns a static reason, or nullptr. Shared by the C API and ggml adapter.
const char * cl11_device_rejection_reason(const cl11_device_info & info);

enum cl11_rkind {
    CL11_RK_Q4_0,
    CL11_RK_Q8_0,
    CL11_RK_Q4_K,
    CL11_RK_Q6_K,
    CL11_RK_COUNT,
};

struct cl11_mmx {
    bool on = false;
    int  bm = 64, bn = 64;
    int  rm = 8, cn = 4;
    int  bk = 16;
    int  ku = 16;
    int  pf = 0;
    int  ldg = 0;
    int  reqd = 1;
    int  maxrreg = 0;
    int  nt() const { return (bm / rm) * (bn / cn); }
};

struct cl11_config {
    int wg_ew  = 128;
    int wg_row = 128;
    int wg_mv  = 128;
    int gt     = 16;
    int mvs_nrw = 2;
    int wg_mvr[CL11_RK_COUNT]  = { 64, 64, 64, 64 };
    int mvr_nrr[CL11_RK_COUNT] = { 2, 2, 2, 2 };
    int mvr_ldg[CL11_RK_COUNT] = { 0, 0, 0, 0 };
    int mvr_dn[CL11_RK_COUNT]  = { 0, 0, 0, 0 };
    int att_p   = 4;
    int wg_att  = 1024;
    int nv_maxrreg = 0;
    int mm_bn   = 8;
    bool use_mad = false;
    cl11_mmx mmx;
    bool mvt_groups = false;
};

struct cl11_tuning {
    cl11_config cfg;
    int         mvs_min_k   = 256;
    int         mvs_family  = 0;
    bool        mvr_const_x = false;
    bool        strict_alloc = false;
    bool        fuse_set_rows = false;
    bool        multi_mv   = false;
    const char * mmx_variant = nullptr;
    bool        split_programs = false;
    const char * name       = "default";
};

int cl11_pow2_floor(size_t x);
std::string cl11_lower(std::string s);
std::string cl11_device_str(cl_device_id dev, cl_device_info param);

template <typename T>
T cl11_device_val(cl_device_id dev, cl_device_info param, T def = T()) {
    T v = def;
    if (clGetDeviceInfo(dev, param, sizeof(T), &v, nullptr) != CL_SUCCESS) {
        return def;
    }
    return v;
}

cl11_tuning cl11_tuning_for(const cl11_device_info & info);
bool cl11_mmx_parse(const std::string & spec_in, cl11_mmx & m);
bool cl11_mmx_valid(const cl11_mmx & m, size_t local_mem, size_t max_wg, std::string & why);
bool cl11_device_selected(const std::string & name);
bool cl11_platform_selected(const std::string & name);
// `dropped` (optional) is called for each device that cannot be opened next to an earlier one (the NVIDIA 390 Fermi / Kepler conflict),
// so that the caller can serve it from another process. Without it the device is only dropped with a warning.
bool cl11_probe_device_infos(void (*emit)(const cl11_device_info & info, bool strict_alloc),
                             void (*dropped)(const cl11_device_info & info) = nullptr);

#endif
