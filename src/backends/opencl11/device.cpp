#include "device.h"
#include "context.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool cl11_backend_debug() {
    static const bool debug = getenv("GGML_CL11_DEBUG") != nullptr;
    return debug;
}

int cl11_pow2_floor(size_t x) {
    int p = 1;
    while ((size_t) p * 2 <= x) {
        p *= 2;
    }
    return p;
}

//
// device enumeration
//

std::string cl11_device_str(cl_device_id dev, cl_device_info param) {
    size_t size = 0;
    if (clGetDeviceInfo(dev, param, 0, nullptr, &size) != CL_SUCCESS || size == 0) {
        return "";
    }
    std::string s(size, '\0');
    if (clGetDeviceInfo(dev, param, size, &s[0], nullptr) != CL_SUCCESS) {
        return "";
    }
    while (!s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

std::string cl11_lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

// AMD generation, for the tuning of cl11_tuning_for(). Mesa (Clover, rusticl) does not implement cl_amd_device_attribute_query, so the
// generation is read from, in this order:
//  1. CL_DEVICE_GFXIP_MAJOR_AMD (cl_amd_device_attribute_query: AMD's own OpenCL drivers, ROCm): 6..9 = GCN, >= 10 = RDNA;
//  2. a "gfxNNN" token in the device name (ROCm names the device after the ISA: "gfx803", "gfx90a", "gfx1034", "gfx10-3-generic");
//  3. the code name that Mesa puts into the device name: rusticl "AMD Radeon Pro WX 2100 (radeonsi, polaris12, LLVM 19.1.7, DRM 3.61, 6.12.x)",
//     Clover "AMD Radeon Pro WX 2100 (polaris12, LLVM 19.1.7, DRM 3.61, 6.12.x)", RX 6400 "(radeonsi, navi24, ...)" (the tokens of the
//     parentheses are the family names of radeonsi: polarisNN, vegaNN, tonga, hawaii, ... are GCN, naviNN and the APUs below are RDNA);
//  4. the marketing name: "Pro WX" (Polaris / Vega) and "RX nnn" (3 digits: RX 4xx / 5xx = Polaris) are GCN, "RX nnnn" (4 digits: 5000 - 9000) RDNA.
// A GPU whose generation cannot be determined is CL11_AMD_UNKNOWN and gets the same (safe) settings as RDNA.
cl11_amd_gen cl11_amd_generation(const cl11_device_info & info) {
    const std::string lname   = cl11_lower(info.name);
    const std::string lvendor = cl11_lower(info.vendor);
    const bool amd = lvendor.find("amd") != std::string::npos || lvendor.find("advanced micro") != std::string::npos ||
                     lname.find("radeon") != std::string::npos;
    if (!amd) {
        return CL11_AMD_NONE;
    }
    if (info.amd_gfx_major > 0) {
        return info.amd_gfx_major >= 10 ? CL11_AMD_RDNA : CL11_AMD_GCN;
    }
    // "gfxNNN": gfx803 / gfx90a (3 characters: the major version is the first digit), gfx1034 (4: the first two), gfx10-3-generic (the digits up to the '-')
    const size_t g = lname.find("gfx");
    if (g != std::string::npos) {
        size_t e = g + 3;
        while (e < lname.size() && std::isalnum((unsigned char) lname[e])) {
            e++;
        }
        const std::string id = lname.substr(g + 3, e - (g + 3));
        int major = 0;
        if (!id.empty() && std::isdigit((unsigned char) id[0])) {
            if (e < lname.size() && lname[e] == '-') {
                major = atoi(id.c_str());                       // gfx9-generic, gfx10-3-generic
            } else if (id.size() == 3) {
                major = id[0] - '0';                            // gfx803, gfx90a
            } else if (id.size() == 4 && std::isdigit((unsigned char) id[1])) {
                major = (id[0] - '0') * 10 + (id[1] - '0');     // gfx1034, gfx1201
            }
        }
        if (major >= 10) {
            return CL11_AMD_RDNA;
        }
        if (major >= 6) {
            return CL11_AMD_GCN;
        }
    }
    // the code name of Mesa: the tokens of the last parentheses of the name
    const size_t lp = lname.rfind('(');
    if (lp != std::string::npos) {
        static const char * const gcn[] = {
            "tahiti", "pitcairn", "verde", "oland", "hainan", "bonaire", "kaveri", "kabini", "hawaii", "mullins", "topaz", "tonga", "fiji",
            "carrizo", "stoney", "polaris10", "polaris11", "polaris12", "vegam", "vega10", "vega12", "vega20", "raven", "raven2", "renoir",
            "arcturus", "aldebaran",
        };
        static const char * const rdna_apu[] = { "vangogh", "rembrandt", "raphael_mendocino", "phoenix", "strix1", "strix_halo", "krackan1" };
        std::string tok;
        const std::string list = lname.substr(lp + 1) + ",";
        for (char ch : list) {
            if (ch != ',') {
                tok += ch;
                continue;
            }
            const size_t b = tok.find_first_not_of(" \t");
            const size_t e = tok.find_last_not_of(" \t");
            const std::string t = b == std::string::npos ? "" : tok.substr(b, e - b + 1);
            tok.clear();
            if (t.empty()) {
                continue;
            }
            for (const char * n : gcn) {
                if (t == n) {
                    return CL11_AMD_GCN;
                }
            }
            if (t.compare(0, 4, "navi") == 0) {
                return CL11_AMD_RDNA;
            }
            for (const char * n : rdna_apu) {
                if (t == n) {
                    return CL11_AMD_RDNA;
                }
            }
        }
    }
    // the marketing name
    if (lname.find(" wx ") != std::string::npos) {
        return CL11_AMD_GCN;
    }
    const size_t rx = lname.find(" rx ");
    if (rx != std::string::npos) {
        size_t digits = 0;
        while (rx + 4 + digits < lname.size() && std::isdigit((unsigned char) lname[rx + 4 + digits])) {
            digits++;
        }
        if (digits == 3) {
            return CL11_AMD_GCN;   // RX 460 .. 590 (Polaris)
        }
        if (digits == 4) {
            return CL11_AMD_RDNA;  // RX 5500 .. 9070
        }
    }
    return CL11_AMD_UNKNOWN;
}

//
// parameterized mat-mat kernels (k_mmx_*, mul_mm_x.cl)
//

struct cl11_mmx_preset {
    const char * name;
    const char * spec;
};

// candidates for the Kepler GPUs (tools/kepler-sweep compares them on the GT 730 / GT 710); "x84" is the 8 x 4 / 64 x 64 / BK 16 tile of the default kernel
static const cl11_mmx_preset cl11_mmx_presets[] = {
    { "x84",   "bm=64,bn=64,rm=8,cn=4,bk=16,pf=0" },
    { "x84p",  "bm=64,bn=64,rm=8,cn=4,bk=16,pf=1" },
    { "x88",   "bm=64,bn=128,rm=8,cn=8,bk=16,pf=1" },
    { "x88w",  "bm=128,bn=64,rm=8,cn=8,bk=16,pf=1" },
    { "x88s",  "bm=64,bn=64,rm=8,cn=8,bk=16,pf=1" },
    { "x88k",  "bm=64,bn=128,rm=8,cn=8,bk=32,pf=1" },
    { "x88d",  "bm=64,bn=128,rm=8,cn=8,bk=16,pf=2" },
    { "x164",  "bm=128,bn=64,rm=16,cn=4,bk=16,pf=1" },
    { "x168",  "bm=128,bn=128,rm=16,cn=8,bk=16,pf=1" },
    { "x168s", "bm=128,bn=64,rm=16,cn=8,bk=16,pf=1" },
    { "x168w", "bm=64,bn=128,rm=16,cn=8,bk=16,pf=1" },
    { "x88g",  "bm=64,bn=128,rm=8,cn=8,bk=16,pf=1,ldg=1" },
};

// spec: a preset name, or "key=value,key=value,..." (keys: bm, bn, rm, cn, bk, ku, pf, ldg, reqd, maxrreg; the missing ones keep the defaults of cl11_mmx),
// a preset name followed by overrides is accepted too ("x88,bk=32"). "base" / "off" / "0" / "" select the default kernels.
bool cl11_mmx_parse(const std::string & spec_in, cl11_mmx & m) {
    m = cl11_mmx();
    std::string spec = spec_in;
    if (spec.empty() || spec == "base" || spec == "off" || spec == "0" || spec == "none") {
        return true;
    }
    std::vector<std::string> parts;
    {
        size_t pos = 0;
        while (pos <= spec.size()) {
            size_t e = spec.find(',', pos);
            if (e == std::string::npos) {
                e = spec.size();
            }
            parts.push_back(spec.substr(pos, e - pos));
            pos = e + 1;
        }
    }
    std::vector<std::string> kv;
    for (const auto & part : parts) {
        if (part.empty()) {
            continue;
        }
        if (part.find('=') == std::string::npos) {
            bool found = false;
            for (const auto & pr : cl11_mmx_presets) {
                if (part == pr.name) {
                    std::string rest = pr.spec;
                    size_t pos = 0;
                    while (pos <= rest.size()) {
                        size_t e = rest.find(',', pos);
                        if (e == std::string::npos) {
                            e = rest.size();
                        }
                        kv.push_back(rest.substr(pos, e - pos));
                        pos = e + 1;
                    }
                    found = true;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "%s: unknown mat-mat variant '%s'\n", __func__, part.c_str());
                return false;
            }
        } else {
            kv.push_back(part);
        }
    }
    for (const auto & item : kv) {
        const size_t eq = item.find('=');
        const std::string key = item.substr(0, eq);
        const int val = atoi(item.c_str() + eq + 1);
        if      (key == "bm")      { m.bm = val; }
        else if (key == "bn")      { m.bn = val; }
        else if (key == "rm")      { m.rm = val; }
        else if (key == "cn")      { m.cn = val; }
        else if (key == "bk")      { m.bk = val; }
        else if (key == "ku")      { m.ku = val; }
        else if (key == "pf")      { m.pf = val; }
        else if (key == "ldg")     { m.ldg = val; }
        else if (key == "reqd")    { m.reqd = val; }
        else if (key == "maxrreg") { m.maxrreg = val; }
        else {
            fprintf(stderr, "%s: unknown key '%s' in the mat-mat variant '%s'\n", __func__, key.c_str(), spec.c_str());
            return false;
        }
    }
    m.on = true;
    return true;
}

// the combination must be one the kernel supports; `local_mem` and `max_wg` are the device limits
bool cl11_mmx_valid(const cl11_mmx & m, size_t local_mem, size_t max_wg, std::string & why) {
    char buf[160];
    if (m.rm != 8 && m.rm != 16) { why = "rm must be 8 or 16"; return false; }
    if (m.cn != 4 && m.cn != 8)  { why = "cn must be 4 or 8";  return false; }
    if (m.bk != 16 && m.bk != 32) { why = "bk must be 16 or 32"; return false; }
    if (m.ku != 1 && m.ku != 2 && m.ku != 4 && m.ku != 8 && m.ku != 16) { why = "ku must be 1, 2, 4, 8 or 16"; return false; }
    if (m.ku > m.bk) { why = "ku must not exceed bk"; return false; }
    if (m.pf < 0 || m.pf > 2) { why = "pf must be 0, 1 or 2"; return false; }
    if (m.bm <= 0 || m.bn <= 0 || m.bm % m.rm != 0 || m.bn % m.cn != 0) { why = "bm must be a multiple of rm and bn of cn"; return false; }
    if (m.bm % 4 != 0 || m.bn % 4 != 0) { why = "bm and bn must be multiples of 4"; return false; }
    const int nt = m.nt();
    if (nt < 32 || nt % 32 != 0 || (size_t) nt > max_wg) {
        snprintf(buf, sizeof(buf), "work-group of %d work-items (a multiple of 32, at most %zu)", nt, max_wg);
        why = buf;
        return false;
    }
    const size_t bufsz = (size_t) m.bk * ((size_t) (m.bm + 4) + (size_t) (m.bn + 4)) * sizeof(float) * (m.pf == 2 ? 2 : 1);
    if (bufsz > local_mem) {
        snprintf(buf, sizeof(buf), "%zu bytes of local memory (device: %zu)", bufsz, local_mem);
        why = buf;
        return false;
    }
    return true;
}

cl11_tuning cl11_tuning_for(const cl11_device_info & info) {
    cl11_tuning t;
    const std::string lname = cl11_lower(info.name);
    // GGML_CL11_FORCE_GEN=kepler|fermi: apply the values of that branch to any device (testing the tuning on a machine without the GPU; the
    // NVIDIA-only parts (inline PTX, ld.global.nc, -cl-nv-maxrregcount) stay off where the platform is not NVIDIA)
    const char * force_gen = getenv("GGML_CL11_FORCE_GEN");
    const bool force_fermi  = force_gen != nullptr && cl11_lower(force_gen) == "fermi";
    const bool force_kepler = force_gen != nullptr && cl11_lower(force_gen) == "kepler";
    const bool fermi = force_fermi || (!force_kepler && (info.nv_cc_major == 2 || (info.nv_cc_major == 0 && lname.find("gt 430") != std::string::npos)));
    if (fermi) {
        // GeForce GT 430 (GF108: 2 SMs, 96 cores, 1.4 GHz, ~22 GB/s, 32768 registers per SM)
        t.name = "Fermi (sm_2x)";
        t.cfg.wg_mv    = 128;
        t.cfg.mvs_nrw  = 2;
        for (int k = 0; k < CL11_RK_COUNT; ++k) {
            t.cfg.wg_mvr[k]  = 64;   // 2 warps share the rows of a work-group (tg of Qwen3-0.6B Q8_0 / Qwen2-0.5B-ft Q4_0: 128: 20.4 / 28.6, 64: 21.2 / 29.0, 256: 18.3 / 25.2)
            t.cfg.mvr_nrr[k] = 2;    // 2 rows per lane: halves the loads of src1 (4 rows per lane: fewer warps, slower)
        }
        t.cfg.mm_bn    = 8;    // 64 x 128 tiles: src0 is read once for the 128 columns of a prompt batch (fewer re-reads of src1 than 64 x 64)
        // K-quants: the kernels need more registers than the 63 of a Fermi work-item (spills): one row per lane (m4096 k14336: Q4_K 4.4 -> 8.7 GB/s, Q6_K 5.2 -> 14.4)
        for (int k = CL11_RK_Q4_K; k < CL11_RK_COUNT; ++k) {
            t.cfg.wg_mvr[k]  = 128;
            t.cfg.mvr_nrr[k] = 1;
        }
        t.cfg.mvr_dn[CL11_RK_Q4_0] = 1;  // the nibbles as denormal floats (FFMA of Fermi handles denormals at full speed): 1 AND + 1 FFMA per weight instead of a PRMT (a 16-per-clock instruction on Fermi) + FSUB + FFMA
        t.cfg.mvr_dn[CL11_RK_Q4_K] = 1;  // GT 430, m4096 k14336: Q4_0 12.5 -> 17.4 GB/s, Q4_K 10.3 -> 12.5; tg Q4_0 model 30.0 -> 36.5, Q4_K_M 21.2 -> 23.2 t/s
        // Q6_K: no DN (14.4 -> 14.0 GB/s at m4096 k14336, 11.6 -> 10.8 at m3072 k1024: the extra instructions of the 6-bit unpacking are not paid back)
        t.mvs_min_k    = 256;
        t.mvs_family   = 0;
        t.mvr_const_x  = true;   // Fermi: +15% (Q8_0) .. +30% (Q4_0), the x loads of 2 rows compete with the weights for the load units
    }
    const bool kepler = force_kepler || (!force_fermi && (info.nv_cc_major == 3 || (info.nv_cc_major == 0 && (lname.find("gt 730") != std::string::npos || lname.find("gt 710") != std::string::npos))));
    if (kepler) {
        // GeForce GT 730 (GK208: 2 SMX, 384 cores, 64-bit GDDR5 ~38 GB/s) / GT 710 (GK208: 1 SMX, 192 cores, 64-bit DDR3 ~11.7 GB/s)
        t.name = "Kepler (sm_3x)";
        t.mvr_const_x = false;   // NVIDIA 390 on Kepler ignores the offset of a sub-buffer passed as __constant (reads the base of the parent): wrong results
        // mat-vec of the repacked weights (mat-vec m4096 k14336 and tg of the models, GT 730 / GT 710):
        //   Q4_0: 4 warps per group, 2 rows per lane; Q8_0: the same, 1 row per lane on the GT 710 (1 SMX: 32 groups of 4 warps instead of 16);
        //   Q4_K / Q6_K: 2 warps, 2 rows per lane, src1 through ld.global.nc and a program with at most 128 registers (see cl11_ensure_kprogram)
        for (int k = 0; k < CL11_RK_COUNT; ++k) {
            t.cfg.wg_mvr[k]  = 128;
            t.cfg.mvr_nrr[k] = 2;
        }
        t.cfg.mvr_nrr[CL11_RK_Q8_0] = info.compute_units == 1 ? 1 : 2;
        t.cfg.mm_bn = 4;         // mat-mat: 64 x 64 tiles (128 work-items): +18% (GT 730) / +22% (GT 710) for the quantized types over 64 x 128
        t.cfg.wg_mvr[CL11_RK_Q4_K] = 64;
        t.cfg.wg_mvr[CL11_RK_Q6_K] = 64;
        t.cfg.mvr_ldg[CL11_RK_Q4_K] = 1;
        t.cfg.mvr_ldg[CL11_RK_Q6_K] = 1;
        t.fuse_set_rows = true;  // the K and V cache writes of a layer in one launch (28 launches less per token of Qwen3-0.6B)
        t.multi_mv = true;       // Q / K / V and FFN gate / up in one launch each (decode launches of Qwen3-0.6B 564 -> 452 with both fusions):
                                 //   tg GT 730 Q8_0 33.6 -> 34.9, Q4_K_M 35.4 -> 39.2; GT 710 Q8_0 14.6 -> 14.7, Q4_K_M 16.7 -> 17.8 t/s
        t.mmx_variant = "x84";   // mat-mat k_mmx_* (mul_mm_x.cl) with the 8 x 4 / 64 x 64 / BK 16 tile: pp128 GT 730 Q8_0 215.8 -> 241.9, Q4_K_M 208.0 -> 231.5;
                                 //   GT 710 110.7 -> 120.5, 107.9 -> 116.4 t/s. The larger register tiles (8 x 8, 16 x 4, 16 x 8) were all slower (tools/kepler-sweep)
        t.cfg.nv_maxrreg = 128;  // Q4_K 56% -> 67%, Q6_K 74% -> 75% of the streaming rate on a GT 730; Q4_0 / Q8_0 get slower with the cap (hence the second program)
    }
    // AMD (Mesa rusticl / Clover, or the AMD drivers). The generation decides what is applied, see cl11_amd_generation():
    //  - every AMD GPU: use_mad, strict_alloc, mvt_groups and the K-quant mat-vec (measured on the WX 2100 and the RX 6400);
    //  - GCN (Polaris 12 = Radeon Pro WX 2100 was measured): in addition the mat-vec of Q4_0 / Q8_0 and the mat-mat tile below;
    //  - RDNA (RX 6400 = navi24 was measured) and AMD GPUs of an unknown generation: nothing else (the Vulkan backend is recommended there and
    //    no further tuning of CL11 is needed, see docs/HARDWARE.md); everything else keeps the defaults.
    const cl11_amd_gen amd = (!fermi && info.nv_cc_major == 0) ? cl11_amd_generation(info) : CL11_AMD_NONE;
    if (amd != CL11_AMD_NONE) {
        t.cfg.use_mad = true;  // rusticl emulates fma() in software on GCN (mat-mat 3.5 -> 450+ GFLOPS with mad()); mad() is never slower
        t.strict_alloc = true; // Mesa (Clover, rusticl) refuses a buffer above CL_DEVICE_MAX_MEM_ALLOC_SIZE: CL_INVALID_BUFFER_SIZE
        t.name = amd == CL11_AMD_GCN ? "AMD (GCN)" : amd == CL11_AMD_RDNA ? "AMD (RDNA)" : "AMD (unknown generation)";
        t.cfg.mvt_groups = true; // Mesa limits the work-group to 256: the single work-group V cache kernel is 8x slower than the many-groups one (WX 2100: 307 vs 38 us per call)
        // K-quants (Q4_K / Q6_K mat-vec): 4 warps and 1 row per lane instead of the defaults (2 warps, 2 rows). tg of Qwen3-0.6B Q4_K_M, default -> 128 x 1:
        // WX 2100 Clover 16.6 -> 19.6 t/s, WX 2100 rusticl 14.7 -> 16.2, RX 6400 rusticl 20.5 -> 24.9 (64 x 1 and 256 x 1 are slower on the WX 2100, 2 rows per lane slower everywhere)
        for (int k = CL11_RK_Q4_K; k < CL11_RK_COUNT; ++k) {
            t.cfg.wg_mvr[k]  = 128;
            t.cfg.mvr_nrr[k] = 1;
        }
    }
    if (amd == CL11_AMD_GCN) {
        // Polaris 12: Radeon Pro WX 2100, 8 CUs, wave64, ~44 GB/s. Mat-vec of Q4_0 / Q8_0: 8 warps per group, 1 row per lane: 96 groups for a
        // 3072-row matrix (8 CUs); 64 x 2 rows: 48 groups, 21 -> 35 GB/s (tg of Qwen3-0.6B Q8_0 / Qwen2-0.5B-ft Q4_0 on Clover: 64x2: 18.0 / 21.2,
        // 64x1: 19.5 / 23.0, 128x1: 19.7 / 25.4, 256x1: 20.2 / 25.6 t/s). The K-quants are not tuned for AMD yet (defaults).
        t.cfg.wg_mvr[CL11_RK_Q4_0]  = 256;
        t.cfg.wg_mvr[CL11_RK_Q8_0]  = 256;
        t.cfg.mvr_nrr[CL11_RK_Q4_0] = 1;
        t.cfg.mvr_nrr[CL11_RK_Q8_0] = 1;
        t.cfg.mm_bn    = 4;    // 64 x 64 tiles (128 work-items of 8 x 4 values): 530-620 GFLOPS (Q8_0) instead of 345-410 with 64 x 128
    }
    // Mesa Clover (OpenCL 1.1, any GPU): the cost of a launch grows with the size of the program that holds the kernel
    if (cl11_lower(info.platform_name).find("clover") != std::string::npos) {
        t.split_programs = true;
    }
    // everything else: not tuned yet, same values as the defaults
    return t;
}

// GGML_CL11_DEVICES (or its alias SCRAPGPU_OPENCL11_DEVICES, used when the first is not set):
// comma separated list of substrings of the device names (case insensitive)
// true when `name` contains one of the comma separated substrings of the environment variable `var` (or of its alias),
// or when neither is set
static bool cl11_env_list_selected(const char * var, const char * alias, const std::string & name) {
    const char * env = getenv(var);
    if (env == nullptr) {
        env = getenv(alias);
    }
    if (env == nullptr || *env == '\0') {
        return true;
    }
    const std::string lname = cl11_lower(name);
    std::string item;
    std::string list = std::string(env) + ",";
    for (char c : list) {
        if (c == ',') {
            // trim
            size_t b = item.find_first_not_of(" \t");
            size_t e = item.find_last_not_of(" \t");
            if (b != std::string::npos) {
                const std::string pat = cl11_lower(item.substr(b, e - b + 1));
                if (lname.find(pat) != std::string::npos) {
                    return true;
                }
            }
            item.clear();
        } else {
            item += c;
        }
    }
    return false;
}

bool cl11_device_selected(const std::string & name) {
    return cl11_env_list_selected("GGML_CL11_DEVICES", "SCRAPGPU_OPENCL11_DEVICES", name);
}

const char * cl11_device_rejection_reason(const cl11_device_info & info) {
    // This exact combination hangs even on the second trivial kernel launch
    // on TB250 (docs/HARDWARE.md). Reject before creating a context; rusticl
    // and Vulkan remain available. Do not generalize to unmeasured GPUs.
    if (info.platform_name.find("Clover") != std::string::npos &&
        (info.name.find("RX 6400") != std::string::npos || info.name.find("navi24") != std::string::npos)) {
        return "RX 6400 / Mesa Clover is disabled after reproduced GPU hangs; use rusticl or Vulkan";
    }
    return nullptr;
}

// GGML_CL11_PLATFORMS (alias SCRAPGPU_OPENCL11_PLATFORMS): comma separated substrings of the OpenCL platform names to use, e.g.
// "NVIDIA,rusticl". A Radeon is visible on Mesa Clover and rusticl at the same time; Clover hangs the RX 6400 (docs/HARDWARE.md).
bool cl11_platform_selected(const std::string & name) {
    return cl11_env_list_selected("GGML_CL11_PLATFORMS", "SCRAPGPU_OPENCL11_PLATFORMS", name);
}

bool cl11_probe_device_infos(void (*emit)(const cl11_device_info & info, bool strict_alloc)) {
    cl_uint n_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &n_platforms);
    if (err != CL_SUCCESS || n_platforms == 0) {
        if (cl11_backend_debug()) {
            fprintf(stderr, "%s: no OpenCL platform (%s)\n", __func__, cl11_err_str(err));
        }
        return true;
    }
    std::vector<cl_platform_id> platforms(n_platforms);
    if (clGetPlatformIDs(n_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }

    for (cl_platform_id platform : platforms) {
        cl_uint n_devices = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devices) != CL_SUCCESS || n_devices == 0) {
            continue;
        }
        std::vector<cl_device_id> devices(n_devices);
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, n_devices, devices.data(), nullptr) != CL_SUCCESS) {
            continue;
        }

        char pname[256] = {};
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(pname) - 1, pname, nullptr);
        if (!cl11_platform_selected(pname)) {
            if (cl11_backend_debug()) {
                fprintf(stderr, "%s: platform '%s' not selected by GGML_CL11_PLATFORMS\n", __func__, pname);
            }
            continue;
        }

        for (cl_device_id device : devices) {
            cl11_device_info info;
            info.platform      = platform;
            info.device        = device;
            info.name          = cl11_device_str(device, CL_DEVICE_NAME);
            info.platform_name = pname;
            info.vendor        = cl11_device_str(device, CL_DEVICE_VENDOR);
            info.device_version = cl11_device_str(device, CL_DEVICE_VERSION);
            info.c_version     = cl11_device_str(device, CL_DEVICE_OPENCL_C_VERSION);
            info.max_alloc     = (size_t) cl11_device_val<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            info.global_mem    = (size_t) cl11_device_val<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            info.local_mem     = (size_t) cl11_device_val<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
            info.max_wg        = cl11_device_val<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            info.align_bytes   = cl11_device_val<cl_uint>(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN) / 8;
            info.compute_units = (int) cl11_device_val<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
            if (cl11_device_str(device, CL_DEVICE_EXTENSIONS).find("cl_nv_device_attribute_query") != std::string::npos) {
                // CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV / MINOR_NV
                info.nv_cc_major = (int) cl11_device_val<cl_uint>(device, (cl_device_info) 0x4000);
                info.nv_cc_minor = (int) cl11_device_val<cl_uint>(device, (cl_device_info) 0x4001);
            }
            if (cl11_device_str(device, CL_DEVICE_EXTENSIONS).find("cl_amd_device_attribute_query") != std::string::npos) {
                info.amd_gfx_major = (int) cl11_device_val<cl_uint>(device, (cl_device_info) 0x404A); // CL_DEVICE_GFXIP_MAJOR_AMD
            }

            if (const char * env = getenv("GGML_CL11_FAKE_MAX_ALLOC")) {
                // debugging: pretend that CL_DEVICE_MAX_MEM_ALLOC_SIZE is this many bytes (e.g. 252706816 for a GT 430)
                info.max_alloc = std::min(info.max_alloc, (size_t) strtoull(env, nullptr, 10));
            }

            size_t item_sizes[3] = {0, 0, 0};
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(item_sizes), item_sizes, nullptr);
            info.max_item0 = item_sizes[0];

            const bool available = cl11_device_val<cl_bool>(device, CL_DEVICE_AVAILABLE, CL_FALSE);
            const bool compiler  = cl11_device_val<cl_bool>(device, CL_DEVICE_COMPILER_AVAILABLE, CL_FALSE);
            const bool little    = cl11_device_val<cl_bool>(device, CL_DEVICE_ENDIAN_LITTLE, CL_FALSE);

            // the smallest configuration needs 64 work-items per group and 16 KiB of local memory
            if (!available || !compiler || !little || info.local_mem < 16 * 1024 ||
                info.max_wg < 64 || info.max_item0 < 64 || info.max_alloc < 1024 * 1024) {
                if (cl11_backend_debug()) {
                    fprintf(stderr, "%s: skipping device '%s' (available=%d compiler=%d little=%d local_mem=%zu max_wg=%zu max_alloc=%zu)\n",
                            __func__, info.name.c_str(), available, compiler, little, info.local_mem, info.max_wg, info.max_alloc);
                }
                continue;
            }
            if (!cl11_device_selected(info.name)) {
                if (cl11_backend_debug()) {
                    fprintf(stderr, "%s: device '%s' not selected by GGML_CL11_DEVICES\n", __func__, info.name.c_str());
                }
                continue;
            }

            if (const char * reason = cl11_device_rejection_reason(info)) {
                fprintf(stderr, "%s: skipping '%s': %s\n", __func__, info.name.c_str(), reason);
                continue;
            }
            emit(info, cl11_tuning_for(info).strict_alloc);
        }
    }
    return true;
}
