// Cross-check scrapgpu_ref_execute against ggml's CPU backend (one graph node per case).
//
// Build:
//   c++ -std=c++17 -O1 -I/Users/eightman/dev/sandbox/llama.cpp-b11384/ggml/include -Iinclude \
//     adapters/ggml/tools/ref_check.cpp src/core/*.cpp src/core/ref/*.cpp -Isrc/core/ref \
//     -L/Users/eightman/dev/sandbox/llama.cpp-b11384/build-dl/bin -lggml -lggml-base \
//     -Wl,-rpath,/Users/eightman/dev/sandbox/llama.cpp-b11384/build-dl/bin -o _w/ref_check
// Run:
//   GGML_LIB_DIR=/Users/eightman/dev/sandbox/llama.cpp-b11384/build-dl/bin ./_w/ref_check
//
// NMSE threshold 1e-7 for float ops; 5e-4 when src0 of MUL_MAT is quantized (ggml's CPU path
// quantizes src1 to Q8_0 for the dot product, so results differ slightly from a pure dequant*F32 ref).

#include "ggml.h"
#include "ggml-backend.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/status.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static constexpr double kNmseFloat = 1e-7;
// Relaxed NMSE when ggml MUL_MAT uses Q8_0 activations against quantized weights.
static constexpr double kNmseQuantMat = 5e-4;

static int g_cases = 0;
static int g_ok = 0;
static int g_fail = 0;

static uint32_t g_rng = 0xC0FFEEu;

static float frand() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return (float) (g_rng >> 8) * (1.0f / 16777216.0f) - 0.5f;
}

static double nmse_f32(const float * a, const float * b, size_t n) {
    double num = 0.0;
    double den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    if (den <= 1e-30) {
        return num;
    }
    return num / den;
}

static scrapgpu_dtype dtype_from_ggml(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32: return SCRAPGPU_DTYPE_F32;
        case GGML_TYPE_F16: return SCRAPGPU_DTYPE_F16;
        case GGML_TYPE_I32: return SCRAPGPU_DTYPE_I32;
        case GGML_TYPE_I64: return SCRAPGPU_DTYPE_I64;
        case GGML_TYPE_Q4_0: return SCRAPGPU_DTYPE_Q4_0;
        case GGML_TYPE_Q8_0: return SCRAPGPU_DTYPE_Q8_0;
        case GGML_TYPE_Q4_K: return SCRAPGPU_DTYPE_Q4_K;
        case GGML_TYPE_Q6_K: return SCRAPGPU_DTYPE_Q6_K;
        default: return SCRAPGPU_DTYPE_F32;
    }
}

static scrapgpu_tensor_desc desc_from(const ggml_tensor * t) {
    scrapgpu_tensor_desc d{};
    d.struct_size = sizeof(d);
    d.dtype = dtype_from_ggml(t->type);
    for (int i = 0; i < 4; ++i) {
        d.ne[i] = t->ne[i];
        d.nb[i] = t->nb[i];
    }
    d.layout = SCRAPGPU_LAYOUT_NATURAL;
    return d;
}

struct RunCtx {
    ggml_backend_t cpu = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;

    ~RunCtx() {
        if (buf) {
            ggml_backend_buffer_free(buf);
        }
        if (ctx) {
            ggml_free(ctx);
        }
    }

    bool init(size_t mem = 1u << 26) {
        ggml_init_params ip{};
        ip.mem_size = mem;
        ip.no_alloc = true;
        ctx = ggml_init(ip);
        return ctx != nullptr;
    }

    bool alloc() {
        buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        return buf != nullptr;
    }

    bool compute(ggml_tensor * root) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, root);
        return ggml_backend_graph_compute(cpu, gf) == GGML_STATUS_SUCCESS;
    }

    void fill_f32(ggml_tensor * t, float lo = -1.5f, float hi = 1.5f) {
        const size_t n = ggml_nelements(t);
        std::vector<float> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = lo + (hi - lo) * (frand() + 0.5f);
        }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }

    void fill_i32(ggml_tensor * t, int32_t lo, int32_t hi) {
        const size_t n = ggml_nelements(t);
        std::vector<int32_t> v(n);
        for (size_t i = 0; i < n; ++i) {
            v[i] = lo + (int32_t) ((hi - lo + 1) * (size_t) (g_rng++ % 1000) / 1000);
        }
        ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
    }
};

static void report(const char * op, const std::string & case_desc, double err, double thr, bool ok) {
    ++g_cases;
    if (ok) {
        ++g_ok;
    } else {
        ++g_fail;
    }
    std::printf("%s %s: nmse %g %s\n", op, case_desc.c_str(), err, ok ? "OK" : "FAIL");
    std::fflush(stdout);
}

static void report_exact(const char * op, const std::string & case_desc, bool ok) {
    ++g_cases;
    if (ok) {
        ++g_ok;
    } else {
        ++g_fail;
    }
    std::printf("%s %s: exact %s\n", op, case_desc.c_str(), ok ? "OK" : "FAIL");
    std::fflush(stdout);
}

static scrapgpu_status ref_exec(scrapgpu_op op,
                                const scrapgpu_tensor_desc * ins,
                                size_t n_in,
                                void ** in_ptr,
                                scrapgpu_tensor_desc * outs,
                                size_t n_out,
                                void ** out_ptr,
                                const void * params,
                                size_t params_size) {
    scrapgpu_op_desc d{};
    d.struct_size = sizeof(d);
    d.op = op;
    d.inputs = ins;
    d.input_count = n_in;
    d.outputs = outs;
    d.output_count = n_out;
    d.params = params;
    d.params_size = params_size;
    return scrapgpu_ref_execute(&d, in_ptr, out_ptr);
}

static std::vector<float> tensor_to_f32(const ggml_tensor * t) {
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
    std::vector<float> out;
    out.reserve((size_t) ggml_nelements(t));
    const int64_t bs = ggml_blck_size(t->type);
    const auto * tr = ggml_get_type_traits(t->type);
    std::vector<float> block(bs);
    for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < t->ne[0]; i0 += bs) {
                    const size_t off = (size_t) (i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + (i0 / bs) * t->nb[0]);
                    if (t->type == GGML_TYPE_F32) {
                        float v;
                        std::memcpy(&v, raw.data() + off, sizeof(v));
                        out.push_back(v);
                    } else if (t->type == GGML_TYPE_F16) {
                        ggml_fp16_t h;
                        std::memcpy(&h, raw.data() + off, sizeof(h));
                        out.push_back(ggml_fp16_to_fp32(h));
                    } else if (ggml_is_quantized(t->type)) {
                        tr->to_float(raw.data() + off, block.data(), (int) bs);
                        for (int64_t j = 0; j < bs; ++j) {
                            out.push_back(block[(size_t) j]);
                        }
                    } else {
                        out.push_back(0.0f);
                    }
                }
            }
        }
    }
    return out;
}

static bool bytes_equal(const ggml_tensor * a, const std::vector<uint8_t> & b) {
    std::vector<uint8_t> ra(ggml_nbytes(a));
    ggml_backend_tensor_get(a, ra.data(), 0, ra.size());
    return ra == b;
}

static bool valid_argsort_row(const float * keys, int64_t n, const int32_t * idx, bool asc) {
    for (int64_t i = 0; i + 1 < n; ++i) {
        const float va = keys[idx[i]];
        const float vb = keys[idx[i + 1]];
        if (asc) {
            if (va > vb) {
                return false;
            }
            if (va == vb && idx[i] > idx[i + 1]) {
                return false;
            }
        } else {
            if (va < vb) {
                return false;
            }
            if (va == vb && idx[i] > idx[i + 1]) {
                return false;
            }
        }
    }
    return true;
}

static bool indices_match_sort(const ggml_tensor * keys, const ggml_tensor * gg_idx, const int32_t * ref_idx, bool asc) {
    const auto kf = tensor_to_f32(keys);
    const int64_t ne0 = keys->ne[0];
    const int64_t rows = ggml_nrows(keys);
    std::vector<int32_t> gg((size_t) ggml_nelements(gg_idx));
    ggml_backend_tensor_get(gg_idx, gg.data(), 0, ggml_nbytes(gg_idx));
    for (int64_t r = 0; r < rows; ++r) {
        const float * row = kf.data() + (size_t) r * (size_t) ne0;
        const int32_t * gr = gg.data() + (size_t) r * (size_t) ne0;
        const int32_t * rr = ref_idx + (size_t) r * (size_t) ne0;
        if (!valid_argsort_row(row, ne0, gr, asc) || !valid_argsort_row(row, ne0, rr, asc)) {
            return false;
        }
        for (int64_t i = 0; i < ne0; ++i) {
            if (row[gr[i]] != row[rr[i]]) {
                return false;
            }
        }
    }
    return true;
}

static bool top_k_match(const ggml_tensor * keys, const ggml_tensor * gg_idx, const int32_t * ref_idx, int k, bool asc) {
    const auto kf = tensor_to_f32(keys);
    const int64_t ne0 = keys->ne[0];
    const int64_t rows = ggml_nrows(keys);
    std::vector<int32_t> a((size_t) ggml_nelements(gg_idx));
    ggml_backend_tensor_get(gg_idx, a.data(), 0, ggml_nbytes(gg_idx));
    for (int64_t r = 0; r < rows; ++r) {
        const float * row = kf.data() + (size_t) r * (size_t) ne0;
        std::vector<float> ref_vals;
        std::vector<float> gg_vals;
        for (int i = 0; i < k; ++i) {
            gg_vals.push_back(row[a[(size_t) r * (size_t) k + (size_t) i]]);
            ref_vals.push_back(row[ref_idx[(size_t) r * (size_t) k + (size_t) i]]);
        }
        if (asc) {
            std::sort(ref_vals.begin(), ref_vals.end());
            std::sort(gg_vals.begin(), gg_vals.end());
        } else {
            std::sort(ref_vals.begin(), ref_vals.end(), std::greater<float>());
            std::sort(gg_vals.begin(), gg_vals.end(), std::greater<float>());
        }
        if (ref_vals != gg_vals) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// Op families

static void check_binary(RunCtx & rc, const char * name, scrapgpu_op sop,
                         ggml_tensor * (*ggml_fn)(ggml_context *, ggml_tensor *, ggml_tensor *),
                         int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
                         int64_t b0, int64_t b1, int64_t b2, int64_t b3) {
    ggml_tensor * a = ggml_new_tensor_4d(rc.ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_tensor * b = ggml_new_tensor_4d(rc.ctx, GGML_TYPE_F32, b0, b1, b2, b3);
    ggml_tensor * out = ggml_fn(rc.ctx, a, b);
    if (!rc.alloc()) {
        return;
    }
    rc.fill_f32(a);
    rc.fill_f32(b);
    if (!rc.compute(out)) {
        report(name, "compute", 1.0, 0.0, false);
        return;
    }
    std::vector<uint8_t> ref_buf(ggml_nbytes(out));
    std::memcpy(ref_buf.data(), ggml_get_data(out), ref_buf.size());

    scrapgpu_tensor_desc da = desc_from(a);
    scrapgpu_tensor_desc db = desc_from(b);
    scrapgpu_tensor_desc dout = desc_from(out);
    const scrapgpu_tensor_desc ins[2] = {da, db};
    scrapgpu_tensor_desc outs_d[1] = {dout};
    void * in_ptr[2] = {ggml_get_data(a), ggml_get_data(b)};
    void * out_ptr[1] = {ref_buf.data()};
    if (ref_exec(sop, ins, 2, in_ptr, outs_d, 1, out_ptr, nullptr, 0) != SCRAPGPU_STATUS_OK) {
        report(name, "ref status", 1.0, 0.0, false);
        return;
    }
    const auto gg = tensor_to_f32(out);
    std::vector<float> rf(gg.size());
    std::memcpy(rf.data(), ref_buf.data(), std::min(ref_buf.size(), rf.size() * sizeof(float)));
    const double e = nmse_f32(rf.data(), gg.data(), gg.size());
    char buf[128];
    std::snprintf(buf, sizeof(buf), "shape %lldx%lldx%lldx%lld bc %lldx%lldx%lldx%lld",
                  (long long) ne0, (long long) ne1, (long long) ne2, (long long) ne3,
                  (long long) b0, (long long) b1, (long long) b2, (long long) b3);
    report(name, buf, e, kNmseFloat, e <= kNmseFloat);
}

static ggml_tensor * ggml_scale_test(ggml_context * c, ggml_tensor * a) {
    return ggml_scale_bias(c, a, 1.75f, 0.25f);
}

static ggml_tensor * ggml_clamp_test(ggml_context * c, ggml_tensor * a) {
    return ggml_clamp(c, a, -0.5f, 1.25f);
}

static void check_unary_f32(RunCtx & rc, scrapgpu_op sop, const void * params, size_t psz,
                            ggml_tensor * (*ggml_fn)(ggml_context *, ggml_tensor *),
                            const char * tag, int64_t n) {
    ggml_tensor * a = ggml_new_tensor_1d(rc.ctx, GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_fn(rc.ctx, a);
    if (!rc.alloc()) {
        return;
    }
    rc.fill_f32(a, -2.0f, 2.0f);
    if (!rc.compute(out)) {
        report(tag, "compute", 1.0, 0.0, false);
        return;
    }
    std::vector<uint8_t> ref(ggml_nbytes(out));
    scrapgpu_tensor_desc sa = desc_from(a);
    scrapgpu_tensor_desc so = desc_from(out);
    void * ip[1] = {ggml_get_data(a)};
    void * op[1] = {ref.data()};
    if (ref_exec(sop, &sa, 1, ip, &so, 1, op, params, psz) != SCRAPGPU_STATUS_OK) {
        report(tag, "ref", 1.0, 0.0, false);
        return;
    }
    const auto gg = tensor_to_f32(out);
    std::vector<float> rf(gg.size());
    std::memcpy(rf.data(), ref.data(), ref.size());
    const double e = nmse_f32(rf.data(), gg.data(), gg.size());
    report(tag, std::to_string(n), e, kNmseFloat, e <= kNmseFloat);
}

static void check_mul_mat_case(ggml_backend_t cpu, enum ggml_type wt, int K, int M, int N, int ne02, int ne12, bool nc_src1) {
    RunCtx rc;
    rc.cpu = cpu;
    if (!rc.init()) {
        return;
    }
    ggml_tensor * x_base = ggml_new_tensor_3d(rc.ctx, GGML_TYPE_F32, nc_src1 ? K + 3 : K, N, ne12);
    ggml_tensor * w = ggml_new_tensor_3d(rc.ctx, wt, K, M, ne02);
    ggml_tensor * x = nc_src1
                          ? ggml_view_3d(rc.ctx, x_base, K, N, ne12, (K + 3) * sizeof(float), x_base->nb[2], 0)
                          : x_base;
    ggml_tensor * out = ggml_mul_mat(rc.ctx, w, x);
    if (!rc.alloc()) {
        return;
    }
    std::vector<float> wf((size_t) K * M * ne02);
    for (auto & v : wf) {
        v = frand() * 3.0f;
    }
    std::vector<uint8_t> wq(ggml_nbytes(w));
    ggml_quantize_chunk(wt, wf.data(), wq.data(), 0, (int) (M * ne02), K, nullptr);
    ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    rc.fill_f32(x);
    if (!rc.compute(out)) {
        report("MUL_MAT", "compute", 1.0, 0.0, false);
        return;
    }
    std::vector<uint8_t> ref(ggml_nbytes(out));
    scrapgpu_tensor_desc dw = desc_from(w);
    scrapgpu_tensor_desc dx = desc_from(x);
    scrapgpu_tensor_desc dout = desc_from(out);
    const scrapgpu_tensor_desc ins[2] = {dw, dx};
    scrapgpu_mul_mat_params mp{sizeof(mp), 1};
    void * in_ptr[2] = {ggml_get_data(w), ggml_get_data(x)};
    void * out_ptr[1] = {ref.data()};
    if (ref_exec(SCRAPGPU_OP_MUL_MAT, ins, 2, in_ptr, &dout, 1, out_ptr, &mp, sizeof(mp)) != SCRAPGPU_STATUS_OK) {
        char ebuf[128];
        std::snprintf(ebuf, sizeof(ebuf), "ref error K=%d (%s)", K, scrapgpu_last_error());
        report("MUL_MAT", ebuf, 1.0, 0.0, false);
        return;
    }
    const auto gg = tensor_to_f32(out);
    std::vector<float> rf(gg.size());
    std::memcpy(rf.data(), ref.data(), ref.size());
    double thr = kNmseFloat;
    if (ggml_is_quantized(wt)) {
        thr = kNmseQuantMat;
    } else if (wt == GGML_TYPE_F16) {
        thr = 5e-7;
    }
    const double e = nmse_f32(rf.data(), gg.data(), gg.size());
    char buf[160];
    std::snprintf(buf, sizeof(buf), "w=%d K=%d M=%d N=%d bc=%d,%d nc1=%d", (int) wt, K, M, N, ne02, ne12, (int) nc_src1);
    const bool ok = std::isfinite(e) && e <= thr;
    report("MUL_MAT", buf, ok ? e : 1.0, thr, ok);
}

// ---------------------------------------------------------------------------------------------
// Test drivers

static void test_binary_ops(RunCtx & rc) {
    struct Case {
        int64_t a[4];
        int64_t b[4];
    };
    const Case cases[] = {
        {{5, 3, 1, 1}, {5, 1, 1, 1}},
        {{7, 2, 2, 1}, {7, 2, 2, 1}},
        {{4, 3, 2, 1}, {1, 1, 1, 1}},
        {{3, 2, 1, 1}, {3, 2, 1, 1}},
    };
    for (const auto & c : cases) {
        RunCtx r2;
        r2.cpu = rc.cpu;
        r2.init();
        check_binary(r2, "ADD", SCRAPGPU_OP_ADD, ggml_add, c.a[0], c.a[1], c.a[2], c.a[3], c.b[0], c.b[1], c.b[2], c.b[3]);
        RunCtx r3;
        r3.cpu = rc.cpu;
        r3.init();
        check_binary(r3, "SUB", SCRAPGPU_OP_SUB, ggml_sub, c.a[0], c.a[1], c.a[2], c.a[3], c.b[0], c.b[1], c.b[2], c.b[3]);
        RunCtx r4;
        r4.cpu = rc.cpu;
        r4.init();
        check_binary(r4, "MUL", SCRAPGPU_OP_MUL, ggml_mul, c.a[0], c.a[1], c.a[2], c.a[3], c.b[0], c.b[1], c.b[2], c.b[3]);
        RunCtx r5;
        r5.cpu = rc.cpu;
        r5.init();
        check_binary(r5, "DIV", SCRAPGPU_OP_DIV, ggml_div, c.a[0], c.a[1], c.a[2], c.a[3], c.b[0], c.b[1], c.b[2], c.b[3]);
    }
}

static void test_unary_family(RunCtx & rc) {
    const int sizes[] = {5, 11, 17};
    for (int n : sizes) {
        RunCtx r;
        r.cpu = rc.cpu;
        r.init();
        scrapgpu_scale_params sp{sizeof(sp), 1.75f, 0.25f};
        check_unary_f32(r, SCRAPGPU_OP_SCALE, &sp, sizeof(sp), ggml_scale_test, "SCALE", n);

        RunCtx r2;
        r2.cpu = rc.cpu;
        r2.init();
        scrapgpu_clamp_params cp{sizeof(cp), -0.5f, 1.25f};
        check_unary_f32(r2, SCRAPGPU_OP_CLAMP, &cp, sizeof(cp), ggml_clamp_test, "CLAMP", n);

        RunCtx r3;
        r3.cpu = rc.cpu;
        r3.init();
        check_unary_f32(r3, SCRAPGPU_OP_SQR, nullptr, 0, ggml_sqr, "SQR", n);

        RunCtx r4;
        r4.cpu = rc.cpu;
        r4.init();
        ggml_tensor * a4 = ggml_new_tensor_1d(r4.ctx, GGML_TYPE_F32, n);
        ggml_tensor * out4 = ggml_sqrt(r4.ctx, a4);
        if (r4.alloc()) {
            r4.fill_f32(a4, 0.0f, 4.0f);
            r4.compute(out4);
            std::vector<uint8_t> ref(ggml_nbytes(out4));
            scrapgpu_tensor_desc sa = desc_from(a4);
            scrapgpu_tensor_desc so = desc_from(out4);
            void * ip[1] = {ggml_get_data(a4)};
            void * op[1] = {ref.data()};
            ref_exec(SCRAPGPU_OP_SQRT, &sa, 1, ip, &so, 1, op, nullptr, 0);
            const auto gg = tensor_to_f32(out4);
            std::vector<float> rf(gg.size());
            std::memcpy(rf.data(), ref.data(), ref.size());
            const double e = nmse_f32(rf.data(), gg.data(), gg.size());
            report("SQRT", std::to_string(n), e, kNmseFloat, e <= kNmseFloat);
        }

        struct U {
            scrapgpu_unary_kind sk;
            ggml_unary_op gk;
            const char * name;
        };
        const U kinds[] = {
            {SCRAPGPU_UNARY_SILU, GGML_UNARY_OP_SILU, "UNARY_SILU"},
            {SCRAPGPU_UNARY_RELU, GGML_UNARY_OP_RELU, "UNARY_RELU"},
            {SCRAPGPU_UNARY_NEG, GGML_UNARY_OP_NEG, "UNARY_NEG"},
            {SCRAPGPU_UNARY_ABS, GGML_UNARY_OP_ABS, "UNARY_ABS"},
            {SCRAPGPU_UNARY_SIGMOID, GGML_UNARY_OP_SIGMOID, "UNARY_SIGMOID"},
            {SCRAPGPU_UNARY_TANH, GGML_UNARY_OP_TANH, "UNARY_TANH"},
        };
        for (const auto & u : kinds) {
            RunCtx ru;
            ru.cpu = rc.cpu;
            ru.init();
            scrapgpu_unary_params up{sizeof(up), u.sk};
            ggml_tensor * a = ggml_new_tensor_1d(ru.ctx, GGML_TYPE_F32, n);
            ggml_tensor * out = ggml_unary(ru.ctx, a, u.gk);
            if (!ru.alloc()) {
                continue;
            }
            ru.fill_f32(a, -2.0f, 2.0f);
            if (!ru.compute(out)) {
                report(u.name, "compute", 1.0, 0.0, false);
                continue;
            }
            std::vector<uint8_t> ref(ggml_nbytes(out));
            scrapgpu_tensor_desc sa = desc_from(a);
            scrapgpu_tensor_desc so = desc_from(out);
            void * ip[1] = {ggml_get_data(a)};
            void * op[1] = {ref.data()};
            if (ref_exec(SCRAPGPU_OP_UNARY, &sa, 1, ip, &so, 1, op, &up, sizeof(up)) != SCRAPGPU_STATUS_OK) {
                report(u.name, "ref", 1.0, 0.0, false);
                continue;
            }
            const auto gg = tensor_to_f32(out);
            std::vector<float> rf(gg.size());
            std::memcpy(rf.data(), ref.data(), ref.size());
            const double e = nmse_f32(rf.data(), gg.data(), gg.size());
            report(u.name, std::to_string(n), e, kNmseFloat, e <= kNmseFloat);
        }
    }
}

static void test_glu(RunCtx & rc) {
    const int n = 8;
    // fused halves
    for (int swapped : {0, 1}) {
        for (auto gop : {GGML_GLU_OP_SWIGLU, GGML_GLU_OP_REGLU}) {
            RunCtx r;
            r.cpu = rc.cpu;
            r.init();
            ggml_tensor * a = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, n);
            ggml_tensor * out = ggml_glu(r.ctx, a, gop, swapped != 0);
            if (!r.alloc()) {
                continue;
            }
            r.fill_f32(a);
            if (!r.compute(out)) {
                report("GLU", "fused", 1.0, 0.0, false);
                continue;
            }
            std::vector<uint8_t> ref(ggml_nbytes(out));
            scrapgpu_glu_params gp{};
            gp.struct_size = sizeof(gp);
            gp.kind = gop == GGML_GLU_OP_SWIGLU ? SCRAPGPU_GLU_SWIGLU : SCRAPGPU_GLU_REGLU;
            gp.split_inputs = 0;
            gp.swapped = swapped;
            scrapgpu_tensor_desc sa = desc_from(a);
            scrapgpu_tensor_desc so = desc_from(out);
            void * ip[1] = {ggml_get_data(a)};
            void * op[1] = {ref.data()};
            ref_exec(SCRAPGPU_OP_GLU, &sa, 1, ip, &so, 1, op, &gp, sizeof(gp));
            const auto gg = tensor_to_f32(out);
            std::vector<float> rf(gg.size());
            std::memcpy(rf.data(), ref.data(), ref.size());
            const double e = nmse_f32(rf.data(), gg.data(), gg.size());
            char buf[64];
            std::snprintf(buf, sizeof(buf), "fused op=%d swap=%d", (int) gop, swapped);
            report("GLU", buf, e, kNmseFloat, e <= kNmseFloat);
        }
    }
    // split
    RunCtx rs;
    rs.cpu = rc.cpu;
    rs.init();
    ggml_tensor * x = ggml_new_tensor_1d(rs.ctx, GGML_TYPE_F32, n / 2);
    ggml_tensor * g = ggml_new_tensor_1d(rs.ctx, GGML_TYPE_F32, n / 2);
    ggml_tensor * out = ggml_glu_split(rs.ctx, x, g, GGML_GLU_OP_SWIGLU);
    if (rs.alloc()) {
        rs.fill_f32(x);
        rs.fill_f32(g);
        rs.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_glu_params gp{};
        gp.struct_size = sizeof(gp);
        gp.kind = SCRAPGPU_GLU_SWIGLU;
        gp.split_inputs = 1;
        gp.swapped = 0;
        scrapgpu_tensor_desc tx = desc_from(x);
        scrapgpu_tensor_desc tg = desc_from(g);
        scrapgpu_tensor_desc to = desc_from(out);
        const scrapgpu_tensor_desc ins[2] = {tx, tg};
        void * in_ptr[2] = {ggml_get_data(x), ggml_get_data(g)};
        void * out_ptr[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_GLU, ins, 2, in_ptr, &to, 1, out_ptr, &gp, sizeof(gp));
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        report("GLU", "split swiglu", e, kNmseFloat, e <= kNmseFloat);
    }
}

static void test_rms_norm(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    ggml_tensor * a = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, 7, 5);
    const float eps = 1e-5f;
    ggml_tensor * out = ggml_rms_norm(r.ctx, a, eps);
    if (!r.alloc()) {
        return;
    }
    r.fill_f32(a);
    r.compute(out);
    std::vector<uint8_t> ref(ggml_nbytes(out));
    scrapgpu_rms_norm_params rp{sizeof(rp), eps};
    scrapgpu_tensor_desc sa = desc_from(a);
    scrapgpu_tensor_desc so = desc_from(out);
    void * ip[1] = {ggml_get_data(a)};
    void * op[1] = {ref.data()};
    ref_exec(SCRAPGPU_OP_RMS_NORM, &sa, 1, ip, &so, 1, op, &rp, sizeof(rp));
    const auto gg = tensor_to_f32(out);
    std::vector<float> rf(gg.size());
    std::memcpy(rf.data(), ref.data(), ref.size());
    const double e = nmse_f32(rf.data(), gg.data(), gg.size());
    report("RMS_NORM", "7x5", e, kNmseFloat, e <= kNmseFloat);
}

static void test_rope(RunCtx & rc) {
    const int modes[] = {GGML_ROPE_TYPE_NORMAL, GGML_ROPE_TYPE_NEOX};
    for (int mode : modes) {
        RunCtx r;
        r.cpu = rc.cpu;
        r.init();
        const int n_embd = 8;
        const int n_tok = 3;
        ggml_tensor * a = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, n_embd, 1, n_tok, 1);
        ggml_tensor * pos = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, n_tok);
        ggml_tensor * out = ggml_rope_ext(r.ctx, a, pos, nullptr, n_embd, mode, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        if (!r.alloc()) {
            continue;
        }
        r.fill_f32(a);
        r.fill_i32(pos, 0, 16);
        r.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_rope_params rp{};
        rp.struct_size = sizeof(rp);
        rp.mode = mode == GGML_ROPE_TYPE_NEOX ? SCRAPGPU_ROPE_MODE_NEOX : SCRAPGPU_ROPE_MODE_NORMAL;
        rp.n_dims = n_embd;
        rp.freq_base = 10000.0f;
        rp.freq_scale = 1.0f;
        rp.attn_factor = 1.0f;
        scrapgpu_tensor_desc ta = desc_from(a);
        scrapgpu_tensor_desc tp = desc_from(pos);
        scrapgpu_tensor_desc to = desc_from(out);
        const scrapgpu_tensor_desc ins[2] = {ta, tp};
        void * in_ptr[2] = {ggml_get_data(a), ggml_get_data(pos)};
        void * out_ptr[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_ROPE, ins, 2, in_ptr, &to, 1, out_ptr, &rp, sizeof(rp));
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        char buf[48];
        std::snprintf(buf, sizeof(buf), "mode=%d full dims", mode);
        report("ROPE", buf, e, kNmseFloat, e <= kNmseFloat);
    }
    // n_dims < ne0
    RunCtx r2;
    r2.cpu = rc.cpu;
    r2.init();
    ggml_tensor * a = ggml_new_tensor_4d(r2.ctx, GGML_TYPE_F32, 10, 1, 2, 1);
    ggml_tensor * pos = ggml_new_tensor_1d(r2.ctx, GGML_TYPE_I32, 2);
    ggml_tensor * out = ggml_rope_ext(r2.ctx, a, pos, nullptr, 6, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (r2.alloc()) {
        r2.fill_f32(a);
        r2.fill_i32(pos, 1, 9);
        r2.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_rope_params rp{};
        rp.struct_size = sizeof(rp);
        rp.mode = SCRAPGPU_ROPE_MODE_NORMAL;
        rp.n_dims = 6;
        rp.freq_base = 10000.0f;
        rp.freq_scale = 1.0f;
        rp.attn_factor = 1.0f;
        scrapgpu_tensor_desc ta = desc_from(a);
        scrapgpu_tensor_desc tp = desc_from(pos);
        scrapgpu_tensor_desc to = desc_from(out);
        const scrapgpu_tensor_desc ins[2] = {ta, tp};
        void * in_ptr[2] = {ggml_get_data(a), ggml_get_data(pos)};
        void * out_ptr[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_ROPE, ins, 2, in_ptr, &to, 1, out_ptr, &rp, sizeof(rp));
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        report("ROPE", "n_dims=6 ne0=10", e, kNmseFloat, e <= kNmseFloat);
    }
    // freq factors
    RunCtx r3;
    r3.cpu = rc.cpu;
    r3.init();
    ggml_tensor * a3 = ggml_new_tensor_4d(r3.ctx, GGML_TYPE_F32, 4, 1, 1, 1);
    ggml_tensor * pos3 = ggml_new_tensor_1d(r3.ctx, GGML_TYPE_I32, 1);
    ggml_tensor * ff = ggml_new_tensor_1d(r3.ctx, GGML_TYPE_F32, 2);
    ggml_tensor * out3 = ggml_rope_ext(r3.ctx, a3, pos3, ff, 4, GGML_ROPE_TYPE_NORMAL, 0, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (r3.alloc()) {
        r3.fill_f32(a3);
        std::vector<int32_t> pv = {2};
        ggml_backend_tensor_set(pos3, pv.data(), 0, pv.size() * sizeof(int32_t));
        std::vector<float> fv = {2.0f, 4.0f};
        ggml_backend_tensor_set(ff, fv.data(), 0, fv.size() * sizeof(float));
        r3.compute(out3);
        std::vector<uint8_t> ref(ggml_nbytes(out3));
        scrapgpu_rope_params rp{};
        rp.struct_size = sizeof(rp);
        rp.mode = SCRAPGPU_ROPE_MODE_NORMAL;
        rp.n_dims = 4;
        rp.freq_base = 10000.0f;
        rp.freq_scale = 1.0f;
        rp.attn_factor = 1.0f;
        scrapgpu_tensor_desc ta = desc_from(a3);
        scrapgpu_tensor_desc tp = desc_from(pos3);
        scrapgpu_tensor_desc tf = desc_from(ff);
        scrapgpu_tensor_desc to = desc_from(out3);
        const scrapgpu_tensor_desc ins[3] = {ta, tp, tf};
        void * in_ptr[3] = {ggml_get_data(a3), ggml_get_data(pos3), ggml_get_data(ff)};
        void * out_ptr[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_ROPE, ins, 3, in_ptr, &to, 1, out_ptr, &rp, sizeof(rp));
        const auto gg = tensor_to_f32(out3);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        report("ROPE", "freq_factors", e, kNmseFloat, e <= kNmseFloat);
    }
}

static void test_soft_max(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    ggml_tensor * a = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, 5, 3, 2);
    const float scale = 0.5f;
    ggml_tensor * out0 = ggml_soft_max_ext(r.ctx, a, nullptr, scale, 0.0f);
    if (r.alloc()) {
        r.fill_f32(a);
        r.compute(out0);
        std::vector<uint8_t> ref(ggml_nbytes(out0));
        scrapgpu_soft_max_params sp{sizeof(sp), scale, nullptr};
        scrapgpu_tensor_desc sa = desc_from(a);
        scrapgpu_tensor_desc so = desc_from(out0);
        void * ip[1] = {ggml_get_data(a)};
        void * op[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_SOFT_MAX, &sa, 1, ip, &so, 1, op, &sp, sizeof(sp));
        const auto gg = tensor_to_f32(out0);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        report("SOFT_MAX", "no mask", e, kNmseFloat, e <= kNmseFloat);
    }
    for (auto mt : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        RunCtx r2;
        r2.cpu = rc.cpu;
        r2.init();
        a = ggml_new_tensor_3d(r2.ctx, GGML_TYPE_F32, 5, 3, 2);
        ggml_tensor * mask = ggml_new_tensor_3d(r2.ctx, mt, 5, 3, 1);
        ggml_tensor * out = ggml_soft_max_ext(r2.ctx, a, mask, scale, 0.0f);
        if (!r2.alloc()) {
            continue;
        }
        r2.fill_f32(a);
        if (mt == GGML_TYPE_F32) {
            r2.fill_f32(mask, -1.0f, 1.0f);
        } else {
            std::vector<float> mf(5 * 3);
            for (auto & v : mf) {
                v = frand();
            }
            std::vector<ggml_fp16_t> mh(mf.size());
            ggml_fp32_to_fp16_row(mf.data(), mh.data(), (int) mh.size());
            ggml_backend_tensor_set(mask, mh.data(), 0, mh.size() * sizeof(ggml_fp16_t));
        }
        r2.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_soft_max_params sp{sizeof(sp), scale, nullptr};
        scrapgpu_tensor_desc sa = desc_from(a);
        scrapgpu_tensor_desc sm = desc_from(mask);
        scrapgpu_tensor_desc so = desc_from(out);
        const scrapgpu_tensor_desc ins[2] = {sa, sm};
        void * in_ptr[2] = {ggml_get_data(a), ggml_get_data(mask)};
        void * out_ptr[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_SOFT_MAX, ins, 2, in_ptr, &so, 1, out_ptr, &sp, sizeof(sp));
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        char buf[32];
        std::snprintf(buf, sizeof(buf), "mask %s", mt == GGML_TYPE_F32 ? "F32" : "F16");
        report("SOFT_MAX", buf, e, kNmseFloat, e <= kNmseFloat);
    }
}

static void test_get_rows(RunCtx & rc) {
    const enum ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    for (auto wt : types) {
        RunCtx r;
        r.cpu = rc.cpu;
        r.init();
        const int K = (wt == GGML_TYPE_Q4_K || wt == GGML_TYPE_Q6_K) ? 256 : 64;
        ggml_tensor * src = ggml_new_tensor_2d(r.ctx, wt, K, 7);
        ggml_tensor * idx = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, 4);
        ggml_tensor * out = ggml_get_rows(r.ctx, src, idx);
        if (!r.alloc()) {
            continue;
        }
        std::vector<float> f((size_t) K * 7);
        for (auto & v : f) {
            v = frand() * 2.0f;
        }
        std::vector<uint8_t> q(ggml_nbytes(src));
        ggml_quantize_chunk(wt, f.data(), q.data(), 0, 7, K, nullptr);
        ggml_backend_tensor_set(src, q.data(), 0, q.size());
        std::vector<int32_t> iv = {1, 6, 2, 0};
        ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int32_t));
        r.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_get_rows_params gp{sizeof(gp), 0};
        scrapgpu_tensor_desc ts = desc_from(src);
        scrapgpu_tensor_desc ti = desc_from(idx);
        scrapgpu_tensor_desc to = desc_from(out);
        const scrapgpu_tensor_desc ins[2] = {ts, ti};
        void * in_ptr[2] = {ggml_get_data(src), ggml_get_data(idx)};
        void * out_ptr[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_GET_ROWS, ins, 2, in_ptr, &to, 1, out_ptr, &gp, sizeof(gp));
        char buf[32];
        std::snprintf(buf, sizeof(buf), "type=%d", (int) wt);
        report_exact("GET_ROWS", buf, bytes_equal(out, ref));
    }
}

static void test_set_rows(RunCtx & rc) {
    struct Case {
        enum ggml_type dst;
        enum ggml_type src;
        enum ggml_type idt;
        int emb;
        int rows;
        int ntok;
    };
    const Case cases[] = {
        {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_I32, 5, 9, 3},
        {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I64, 6, 11, 4},
    };
    for (const auto & c : cases) {
        RunCtx r;
        r.cpu = rc.cpu;
        r.init();
        ggml_tensor * dst = ggml_new_tensor_2d(r.ctx, c.dst, c.emb, c.rows);
        ggml_tensor * src = ggml_new_tensor_2d(r.ctx, c.src, c.emb, c.ntok);
        ggml_tensor * idx = ggml_new_tensor_1d(r.ctx, c.idt, c.ntok);
        ggml_tensor * out = ggml_set_rows(r.ctx, dst, src, idx);
        if (!r.alloc()) {
            continue;
        }
        r.fill_f32(src);
        if (c.dst == GGML_TYPE_F32) {
            r.fill_f32(dst);
        } else {
            std::vector<float> df((size_t) c.emb * c.rows, 0.25f);
            std::vector<ggml_fp16_t> dh(df.size());
            ggml_fp32_to_fp16_row(df.data(), dh.data(), (int) dh.size());
            ggml_backend_tensor_set(dst, dh.data(), 0, dh.size() * sizeof(ggml_fp16_t));
        }
        if (c.idt == GGML_TYPE_I32) {
            std::vector<int32_t> iv;
            for (int i = 0; i < c.ntok; ++i) {
                iv.push_back((i * 3 + 1) % c.rows);
            }
            ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int32_t));
        } else {
            std::vector<int64_t> iv;
            for (int i = 0; i < c.ntok; ++i) {
                iv.push_back((i * 2 + 2) % c.rows);
            }
            ggml_backend_tensor_set(idx, iv.data(), 0, iv.size() * sizeof(int64_t));
        }
        std::vector<uint8_t> dst_before(ggml_nbytes(dst));
        ggml_backend_tensor_get(dst, dst_before.data(), 0, dst_before.size());
        r.compute(out);
        std::vector<uint8_t> ref_dst = dst_before;
        scrapgpu_set_rows_params sp{};
        sp.struct_size = sizeof(sp);
        sp.indices_are_i64 = c.idt == GGML_TYPE_I64;
        scrapgpu_tensor_desc ts = desc_from(src);
        scrapgpu_tensor_desc ti = desc_from(idx);
        scrapgpu_tensor_desc td = desc_from(dst);
        const scrapgpu_tensor_desc ins[2] = {ts, ti};
        void * in_ptr[2] = {ggml_get_data(src), ggml_get_data(idx)};
        void * out_ptr[1] = {ref_dst.data()};
        ref_exec(SCRAPGPU_OP_SET_ROWS, ins, 2, in_ptr, &td, 1, out_ptr, &sp, sizeof(sp));
        char buf[64];
        std::snprintf(buf, sizeof(buf), "dst=%d src=%d id=%d", (int) c.dst, (int) c.src, (int) c.idt);
        report_exact("SET_ROWS", buf, bytes_equal(dst, ref_dst));
    }
}

static void test_cpy_cont(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    ggml_tensor * src = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, 5, 3);
    ggml_tensor * dst = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F16, 5, 3);
    ggml_tensor * out = ggml_cpy(r.ctx, src, dst);
    if (r.alloc()) {
        r.fill_f32(src);
        r.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(dst));
        scrapgpu_cpy_params cp{sizeof(cp), 0};
        scrapgpu_tensor_desc ss = desc_from(src);
        scrapgpu_tensor_desc sd = desc_from(dst);
        void * ip[1] = {ggml_get_data(src)};
        void * op[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_CPY, &ss, 1, ip, &sd, 1, op, &cp, sizeof(cp));
        report_exact("CPY", "F32->F16", bytes_equal(dst, ref));
    }
    RunCtx r2;
    r2.cpu = rc.cpu;
    r2.init();
    ggml_tensor * base = ggml_new_tensor_2d(r2.ctx, GGML_TYPE_F32, 4, 5);
    ggml_tensor * view = ggml_transpose(r2.ctx, base);
    ggml_tensor * cont = ggml_cont(r2.ctx, view);
    if (r2.alloc()) {
        r2.fill_f32(base);
        r2.compute(cont);
        std::vector<uint8_t> ref(ggml_nbytes(cont));
        scrapgpu_cpy_params cp{sizeof(cp), 1};
        scrapgpu_tensor_desc sv = desc_from(view);
        scrapgpu_tensor_desc sc = desc_from(cont);
        void * ip[1] = {ggml_get_data(view)};
        void * op[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_CONT, &sv, 1, ip, &sc, 1, op, &cp, sizeof(cp));
        report_exact("CONT", "transposed F32", bytes_equal(cont, ref));
    }
}

static int mul_mat_k1(enum ggml_type wt) {
    if (wt == GGML_TYPE_Q4_K || wt == GGML_TYPE_Q6_K) {
        return 256;
    }
    return 32;
}

static int mul_mat_k_odd(enum ggml_type wt) {
    if (wt == GGML_TYPE_Q4_K || wt == GGML_TYPE_Q6_K) {
        return 512;
    }
    return 96;
}

static void test_mul_mat(RunCtx & rc) {
    const enum ggml_type wts[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    for (auto wt : wts) {
        const int k0 = mul_mat_k1(wt);
        const int k1 = mul_mat_k_odd(wt);
        check_mul_mat_case(rc.cpu, wt, k0, 5, 3, 1, 1, false);
        check_mul_mat_case(rc.cpu, wt, k1, 7, 1, 1, 1, false);
        check_mul_mat_case(rc.cpu, wt, k0, 4, 2, 2, 2, false);
        check_mul_mat_case(rc.cpu, wt, k0, 3, 2, 1, 1, false);
        check_mul_mat_case(rc.cpu, wt, k0, 3, 2, 1, 1, true);
    }
}

static void test_mul_mat_id(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    ggml_tensor * w = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, 2, 2, 2);
    ggml_tensor * x = ggml_new_tensor_3d(r.ctx, GGML_TYPE_F32, 2, 1, 2);
    ggml_tensor * ids = ggml_new_tensor_2d(r.ctx, GGML_TYPE_I32, 2, 2);
    ggml_tensor * out = ggml_mul_mat_id(r.ctx, w, x, ids);
    if (!r.alloc()) {
        return;
    }
    const std::vector<float> wf = {1, 2, 3, 4, 10, 20, 30, 40};
    const std::vector<float> xf = {1, 1, 2, 0};
    const std::vector<int32_t> idf = {0, 1, 1, 0};
    ggml_backend_tensor_set(w, wf.data(), 0, wf.size() * sizeof(float));
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size() * sizeof(float));
    ggml_backend_tensor_set(ids, idf.data(), 0, idf.size() * sizeof(int32_t));
    r.compute(out);
    std::vector<uint8_t> ref(ggml_nbytes(out));
    scrapgpu_mul_mat_id_params mp{sizeof(mp), 2, 0};
    scrapgpu_tensor_desc tw = desc_from(w);
    scrapgpu_tensor_desc tx = desc_from(x);
    scrapgpu_tensor_desc ti = desc_from(ids);
    scrapgpu_tensor_desc to = desc_from(out);
    const scrapgpu_tensor_desc ins[3] = {tw, tx, ti};
    void * in_ptr[3] = {ggml_get_data(w), ggml_get_data(x), ggml_get_data(ids)};
    void * out_ptr[1] = {ref.data()};
    ref_exec(SCRAPGPU_OP_MUL_MAT_ID, ins, 3, in_ptr, &to, 1, out_ptr, &mp, sizeof(mp));
    const auto gg = tensor_to_f32(out);
    std::vector<float> rf(gg.size());
    std::memcpy(rf.data(), ref.data(), ref.size());
    const double e = nmse_f32(rf.data(), gg.data(), gg.size());
    report("MUL_MAT_ID", "2x2 experts", e, kNmseFloat, e <= kNmseFloat);
}

static void test_add_id(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    ggml_tensor * a = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, 2, 2);
    ggml_tensor * b = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, 2, 3);
    ggml_tensor * ids = ggml_new_tensor_1d(r.ctx, GGML_TYPE_I32, 2);
    ggml_tensor * out = ggml_add_id(r.ctx, a, b, ids);
    if (!r.alloc()) {
        return;
    }
    const std::vector<float> av = {1, 2, 3, 4};
    const std::vector<float> bv = {10, 20, 30, 40, 50, 60};
    const std::vector<int32_t> iv = {2, 0};
    ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
    ggml_backend_tensor_set(b, bv.data(), 0, bv.size() * sizeof(float));
    ggml_backend_tensor_set(ids, iv.data(), 0, iv.size() * sizeof(int32_t));
    r.compute(out);
    std::vector<uint8_t> ref(ggml_nbytes(out));
    scrapgpu_tensor_desc ta = desc_from(a);
    scrapgpu_tensor_desc tb = desc_from(b);
    scrapgpu_tensor_desc ti = desc_from(ids);
    scrapgpu_tensor_desc to = desc_from(out);
    const scrapgpu_tensor_desc ins[3] = {ta, tb, ti};
    void * in_ptr[3] = {ggml_get_data(a), ggml_get_data(b), ggml_get_data(ids)};
    void * out_ptr[1] = {ref.data()};
    ref_exec(SCRAPGPU_OP_ADD_ID, ins, 3, in_ptr, &to, 1, out_ptr, nullptr, 0);
    const auto gg = tensor_to_f32(out);
    std::vector<float> rf(gg.size());
    std::memcpy(rf.data(), ref.data(), ref.size());
    const double e = nmse_f32(rf.data(), gg.data(), gg.size());
    report("ADD_ID", "2x2", e, kNmseFloat, e <= kNmseFloat);
}

static void test_ssm_conv(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    const int d_conv = 3;
    const int n_t = 5;
    const int d_inner = 2;
    ggml_tensor * sx = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, d_conv - 1 + n_t, d_inner);
    ggml_tensor * c = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, d_conv, d_inner);
    ggml_tensor * out = ggml_ssm_conv(r.ctx, sx, c);
    if (!r.alloc()) {
        return;
    }
    r.fill_f32(sx);
    r.fill_f32(c);
    r.compute(out);
    std::vector<uint8_t> ref(ggml_nbytes(out));
    scrapgpu_ssm_conv_params sp{sizeof(sp), (uint32_t) d_conv};
    scrapgpu_tensor_desc ts = desc_from(sx);
    scrapgpu_tensor_desc tc = desc_from(c);
    scrapgpu_tensor_desc to = desc_from(out);
    const scrapgpu_tensor_desc ins[2] = {ts, tc};
    void * in_ptr[2] = {ggml_get_data(sx), ggml_get_data(c)};
    void * out_ptr[1] = {ref.data()};
    ref_exec(SCRAPGPU_OP_SSM_CONV, ins, 2, in_ptr, &to, 1, out_ptr, &sp, sizeof(sp));
    const auto gg = tensor_to_f32(out);
    std::vector<float> rf(gg.size());
    std::memcpy(rf.data(), ref.data(), ref.size());
    const double e = nmse_f32(rf.data(), gg.data(), gg.size());
    report("SSM_CONV", "d_conv=3", e, kNmseFloat, e <= kNmseFloat);
}

static void test_concat(RunCtx & rc) {
    for (int dim = 0; dim < 4; ++dim) {
        RunCtx r;
        r.cpu = rc.cpu;
        r.init();
        int64_t nea[4] = {3, 2, 2, 1};
        int64_t neb[4] = {3, 2, 2, 1};
        neb[dim] = 1;
        ggml_tensor * a = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, nea[0], nea[1], nea[2], nea[3]);
        ggml_tensor * b = ggml_new_tensor_4d(r.ctx, GGML_TYPE_F32, neb[0], neb[1], neb[2], neb[3]);
        ggml_tensor * out = ggml_concat(r.ctx, a, b, dim);
        if (!r.alloc()) {
            continue;
        }
        r.fill_f32(a);
        r.fill_f32(b);
        r.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_concat_params cp{sizeof(cp), (uint32_t) dim};
        scrapgpu_tensor_desc ta = desc_from(a);
        scrapgpu_tensor_desc tb = desc_from(b);
        scrapgpu_tensor_desc to = desc_from(out);
        const scrapgpu_tensor_desc ins[2] = {ta, tb};
        void * in_ptr[2] = {ggml_get_data(a), ggml_get_data(b)};
        void * out_ptr[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_CONCAT, ins, 2, in_ptr, &to, 1, out_ptr, &cp, sizeof(cp));
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        char buf[16];
        std::snprintf(buf, sizeof(buf), "dim=%d", dim);
        report("CONCAT", buf, e, kNmseFloat, e <= kNmseFloat);
    }
}

static void test_pad(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    ggml_tensor * a = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, 5);
    ggml_tensor * out = ggml_pad_ext(r.ctx, a, 1, 2, 0, 0, 0, 0, 0, 0);
    if (r.alloc()) {
        std::vector<float> av = {1, 2, 3, 4, 5};
        ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
        r.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_pad_params pp{};
        pp.struct_size = sizeof(pp);
        pp.before[0] = 1;
        pp.after[0] = 2;
        scrapgpu_tensor_desc sa = desc_from(a);
        scrapgpu_tensor_desc so = desc_from(out);
        void * ip[1] = {ggml_get_data(a)};
        void * op[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_PAD, &sa, 1, ip, &so, 1, op, &pp, sizeof(pp));
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        report("PAD", "zero ext", e, kNmseFloat, e <= kNmseFloat);
    }
    RunCtx r2;
    r2.cpu = rc.cpu;
    r2.init();
    a = ggml_new_tensor_1d(r2.ctx, GGML_TYPE_F32, 5);
    out = ggml_pad_ext_circular(r2.ctx, a, 1, 1, 0, 0, 0, 0, 0, 0);
    if (r2.alloc()) {
        std::vector<float> av = {1, 2, 3, 4, 5};
        ggml_backend_tensor_set(a, av.data(), 0, av.size() * sizeof(float));
        r2.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_pad_params pp{};
        pp.struct_size = sizeof(pp);
        pp.before[0] = 1;
        pp.after[0] = 1;
        pp.circular = 1;
        scrapgpu_tensor_desc sa = desc_from(a);
        scrapgpu_tensor_desc so = desc_from(out);
        void * ip[1] = {ggml_get_data(a)};
        void * op[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_PAD, &sa, 1, ip, &so, 1, op, &pp, sizeof(pp));
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        report("PAD", "circular", e, kNmseFloat, e <= kNmseFloat);
    }
}

static void test_sum_rows(RunCtx & rc) {
    RunCtx r;
    r.cpu = rc.cpu;
    r.init();
    ggml_tensor * a = ggml_new_tensor_2d(r.ctx, GGML_TYPE_F32, 7, 4);
    ggml_tensor * out = ggml_sum_rows(r.ctx, a);
    if (r.alloc()) {
        r.fill_f32(a);
        r.compute(out);
        std::vector<uint8_t> ref(ggml_nbytes(out));
        scrapgpu_tensor_desc sa = desc_from(a);
        scrapgpu_tensor_desc so = desc_from(out);
        void * ip[1] = {ggml_get_data(a)};
        void * op[1] = {ref.data()};
        ref_exec(SCRAPGPU_OP_SUM_ROWS, &sa, 1, ip, &so, 1, op, nullptr, 0);
        const auto gg = tensor_to_f32(out);
        std::vector<float> rf(gg.size());
        std::memcpy(rf.data(), ref.data(), ref.size());
        const double e = nmse_f32(rf.data(), gg.data(), gg.size());
        report("SUM_ROWS", "7x4", e, kNmseFloat, e <= kNmseFloat);
    }
}

static void test_argsort_one(ggml_backend_t cpu, enum ggml_sort_order ord, const std::vector<float> & kv) {
    RunCtx r;
    r.cpu = cpu;
    r.init();
    ggml_tensor * keys = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, (int64_t) kv.size());
    ggml_tensor * gg_out = ggml_argsort(r.ctx, keys, ord);
    if (!r.alloc()) {
        return;
    }
    ggml_backend_tensor_set(keys, kv.data(), 0, kv.size() * sizeof(float));
    r.compute(gg_out);
    std::vector<uint8_t> ref(ggml_nbytes(gg_out));
    scrapgpu_argsort_params ap{};
    ap.struct_size = sizeof(ap);
    ap.order = ord == GGML_SORT_ORDER_ASC ? SCRAPGPU_ARGSORT_ASC : SCRAPGPU_ARGSORT_DESC;
    ap.top_k = 0;
    scrapgpu_tensor_desc sk = desc_from(keys);
    scrapgpu_tensor_desc so = desc_from(gg_out);
    void * ip[1] = {ggml_get_data(keys)};
    void * op[1] = {ref.data()};
    ref_exec(SCRAPGPU_OP_ARGSORT, &sk, 1, ip, &so, 1, op, &ap, sizeof(ap));
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", ord == GGML_SORT_ORDER_ASC ? "asc" : "desc");
    report_exact("ARGSORT", buf,
                 indices_match_sort(keys, gg_out, reinterpret_cast<const int32_t *>(ref.data()),
                                    ord == GGML_SORT_ORDER_ASC));
}

static void test_top_k_one(ggml_backend_t cpu, const std::vector<float> & kv, int k) {
    RunCtx r;
    r.cpu = cpu;
    r.init();
    ggml_tensor * keys = ggml_new_tensor_1d(r.ctx, GGML_TYPE_F32, (int64_t) kv.size());
    ggml_tensor * tk = ggml_top_k(r.ctx, keys, k);
    if (!r.alloc()) {
        return;
    }
    ggml_backend_tensor_set(keys, kv.data(), 0, kv.size() * sizeof(float));
    r.compute(tk);
    std::vector<uint8_t> refk(ggml_nbytes(tk));
    scrapgpu_argsort_params tp{};
    tp.struct_size = sizeof(tp);
    tp.order = SCRAPGPU_ARGSORT_ASC;
    tp.top_k = (uint32_t) k;
    scrapgpu_tensor_desc sk = desc_from(keys);
    scrapgpu_tensor_desc stk = desc_from(tk);
    void * ip[1] = {ggml_get_data(keys)};
    void * op[1] = {refk.data()};
    ref_exec(SCRAPGPU_OP_TOP_K, &sk, 1, ip, &stk, 1, op, &tp, sizeof(tp));
    report_exact("TOP_K", "k=3 asc", top_k_match(keys, tk, reinterpret_cast<const int32_t *>(refk.data()), k, true));
}

static void test_argsort_topk(RunCtx & rc) {
    const std::vector<float> kv = {3.0f, 1.0f, 2.0f, 1.0f, 2.0f, 0.5f, 4.0f, 4.0f, 3.5f};
    test_argsort_one(rc.cpu, GGML_SORT_ORDER_ASC, kv);
    test_argsort_one(rc.cpu, GGML_SORT_ORDER_DESC, kv);
    test_top_k_one(rc.cpu, kv, 3);
}

int main() {
    ggml_backend_load_all();
    if (const char * dir = std::getenv("GGML_LIB_DIR")) {
        ggml_backend_load_all_from_path(dir);
    }
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu) {
        std::fprintf(stderr, "CPU backend unavailable (set GGML_LIB_DIR)\n");
        return 1;
    }

    RunCtx rc;
    rc.cpu = cpu;
    rc.init();

    test_binary_ops(rc);
    test_unary_family(rc);
    test_glu(rc);
    test_rms_norm(rc);
    test_rope(rc);
    test_soft_max(rc);
    test_get_rows(rc);
    test_set_rows(rc);
    test_cpy_cont(rc);
    test_mul_mat(rc);
    test_mul_mat_id(rc);
    test_add_id(rc);
    test_ssm_conv(rc);
    test_concat(rc);
    test_pad(rc);
    test_sum_rows(rc);
    test_argsort_topk(rc);

    std::fprintf(stderr, "\nSummary: %d cases, %d OK, %d FAIL\n", g_cases, g_ok, g_fail);
    ggml_backend_free(cpu);
    return g_fail > 0 ? 1 : 0;
}
