// Unit tests of scrapgpu_ref_execute: every op gets a small hand-checked case
// plus edge cases (non-contiguous strides via nb, broadcasting, odd sizes).

#include "scrapgpu/reference.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// CHECK stays active in Release builds (assert() is compiled out with NDEBUG)
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);   \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

#define CHECK_FEQ(a, b, tol)                                                                        \
    do {                                                                                            \
        const double va_ = static_cast<double>(a);                                                  \
        const double vb_ = static_cast<double>(b);                                                  \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                                     \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s = %g vs %s = %g (tol %g)\n",              \
                         __FILE__, __LINE__, #a, va_, #b, vb_, static_cast<double>(tol));           \
            std::exit(1);                                                                           \
        }                                                                                           \
    } while (0)

#define CHECK_OK(status)                                                                        \
    do {                                                                                        \
        const scrapgpu_status st_ = (status);                                                   \
        if (st_ != SCRAPGPU_STATUS_OK) {                                                        \
            std::fprintf(stderr, "%s:%d: unexpected status %s: %s\n",                            \
                         __FILE__, __LINE__, scrapgpu_status_string(st_), scrapgpu_last_error()); \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

namespace {

scrapgpu_tensor_desc tdesc(scrapgpu_dtype dt, int64_t n0, int64_t n1 = 1, int64_t n2 = 1, int64_t n3 = 1) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = dt;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = n2;
    t.ne[3] = n3;
    t.nb[0] = scrapgpu_dtype_type_size(dt);
    t.nb[1] = scrapgpu_tensor_row_size(dt, n0);
    t.nb[2] = t.nb[1] * static_cast<uint64_t>(n1 > 0 ? n1 : 1);
    t.nb[3] = t.nb[2] * static_cast<uint64_t>(n2 > 0 ? n2 : 1);
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    return t;
}

scrapgpu_status execute(scrapgpu_op op,
                        const scrapgpu_tensor_desc * ins, const std::vector<void *> & in_data,
                        scrapgpu_tensor_desc * outs, const std::vector<void *> & out_data,
                        const void * params = nullptr, size_t params_size = 0) {
    scrapgpu_op_desc d{};
    d.struct_size = sizeof(d);
    d.op = op;
    d.inputs = ins;
    d.input_count = in_data.size();
    d.outputs = outs;
    d.output_count = out_data.size();
    d.params = params;
    d.params_size = params_size;
    return scrapgpu_ref_execute(&d, in_data.data(), out_data.data());
}

// ---------------------------------------------------------------------------------------------

void test_binary_ops() {
    // src0: 2x3 with a padded row stride (non-contiguous), src1: a row broadcast over dim 1
    scrapgpu_tensor_desc s0 = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
    s0.nb[1] = 4 * 4; // row stride padded to 4 floats
    s0.nb[2] = s0.nb[1] * 2;
    scrapgpu_tensor_desc s1 = tdesc(SCRAPGPU_DTYPE_F32, 3);
    scrapgpu_tensor_desc dd = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);

    std::vector<float> a(8, -7.0f); // 8 floats: the padded element stays untouched
    a[0] = 1; a[1] = 2; a[2] = 3;
    a[4] = 10; a[5] = 20; a[6] = 30;
    const std::vector<float> b = {2.0f, -1.0f, 0.5f};
    std::vector<float> y(6, -99.0f);

    {
        const scrapgpu_tensor_desc in[2] = {s0, s1};
        std::vector<float> yy(6, -99.0f);
        CHECK_OK(execute(SCRAPGPU_OP_ADD, in, {a.data(), const_cast<float *>(b.data())}, &dd, {yy.data()}));
        CHECK_FEQ(yy[0], 3, 1e-6);
        CHECK_FEQ(yy[1], 1, 1e-6);
        CHECK_FEQ(yy[2], 3.5, 1e-6);
        CHECK_FEQ(yy[3], 12, 1e-6);
        CHECK_FEQ(yy[4], 19, 1e-6);
        CHECK_FEQ(yy[5], 30.5, 1e-6);

        CHECK_OK(execute(SCRAPGPU_OP_SUB, in, {a.data(), const_cast<float *>(b.data())}, &dd, {yy.data()}));
        CHECK_FEQ(yy[0], -1, 1e-6);
        CHECK_FEQ(yy[5], 29.5, 1e-6);
        CHECK_OK(execute(SCRAPGPU_OP_MUL, in, {a.data(), const_cast<float *>(b.data())}, &dd, {yy.data()}));
        CHECK_FEQ(yy[0], 2, 1e-6);
        CHECK_FEQ(yy[4], -20, 1e-6);
        CHECK_OK(execute(SCRAPGPU_OP_DIV, in, {a.data(), const_cast<float *>(b.data())}, &dd, {yy.data()}));
        CHECK_FEQ(yy[0], 0.5, 1e-6);
        CHECK_FEQ(yy[4], -20, 1e-6);

        // broadcast on every dimension (src1 is a scalar in a 1x1x1x1 tensor)
        scrapgpu_tensor_desc sc = tdesc(SCRAPGPU_DTYPE_F32, 1, 1, 1, 1);
        const float two = 2.0f;
        const scrapgpu_tensor_desc in2[2] = {s0, sc};
        CHECK_OK(execute(SCRAPGPU_OP_MUL, in2, {a.data(), const_cast<float *>(&two)}, &dd, {yy.data()}));
        CHECK_FEQ(yy[3], 20, 1e-6);
    }
}

void test_unary_family() {
    const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 5);
    const scrapgpu_tensor_desc d = s;
    const std::vector<float> x = {1.0f, -2.0f, 0.0f, 0.5f, -0.25f};
    std::vector<float> y(5);

    scrapgpu_scale_params scale{sizeof(scale), 2.0f, 1.0f};
    CHECK_OK(execute(SCRAPGPU_OP_SCALE, &s, {const_cast<float *>(x.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &scale, sizeof(scale)));
    CHECK_FEQ(y[0], 3, 1e-6);
    CHECK_FEQ(y[1], -3, 1e-6);

    scrapgpu_clamp_params clamp{sizeof(clamp), -1.0f, 1.0f};
    CHECK_OK(execute(SCRAPGPU_OP_CLAMP, &s, {const_cast<float *>(x.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &clamp, sizeof(clamp)));
    CHECK_FEQ(y[0], 1, 1e-6);
    CHECK_FEQ(y[1], -1, 1e-6);
    CHECK_FEQ(y[3], 0.5, 1e-6);

    CHECK_OK(execute(SCRAPGPU_OP_SQR, &s, {const_cast<float *>(x.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}));
    CHECK_FEQ(y[0], 1, 1e-6);
    CHECK_FEQ(y[1], 4, 1e-6);
    CHECK_OK(execute(SCRAPGPU_OP_SQRT, &s, {const_cast<float *>(x.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}));
    CHECK_FEQ(y[0], 1, 1e-6);
    CHECK_FEQ(y[3], std::sqrt(0.5), 1e-6);

    const double silu1 = 1.0 / (1.0 + std::exp(-1.0));
    const double sig0 = 0.5;
    const double tanh1 = std::tanh(1.0);
    struct KindCase {
        scrapgpu_unary_kind kind;
        double expected[5];
    };
    const KindCase cases[] = {
        {SCRAPGPU_UNARY_SILU, {silu1, -2.0 / (1.0 + std::exp(2.0)), 0.0, 0.5 / (1.0 + std::exp(-0.5)), -0.25 / (1.0 + std::exp(0.25))}},
        {SCRAPGPU_UNARY_RELU, {1, 0, 0, 0.5, 0}},
        {SCRAPGPU_UNARY_NEG, {-1, 2, 0, -0.5, 0.25}},
        {SCRAPGPU_UNARY_ABS, {1, 2, 0, 0.5, 0.25}},
        {SCRAPGPU_UNARY_SIGMOID, {1.0 / (1.0 + std::exp(-1.0)), 1.0 / (1.0 + std::exp(2.0)), sig0, 1.0 / (1.0 + std::exp(-0.5)), 1.0 / (1.0 + std::exp(0.25))}},
        {SCRAPGPU_UNARY_TANH, {tanh1, std::tanh(-2.0), 0, std::tanh(0.5), std::tanh(-0.25)}},
    };
    for (const KindCase & c : cases) {
        scrapgpu_unary_params p{sizeof(p), c.kind};
        CHECK_OK(execute(SCRAPGPU_OP_UNARY, &s, {const_cast<float *>(x.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        for (int i = 0; i < 5; ++i) {
            CHECK_FEQ(y[i], c.expected[i], 1e-6);
        }
    }
}

void test_glu() {
    // split inputs
    {
        const scrapgpu_tensor_desc x = tdesc(SCRAPGPU_DTYPE_F32, 2);
        const scrapgpu_tensor_desc g = tdesc(SCRAPGPU_DTYPE_F32, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2);
        const std::vector<float> a = {1.0f, 2.0f};
        const std::vector<float> b = {3.0f, 4.0f};
        std::vector<float> y(2);
        scrapgpu_glu_params p{sizeof(p), SCRAPGPU_GLU_SWIGLU, 1, 0};
        const scrapgpu_tensor_desc in[2] = {x, g};
        CHECK_OK(execute(SCRAPGPU_OP_GLU, in, {const_cast<float *>(a.data()), const_cast<float *>(b.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK_FEQ(y[0], 1.0 / (1.0 + std::exp(-1.0)) * 3.0, 1e-6);
        CHECK_FEQ(y[1], 2.0 / (1.0 + std::exp(-2.0)) * 4.0, 1e-6);
    }
    // single input, halves in one row, swapped
    {
        const scrapgpu_tensor_desc x = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2);
        // swapped: x = src0[2..3] = (-1, 2), g = src0[0..1] = (3, 4); REGLU: relu(x) * g
        const std::vector<float> a = {3.0f, 4.0f, -1.0f, 2.0f};
        std::vector<float> y(2);
        scrapgpu_glu_params p{sizeof(p), SCRAPGPU_GLU_REGLU, 0, 1};
        CHECK_OK(execute(SCRAPGPU_OP_GLU, &x, {const_cast<float *>(a.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK_FEQ(y[0], 0.0, 1e-6);
        CHECK_FEQ(y[1], 8.0, 1e-6);
        // not swapped: x = (3, 4), g = (-1, 2)
        p.swapped = 0;
        CHECK_OK(execute(SCRAPGPU_OP_GLU, &x, {const_cast<float *>(a.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK_FEQ(y[0], -3.0, 1e-6);
        CHECK_FEQ(y[1], 8.0, 1e-6);
        // split_inputs contradicts the single input
        p.split_inputs = 1;
        CHECK(execute(SCRAPGPU_OP_GLU, &x, {const_cast<float *>(a.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)) ==
              SCRAPGPU_STATUS_INVALID_ARGUMENT);
    }
}

void test_norms() {
    const std::vector<float> x = {1, 2, 2, 4};
    const double sum_sq = 25.0;

    const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 4, 2);
    const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 4, 2);
    std::vector<float> xx(8);
    for (int row = 0; row < 2; ++row) {
        for (int i = 0; i < 4; ++i) {
            xx[row * 4 + i] = x[i];
        }
    }

    scrapgpu_rms_norm_params np{sizeof(np), 0.25f};
    std::vector<float> y(8);
    CHECK_OK(execute(SCRAPGPU_OP_RMS_NORM, &s, {xx.data()}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &np, sizeof(np)));
    const double scale = 1.0 / std::sqrt(sum_sq / 4 + 0.25);
    for (int i = 0; i < 4; ++i) {
        CHECK_FEQ(y[i], x[i] * scale, 1e-6);
        CHECK_FEQ(y[4 + i], x[i] * scale, 1e-6);
    }
    // missing params
    CHECK(execute(SCRAPGPU_OP_RMS_NORM, &s, {xx.data()}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}) ==
          SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // RMS_NORM_MUL: dst = rms_norm(x) * w
    const scrapgpu_tensor_desc w = tdesc(SCRAPGPU_DTYPE_F32, 4);
    const std::vector<float> wv = {1.0f, 0.5f, 2.0f, 0.0f};
    const scrapgpu_tensor_desc in[2] = {s, w};
    CHECK_OK(execute(SCRAPGPU_OP_RMS_NORM_MUL, in, {xx.data(), const_cast<float *>(wv.data())},
                     const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &np, sizeof(np)));
    for (int i = 0; i < 4; ++i) {
        CHECK_FEQ(y[i], x[i] * scale * wv[i], 1e-6);
        CHECK_FEQ(y[4 + i], x[i] * scale * wv[i], 1e-6);
    }

    // ADD_RMS_NORM_MUL: outputs are (sum, norm * w)
    const scrapgpu_tensor_desc a = tdesc(SCRAPGPU_DTYPE_F32, 4);
    const scrapgpu_tensor_desc b = tdesc(SCRAPGPU_DTYPE_F32, 4);
    const scrapgpu_tensor_desc s_out = tdesc(SCRAPGPU_DTYPE_F32, 4);
    const scrapgpu_tensor_desc y_out = tdesc(SCRAPGPU_DTYPE_F32, 4);
    const std::vector<float> av = {1, 2, 2, 4};
    const std::vector<float> bv = {1, 0, -1, 0};
    std::vector<float> sum(4);
    std::vector<float> yy(4);
    const scrapgpu_tensor_desc in3[3] = {a, b, w};
    scrapgpu_tensor_desc out2[2] = {s_out, y_out};
    CHECK_OK(execute(SCRAPGPU_OP_ADD_RMS_NORM_MUL, in3, {const_cast<float *>(av.data()), const_cast<float *>(bv.data()), const_cast<float *>(wv.data())},
                     out2, {sum.data(), yy.data()}, &np, sizeof(np)));
    const double sum_sq2 = 2 * 2 + 2 * 2 + 1 * 1 + 4 * 4;
    const double scale2 = 1.0 / std::sqrt(sum_sq2 / 4 + 0.25);
    for (int i = 0; i < 4; ++i) {
        CHECK_FEQ(sum[i], av[i] + bv[i], 1e-6);
        CHECK_FEQ(yy[i], (av[i] + bv[i]) * scale2 * wv[i], 1e-6);
    }
}

void test_rope() {
    const float theta_scale = std::pow(10000.0f, -2.0f / 4.0f);

    // NORMAL mode, n_dims = 4, row [1, 0, 1, 0], position 1
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc pos = tdesc(SCRAPGPU_DTYPE_I32, 1);
        const std::vector<float> x = {1, 0, 1, 0};
        const std::vector<int32_t> pv = {1};
        std::vector<float> y(4, -99.0f);
        scrapgpu_rope_params p{};
        p.struct_size = sizeof(p);
        p.mode = SCRAPGPU_ROPE_MODE_NORMAL;
        p.n_dims = 4;
        p.freq_base = 10000.0f;
        p.freq_scale = 1.0f;
        p.ext_factor = 0.0f;
        p.attn_factor = 1.0f;
        const scrapgpu_tensor_desc in[2] = {s, pos};
        CHECK_OK(execute(SCRAPGPU_OP_ROPE, in, {const_cast<float *>(x.data()), const_cast<int32_t *>(pv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK_FEQ(y[0], std::cos(1.0), 1e-5);
        CHECK_FEQ(y[1], std::sin(1.0), 1e-5);
        CHECK_FEQ(y[2], std::cos(0.01), 1e-5);
        CHECK_FEQ(y[3], std::sin(0.01), 1e-5);
    }
    // NORMAL mode with elements outside n_dims: ne0 = 6, n_dims = 4, the last pair is copied
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 6);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 6);
        const scrapgpu_tensor_desc pos = tdesc(SCRAPGPU_DTYPE_I32, 1);
        const std::vector<float> x = {1, 0, 1, 0, 7, 8};
        const std::vector<int32_t> pv = {1};
        std::vector<float> y(6, -99.0f);
        scrapgpu_rope_params p{};
        p.struct_size = sizeof(p);
        p.mode = SCRAPGPU_ROPE_MODE_NORMAL;
        p.n_dims = 4;
        p.freq_base = 10000.0f;
        p.freq_scale = 1.0f;
        p.attn_factor = 1.0f;
        const scrapgpu_tensor_desc in[2] = {s, pos};
        CHECK_OK(execute(SCRAPGPU_OP_ROPE, in, {const_cast<float *>(x.data()), const_cast<int32_t *>(pv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK_FEQ(y[4], 7, 1e-6);
        CHECK_FEQ(y[5], 8, 1e-6);
    }
    // NEOX mode: e0 = p, e1 = p + n_dims/2
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc pos = tdesc(SCRAPGPU_DTYPE_I32, 1);
        const std::vector<float> x = {1, 2, 3, 4};
        const std::vector<int32_t> pv = {2};
        std::vector<float> y(4, -99.0f);
        scrapgpu_rope_params p{};
        p.struct_size = sizeof(p);
        p.mode = SCRAPGPU_ROPE_MODE_NEOX;
        p.n_dims = 4;
        p.freq_base = 10000.0f;
        p.freq_scale = 1.0f;
        p.attn_factor = 1.0f;
        const scrapgpu_tensor_desc in[2] = {s, pos};
        CHECK_OK(execute(SCRAPGPU_OP_ROPE, in, {const_cast<float *>(x.data()), const_cast<int32_t *>(pv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        const double t0 = 2.0;
        const double t1 = 2.0 * theta_scale;
        CHECK_FEQ(y[0], x[0] * std::cos(t0) - x[2] * std::sin(t0), 1e-5);
        CHECK_FEQ(y[2], x[0] * std::sin(t0) + x[2] * std::cos(t0), 1e-5);
        CHECK_FEQ(y[1], x[1] * std::cos(t1) - x[3] * std::sin(t1), 1e-5);
        CHECK_FEQ(y[3], x[1] * std::sin(t1) + x[3] * std::cos(t1), 1e-5);
    }
    // freq factors as a third input: theta_extrap = theta / ff
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc pos = tdesc(SCRAPGPU_DTYPE_I32, 1);
        const scrapgpu_tensor_desc ff = tdesc(SCRAPGPU_DTYPE_F32, 2);
        const std::vector<float> x = {1, 0, 1, 0};
        const std::vector<int32_t> pv = {2};
        const std::vector<float> ffv = {2.0f, 4.0f};
        std::vector<float> y(4, -99.0f);
        scrapgpu_rope_params p{};
        p.struct_size = sizeof(p);
        p.mode = SCRAPGPU_ROPE_MODE_NORMAL;
        p.n_dims = 4;
        p.freq_base = 10000.0f;
        p.freq_scale = 1.0f;
        p.attn_factor = 1.0f;
        const scrapgpu_tensor_desc in[3] = {s, pos, ff};
        CHECK_OK(execute(SCRAPGPU_OP_ROPE, in, {const_cast<float *>(x.data()), const_cast<int32_t *>(pv.data()), const_cast<float *>(ffv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK_FEQ(y[0], std::cos(1.0), 1e-5);
        CHECK_FEQ(y[1], std::sin(1.0), 1e-5);
        CHECK_FEQ(y[2], std::cos(0.005), 1e-5);
        CHECK_FEQ(y[3], std::sin(0.005), 1e-5);
    }
    // F16 data
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F16, 4);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F16, 4);
        const scrapgpu_tensor_desc pos = tdesc(SCRAPGPU_DTYPE_I32, 1);
        const std::vector<uint16_t> x = {0x3C00, 0x0000, 0x3C00, 0x0000}; // 1, 0, 1, 0
        const std::vector<int32_t> pv = {1};
        std::vector<uint16_t> y(4, 0xFFFF);
        scrapgpu_rope_params p{};
        p.struct_size = sizeof(p);
        p.mode = SCRAPGPU_ROPE_MODE_NORMAL;
        p.n_dims = 4;
        p.freq_base = 10000.0f;
        p.freq_scale = 1.0f;
        p.attn_factor = 1.0f;
        const scrapgpu_tensor_desc in[2] = {s, pos};
        CHECK_OK(execute(SCRAPGPU_OP_ROPE, in, {const_cast<uint16_t *>(x.data()), const_cast<int32_t *>(pv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        // expected halves built through CPY F32 -> F16
        const scrapgpu_tensor_desc fs = tdesc(SCRAPGPU_DTYPE_F32, 4);
        const scrapgpu_tensor_desc hd = tdesc(SCRAPGPU_DTYPE_F16, 4);
        const std::vector<float> ev = {static_cast<float>(std::cos(1.0)), static_cast<float>(std::sin(1.0)),
                                       static_cast<float>(std::cos(0.01)), static_cast<float>(std::sin(0.01))};
        std::vector<uint16_t> eh(4);
        scrapgpu_cpy_params cp{sizeof(cp), 0};
        CHECK_OK(execute(SCRAPGPU_OP_CPY, &fs, {const_cast<float *>(ev.data())}, const_cast<scrapgpu_tensor_desc *>(&hd), {eh.data()}, &cp, sizeof(cp)));
        for (int i = 0; i < 4; ++i) {
            CHECK(y[i] == eh[i]);
        }
    }
}

void test_soft_max() {
    const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 3, 2, 2); // rows over i1, broadcast over i2
    const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 3, 2, 2);
    std::vector<float> x(3 * 2 * 2);
    for (int i = 0; i < 12; ++i) {
        x[i] = static_cast<float>(i % 3);
    }
    // mask: one row per i1, broadcast over i2/i3
    scrapgpu_tensor_desc m = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
    const std::vector<float> mv = {0.0f, -1.0f, 1.0f, 0.5f, -0.5f, 2.0f};
    std::vector<float> y(12, -99.0f);
    scrapgpu_soft_max_params p{sizeof(p), 0.5f, nullptr};
    const scrapgpu_tensor_desc in[2] = {s, m};
    CHECK_OK(execute(SCRAPGPU_OP_SOFT_MAX, in, {x.data(), const_cast<float *>(mv.data())},
                     const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
    for (int i2 = 0; i2 < 2; ++i2) {
        for (int i1 = 0; i1 < 2; ++i1) {
            double v[3];
            double mx = -1e300;
            for (int i = 0; i < 3; ++i) {
                v[i] = x[(i2 * 2 + i1) * 3 + i] * 0.5 + mv[i1 * 3 + i];
                mx = std::max(mx, v[i]);
            }
            double sum = 0;
            for (int i = 0; i < 3; ++i) {
                v[i] = std::exp(v[i] - mx);
                sum += v[i];
            }
            for (int i = 0; i < 3; ++i) {
                CHECK_FEQ(y[(i2 * 2 + i1) * 3 + i], v[i] / sum, 1e-6);
            }
        }
    }
}

void test_rows_ops() {
    // GET_ROWS on F32 with a row-indexed source and an offset view
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 3, 4);
        const scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I32, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
        std::vector<float> a(12);
        for (int i = 0; i < 12; ++i) {
            a[i] = static_cast<float>(i);
        }
        const std::vector<int32_t> idx = {2, 0};
        std::vector<float> y(6, -1.0f);
        scrapgpu_get_rows_params p{sizeof(p), 0};
        const scrapgpu_tensor_desc in[2] = {s, ids};
        CHECK_OK(execute(SCRAPGPU_OP_GET_ROWS, in, {a.data(), const_cast<int32_t *>(idx.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        for (int i = 0; i < 3; ++i) {
            CHECK_FEQ(y[i], a[6 + i], 1e-6);
            CHECK_FEQ(y[3 + i], a[i], 1e-6);
        }
        // out of range
        const std::vector<int32_t> bad = {4};
        const scrapgpu_tensor_desc ids1 = tdesc(SCRAPGPU_DTYPE_I32, 1);
        const scrapgpu_tensor_desc d1 = tdesc(SCRAPGPU_DTYPE_F32, 3);
        const scrapgpu_tensor_desc in2[2] = {s, ids1};
        CHECK(execute(SCRAPGPU_OP_GET_ROWS, in2, {a.data(), const_cast<int32_t *>(bad.data())},
                      const_cast<scrapgpu_tensor_desc *>(&d1), {y.data()}, &p, sizeof(p)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    }
    // GET_ROWS on Q4_0
    {
        const int64_t kN = 64;
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_Q4_0, kN, 3);
        const scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I32, 3);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, kN, 3);
        std::vector<float> a(kN * 3);
        for (int i = 0; i < kN * 3; ++i) {
            a[i] = std::sin(0.37f * i) * 3.0f;
        }
        std::vector<uint8_t> q(s.nb[1] * 3);
        CHECK_OK(scrapgpu_ref_quantize(SCRAPGPU_DTYPE_Q4_0, a.data(), q.data(), kN * 3));
        std::vector<float> dq(kN * 3);
        CHECK_OK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q4_0, q.data(), dq.data(), kN * 3));
        const std::vector<int32_t> idx = {2, 2, 0};
        std::vector<float> y(kN * 3, -1.0f);
        scrapgpu_get_rows_params p{sizeof(p), 0};
        const scrapgpu_tensor_desc in[2] = {s, ids};
        CHECK_OK(execute(SCRAPGPU_OP_GET_ROWS, in, {q.data(), const_cast<int32_t *>(idx.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        for (int64_t r = 0; r < 3; ++r) {
            for (int64_t i = 0; i < kN; ++i) {
                CHECK_FEQ(y[r * kN + i], dq[idx[r] * kN + i], 1e-9);
            }
        }
    }
    // SET_ROWS F32 -> F16 with I64 indices; untouched rows keep their content
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 2, 2);
        const scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I64, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F16, 2, 4);
        const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        const std::vector<int64_t> idx = {3, 1};
        std::vector<uint16_t> y(8, 0x3C00); // 1.0
        scrapgpu_set_rows_params p{sizeof(p), 1};
        const scrapgpu_tensor_desc in[2] = {s, ids};
        CHECK_OK(execute(SCRAPGPU_OP_SET_ROWS, in, {const_cast<float *>(a.data()), const_cast<int64_t *>(idx.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        const scrapgpu_tensor_desc fs = tdesc(SCRAPGPU_DTYPE_F32, 2);
        const scrapgpu_tensor_desc f16 = tdesc(SCRAPGPU_DTYPE_F16, 2);
        const std::vector<float> ev1 = {3.0f, 4.0f};
        const std::vector<float> ev3 = {1.0f, 2.0f};
        std::vector<uint16_t> e1(2);
        std::vector<uint16_t> e3(2);
        scrapgpu_cpy_params cp{sizeof(cp), 0};
        CHECK_OK(execute(SCRAPGPU_OP_CPY, &fs, {const_cast<float *>(ev1.data())}, const_cast<scrapgpu_tensor_desc *>(&f16), {e1.data()}, &cp, sizeof(cp)));
        CHECK_OK(execute(SCRAPGPU_OP_CPY, &fs, {const_cast<float *>(ev3.data())}, const_cast<scrapgpu_tensor_desc *>(&f16), {e3.data()}, &cp, sizeof(cp)));
        CHECK(y[0] == 0x3C00 && y[1] == 0x3C00); // row 0 untouched
        CHECK(y[2] == e1[0] && y[3] == e1[1]);    // row 1 = second source row
        CHECK(y[4] == 0x3C00 && y[5] == 0x3C00); // row 2 untouched
        CHECK(y[6] == e3[0] && y[7] == e3[1]);    // row 3 = first source row
    }
    // audit H-4: SET_ROWS must reject the indices it cannot address before touching the images
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 2, 2);
        const scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I64, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 4);
        const std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> y(8, 1.0f);
        scrapgpu_set_rows_params p{sizeof(p), 1};

        // an out-of-range index value would write past the image (ASan: heap-buffer-overflow)
        const std::vector<int64_t> too_far = {3, 4};
        const scrapgpu_tensor_desc in[2] = {s, ids};
        CHECK(execute(SCRAPGPU_OP_SET_ROWS, in, {const_cast<float *>(a.data()), const_cast<int64_t *>(too_far.data())},
                      const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

        // a negative index would write before the image
        const std::vector<int64_t> neg = {3, -1};
        CHECK(execute(SCRAPGPU_OP_SET_ROWS, in, {const_cast<float *>(a.data()), const_cast<int64_t *>(neg.data())},
                      const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

        // an index tensor shorter than the source row count would be read past its declared
        // extent: the descriptor covers 2 indices but 3 source rows are scattered (with a
        // shorter host image ASan reports a heap-buffer-overflow on the read)
        const scrapgpu_tensor_desc s3 = tdesc(SCRAPGPU_DTYPE_F32, 2, 3);
        const std::vector<float> a3(6, 1.0f);
        const std::vector<int64_t> idx3 = {3, 0, 1};
        const scrapgpu_tensor_desc in3[2] = {s3, ids};
        CHECK(execute(SCRAPGPU_OP_SET_ROWS, in3, {const_cast<float *>(a3.data()), const_cast<int64_t *>(idx3.data())},
                      const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    }
}

void test_cpy() {
    // reshape: 2x3 (row major) -> 3x2, elements keep their linear order
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 3);
        const std::vector<float> a = {1, 2, 3, 4, 5, 6};
        std::vector<float> y(6, -1.0f);
        CHECK_OK(execute(SCRAPGPU_OP_DUP, &s, {const_cast<float *>(a.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}));
        CHECK(y[0] == 1 && y[1] == 2 && y[2] == 3 && y[3] == 4 && y[4] == 5 && y[5] == 6);
    }
    // F32 -> F16 -> F32 round trip and exact half bit patterns
    {
        const std::vector<float> x = {1.0f, -0.0f, 65504.0f, 65520.0f, 65512.0f,
                                      5.960464477539063e-08f, 2.9802322387695312e-08f, 0.0f};
        const std::vector<uint16_t> expect = {0x3C00, 0x8000, 0x7BFF, 0x7C00, 0x7BFF, 0x0001, 0x0000, 0x0000};
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 8);
        const scrapgpu_tensor_desc h = tdesc(SCRAPGPU_DTYPE_F16, 8);
        const scrapgpu_tensor_desc back = tdesc(SCRAPGPU_DTYPE_F32, 8);
        std::vector<uint16_t> hh(8);
        std::vector<float> y(8);
        scrapgpu_cpy_params cp{sizeof(cp), 0};
        CHECK_OK(execute(SCRAPGPU_OP_CPY, &s, {const_cast<float *>(x.data())}, const_cast<scrapgpu_tensor_desc *>(&h), {hh.data()}, &cp, sizeof(cp)));
        for (int i = 0; i < 8; ++i) {
            CHECK(hh[i] == expect[i]);
        }
        CHECK_OK(execute(SCRAPGPU_OP_CONT, &h, {hh.data()}, const_cast<scrapgpu_tensor_desc *>(&back), {y.data()}));
        for (int i = 0; i < 8; ++i) {
            // -0.0 must keep its sign, 65520.0f overflows to +inf
            if (std::signbit(x[i])) {
                CHECK(std::signbit(y[i]));
            }
            if (i == 3) {
                CHECK(std::isinf(y[i]));
            } else if (i == 4) {
                CHECK_FEQ(y[i], 65504.0f, 1e-6); // 65512.0f saturates to max finite F16 (0x7BFF)
            } else {
                CHECK_FEQ(y[i], x[i], 1e-6 * std::max(1.0, std::fabs(static_cast<double>(x[i]))));
            }
        }
    }
}

void test_mul_mat() {
    // F32 MUL_MAT: src0 [K=3,M=2] * src1 [K=3,N=2] -> dst [M=2,N=2] (ggml layout)
    {
        const scrapgpu_tensor_desc a = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
        const scrapgpu_tensor_desc b = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 2);
        const std::vector<float> av = {1, 2, 3, 4, 5, 6};
        const std::vector<float> bv = {7, 9, 11, 8, 10, 12};
        std::vector<float> y(4, -1.0f);
        scrapgpu_mul_mat_params p{sizeof(p), 1};
        const scrapgpu_tensor_desc in[2] = {a, b};
        CHECK_OK(execute(SCRAPGPU_OP_MUL_MAT, in, {const_cast<float *>(av.data()), const_cast<float *>(bv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK_FEQ(y[0], 58, 1e-5);
        CHECK_FEQ(y[1], 139, 1e-5);
        CHECK_FEQ(y[2], 64, 1e-5);
        CHECK_FEQ(y[3], 154, 1e-5);
    }
    // broadcast over dim 2: src1 has two slices, src0 one
    {
        const scrapgpu_tensor_desc a = tdesc(SCRAPGPU_DTYPE_F32, 2, 2, 1);
        const scrapgpu_tensor_desc b = tdesc(SCRAPGPU_DTYPE_F32, 2, 2, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 2, 2);
        const std::vector<float> av = {1, 2, 3, 4};
        const std::vector<float> bv = {1, 0, 0, 1, 5, 6, 7, 8};
        std::vector<float> y(8, -1.0f);
        scrapgpu_mul_mat_params p{sizeof(p), 1};
        const scrapgpu_tensor_desc in[2] = {a, b};
        CHECK_OK(execute(SCRAPGPU_OP_MUL_MAT, in, {const_cast<float *>(av.data()), const_cast<float *>(bv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        for (int s = 0; s < 2; ++s) {
            CHECK_FEQ(y[s * 4 + 0], av[0] * bv[s * 4 + 0] + av[1] * bv[s * 4 + 1], 1e-5);
            CHECK_FEQ(y[s * 4 + 3], av[2] * bv[s * 4 + 2] + av[3] * bv[s * 4 + 3], 1e-5);
        }
        // the same shapes with allow_broadcast == 0 are refused
        scrapgpu_mul_mat_params p0{sizeof(p0), 0};
        CHECK(execute(SCRAPGPU_OP_MUL_MAT, in, {const_cast<float *>(av.data()), const_cast<float *>(bv.data())},
                      const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p0, sizeof(p0)) == SCRAPGPU_STATUS_UNSUPPORTED);
    }
    // Q8_0 weights: the reference product of the dequantized values
    {
        const int64_t K = 96;
        const int64_t M = 5;
        const scrapgpu_tensor_desc a = tdesc(SCRAPGPU_DTYPE_Q8_0, K, M);
        const scrapgpu_tensor_desc b = tdesc(SCRAPGPU_DTYPE_F32, K, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, M, 2);
        std::vector<float> av(K * M);
        for (int i = 0; i < K * M; ++i) {
            av[i] = std::sin(0.71f * i) * 2.5f;
        }
        std::vector<float> bv(K * 2);
        for (int i = 0; i < K * 2; ++i) {
            bv[i] = std::cos(0.13f * i);
        }
        std::vector<uint8_t> q(a.nb[1] * M);
        CHECK_OK(scrapgpu_ref_quantize(SCRAPGPU_DTYPE_Q8_0, av.data(), q.data(), K * M));
        std::vector<float> dq(K * M);
        CHECK_OK(scrapgpu_ref_dequantize(SCRAPGPU_DTYPE_Q8_0, q.data(), dq.data(), K * M));
        std::vector<float> y(M * 2, -1.0f);
        scrapgpu_mul_mat_params p{sizeof(p), 1};
        const scrapgpu_tensor_desc in[2] = {a, b};
        CHECK_OK(execute(SCRAPGPU_OP_MUL_MAT, in, {q.data(), bv.data()}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < 2; ++n) {
                double ref = 0;
                for (int64_t k = 0; k < K; ++k) {
                    ref += static_cast<double>(dq[m * K + k]) * bv[n * K + k];
                }
                CHECK_FEQ(y[m + n * M], ref, 1e-3);
            }
        }
    }
}

void test_mul_mat_id_and_add_id() {
    // two experts, K = 2, M = 2; src1 shares one column per token (ne11 == 1)
    {
        const scrapgpu_tensor_desc s0 = tdesc(SCRAPGPU_DTYPE_F32, 2, 2, 2); // [K, M, E]
        const scrapgpu_tensor_desc s1 = tdesc(SCRAPGPU_DTYPE_F32, 2, 1, 2); // [K, 1, T]
        const scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I32, 2, 2);   // [n_used, T]
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 2, 2);  // [M, n_used, T]
        // expert 0: [[1, 2], [3, 4]]; expert 1: [[10, 20], [30, 40]]
        const std::vector<float> w = {1, 2, 3, 4, 10, 20, 30, 40};
        // token 0 column (1, 1); token 1 column (2, 0)
        const std::vector<float> x = {1, 1, 2, 0};
        const std::vector<int32_t> id = {0, 1, 1, 0};
        std::vector<float> y(8, -1.0f);
        scrapgpu_mul_mat_id_params p{sizeof(p), 2, 0};
        const scrapgpu_tensor_desc in[3] = {s0, s1, ids};
        CHECK_OK(execute(SCRAPGPU_OP_MUL_MAT_ID, in, {const_cast<float *>(w.data()), const_cast<float *>(x.data()), const_cast<int32_t *>(id.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        // dst[:, i, t] = expert id[i, t] @ x[:, 0, t]
        const float expect[2][2][2] = {
            // t = 0: x = (1, 1); experts 0, 1
            {{1 * 1 + 2 * 1, 3 * 1 + 4 * 1}, {10 * 1 + 20 * 1, 30 * 1 + 40 * 1}},
            // t = 1: x = (2, 0); experts 1, 0
            {{10 * 2 + 20 * 0, 30 * 2 + 40 * 0}, {1 * 2 + 2 * 0, 3 * 2 + 4 * 0}},
        };
        for (int t = 0; t < 2; ++t) {
            for (int i = 0; i < 2; ++i) {
                for (int m = 0; m < 2; ++m) {
                    CHECK_FEQ(y[(t * 2 + i) * 2 + m], expect[t][i][m], 1e-5);
                }
            }
        }
    }
    // ADD_ID: dst = src0 + bias[ids]
    {
        const scrapgpu_tensor_desc s0 = tdesc(SCRAPGPU_DTYPE_F32, 2, 2); // [d, n]
        const scrapgpu_tensor_desc s1 = tdesc(SCRAPGPU_DTYPE_F32, 2, 3); // [d, E]
        const scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I32, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 2);
        const std::vector<float> a = {1, 2, 3, 4};
        const std::vector<float> bias = {10, 20, 30, 40, 50, 60};
        const std::vector<int32_t> id = {2, 0};
        std::vector<float> y(4, -1.0f);
        const scrapgpu_tensor_desc in[3] = {s0, s1, ids};
        CHECK_OK(execute(SCRAPGPU_OP_ADD_ID, in, {const_cast<float *>(a.data()), const_cast<float *>(bias.data()), const_cast<int32_t *>(id.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}));
        CHECK_FEQ(y[0], 51, 1e-6);
        CHECK_FEQ(y[1], 62, 1e-6);
        CHECK_FEQ(y[2], 13, 1e-6);
        CHECK_FEQ(y[3], 24, 1e-6);
        // out-of-range expert ids are clamped (reference / CL11 kernel semantics)
        const std::vector<int32_t> bad = {7, 0};
        CHECK_OK(execute(SCRAPGPU_OP_ADD_ID, in, {const_cast<float *>(a.data()), const_cast<float *>(bias.data()), const_cast<int32_t *>(bad.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}));
        CHECK_FEQ(y[0], 51, 1e-6);
    }
}

void test_misc_ops() {
    // SSM_CONV: hand-checked (see the kernel k_ssm_conv)
    {
        const scrapgpu_tensor_desc s0 = tdesc(SCRAPGPU_DTYPE_F32, 5, 2); // [d_conv - 1 + n_t, d_inner]
        const scrapgpu_tensor_desc s1 = tdesc(SCRAPGPU_DTYPE_F32, 3, 2); // [d_conv, d_inner]
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 3);  // [d_inner, n_t]
        const std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};    // [time][ch]
        const std::vector<float> c = {1, 0, 0, 1, 1, 1};                 // [j][ch]
        std::vector<float> y(6, -1.0f);
        scrapgpu_ssm_conv_params p{sizeof(p), 3};
        const scrapgpu_tensor_desc in[2] = {s0, s1};
        CHECK_OK(execute(SCRAPGPU_OP_SSM_CONV, in, {const_cast<float *>(a.data()), const_cast<float *>(c.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        // dst[i0, i1] = sum_j src0[i1 + j, i0] * src1[j, i0] (see k_ssm_conv)
        CHECK_FEQ(y[0], 1, 1e-6);   // ch0, t0
        CHECK_FEQ(y[1], 21, 1e-6);  // ch1, t0
        CHECK_FEQ(y[2], 2, 1e-6);   // ch0, t1
        CHECK_FEQ(y[3], 24, 1e-6);  // ch1, t1
        CHECK_FEQ(y[4], 3, 1e-6);   // ch0, t2
        CHECK_FEQ(y[5], 27, 1e-6);  // ch1, t2
    }
    // CONCAT along dim 0 and dim 1
    {
        const scrapgpu_tensor_desc a = tdesc(SCRAPGPU_DTYPE_F32, 2, 2);
        const scrapgpu_tensor_desc b = tdesc(SCRAPGPU_DTYPE_F32, 1, 2);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
        const std::vector<float> av = {1, 2, 3, 4};
        const std::vector<float> bv = {5, 6};
        std::vector<float> y(6, -1.0f);
        scrapgpu_concat_params p{sizeof(p), 0};
        const scrapgpu_tensor_desc in[2] = {a, b};
        CHECK_OK(execute(SCRAPGPU_OP_CONCAT, in, {const_cast<float *>(av.data()), const_cast<float *>(bv.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &p, sizeof(p)));
        CHECK(y[0] == 1 && y[1] == 2 && y[2] == 5 && y[3] == 3 && y[4] == 4 && y[5] == 6);

        const scrapgpu_tensor_desc b1 = tdesc(SCRAPGPU_DTYPE_F32, 2, 1);
        const scrapgpu_tensor_desc d1 = tdesc(SCRAPGPU_DTYPE_F32, 2, 3);
        const std::vector<float> bv1 = {7, 8};
        p.dim = 1;
        const scrapgpu_tensor_desc in1[2] = {a, b1};
        CHECK_OK(execute(SCRAPGPU_OP_CONCAT, in1, {const_cast<float *>(av.data()), const_cast<float *>(bv1.data())},
                         const_cast<scrapgpu_tensor_desc *>(&d1), {y.data()}, &p, sizeof(p)));
        CHECK(y[0] == 1 && y[1] == 2 && y[2] == 3 && y[3] == 4 && y[4] == 7 && y[5] == 8);
    }
    // PAD: zero padding and circular wrap (odd row length)
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 5);
        const scrapgpu_tensor_desc dz = tdesc(SCRAPGPU_DTYPE_F32, 8);
        const std::vector<float> a = {1, 2, 3, 4, 5};
        std::vector<float> y(8, -1.0f);
        scrapgpu_pad_params p{};
        p.struct_size = sizeof(p);
        p.before[0] = 1;
        p.after[0] = 2;
        CHECK_OK(execute(SCRAPGPU_OP_PAD, &s, {const_cast<float *>(a.data())}, const_cast<scrapgpu_tensor_desc *>(&dz), {y.data()}, &p, sizeof(p)));
        CHECK(y[0] == 0 && y[1] == 1 && y[2] == 2 && y[3] == 3 && y[4] == 4 && y[5] == 5 && y[6] == 0 && y[7] == 0);

        const scrapgpu_tensor_desc dc = tdesc(SCRAPGPU_DTYPE_F32, 7);
        p.after[0] = 1;
        p.circular = 1;
        CHECK_OK(execute(SCRAPGPU_OP_PAD, &s, {const_cast<float *>(a.data())}, const_cast<scrapgpu_tensor_desc *>(&dc), {y.data()}, &p, sizeof(p)));
        CHECK(y[0] == 5 && y[1] == 1 && y[2] == 2 && y[3] == 3 && y[4] == 4 && y[5] == 5 && y[6] == 1);
    }
    // SUM_ROWS over rows with a padded stride
    {
        scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 3, 2);
        s.nb[1] = 4 * 4;
        s.nb[2] = s.nb[1] * 2;
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 1, 2);
        std::vector<float> a(8, 0.0f);
        a[0] = 1; a[1] = 2; a[2] = 3;
        a[4] = 10; a[5] = 20; a[6] = 30;
        std::vector<float> y(2, -1.0f);
        CHECK_OK(execute(SCRAPGPU_OP_SUM_ROWS, &s, {a.data()}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}));
        CHECK_FEQ(y[0], 6, 1e-6);
        CHECK_FEQ(y[1], 60, 1e-6);
    }
    // ARGSORT with ties (smaller index first) and TOP_K
    {
        const scrapgpu_tensor_desc s = tdesc(SCRAPGPU_DTYPE_F32, 5);
        const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_I32, 5);
        const std::vector<float> keys = {3.0f, 1.0f, 2.0f, 1.0f, 2.0f};
        std::vector<int32_t> y(5, -1);
        scrapgpu_argsort_params asc{sizeof(asc), SCRAPGPU_ARGSORT_ASC, 0};
        CHECK_OK(execute(SCRAPGPU_OP_ARGSORT, &s, {const_cast<float *>(keys.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &asc, sizeof(asc)));
        CHECK(y[0] == 1 && y[1] == 3 && y[2] == 2 && y[3] == 4 && y[4] == 0);
        scrapgpu_argsort_params desc{sizeof(desc), SCRAPGPU_ARGSORT_DESC, 0};
        CHECK_OK(execute(SCRAPGPU_OP_ARGSORT, &s, {const_cast<float *>(keys.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}, &desc, sizeof(desc)));
        CHECK(y[0] == 0 && y[1] == 2 && y[2] == 4 && y[3] == 1 && y[4] == 3);

        const scrapgpu_tensor_desc dk = tdesc(SCRAPGPU_DTYPE_I32, 2);
        std::vector<int32_t> yk(2, -1);
        scrapgpu_argsort_params top{sizeof(top), SCRAPGPU_ARGSORT_ASC, 2};
        CHECK_OK(execute(SCRAPGPU_OP_TOP_K, &s, {const_cast<float *>(keys.data())}, const_cast<scrapgpu_tensor_desc *>(&dk), {yk.data()}, &top, sizeof(top)));
        CHECK(yk[0] == 0 && yk[1] == 2);
    }
}

void test_errors() {
    CHECK(scrapgpu_ref_execute(nullptr, nullptr, nullptr) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    const scrapgpu_tensor_desc a = tdesc(SCRAPGPU_DTYPE_F32, 2, 2);
    const scrapgpu_tensor_desc b = tdesc(SCRAPGPU_DTYPE_F32, 3, 1);
    const scrapgpu_tensor_desc d = tdesc(SCRAPGPU_DTYPE_F32, 2, 2);
    const std::vector<float> av(4, 1.0f);
    const std::vector<float> bv(3, 1.0f);
    std::vector<float> y(4);
    const scrapgpu_tensor_desc in[2] = {a, b};
    CHECK(execute(SCRAPGPU_OP_ADD, in, {const_cast<float *>(av.data()), const_cast<float *>(bv.data())},
                  const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // quantized tensor in the R32 layout is refused
    {
        scrapgpu_tensor_desc q = tdesc(SCRAPGPU_DTYPE_Q4_0, 32, 2);
        q.layout = SCRAPGPU_LAYOUT_R32;
        const scrapgpu_tensor_desc ids = tdesc(SCRAPGPU_DTYPE_I32, 1);
        const scrapgpu_tensor_desc o = tdesc(SCRAPGPU_DTYPE_F32, 32);
        std::vector<uint8_t> qq(q.nb[1] * 2, 0);
        std::vector<int32_t> idx = {0};
        std::vector<float> yo(32);
        scrapgpu_get_rows_params p{sizeof(p), 0};
        const scrapgpu_tensor_desc in2[2] = {q, ids};
        CHECK(execute(SCRAPGPU_OP_GET_ROWS, in2, {qq.data(), idx.data()}, const_cast<scrapgpu_tensor_desc *>(&o), {yo.data()}, &p, sizeof(p)) ==
              SCRAPGPU_STATUS_UNSUPPORTED);
    }
    // null data pointer
    CHECK(execute(SCRAPGPU_OP_SQR, &a, {nullptr}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    // tensor desc without struct_size
    {
        scrapgpu_tensor_desc bad = tdesc(SCRAPGPU_DTYPE_F32, 2);
        bad.struct_size = 8;
        const std::vector<float> x(2, 1.0f);
        CHECK(execute(SCRAPGPU_OP_SQR, &bad, {const_cast<float *>(x.data())}, const_cast<scrapgpu_tensor_desc *>(&d), {y.data()}) ==
              SCRAPGPU_STATUS_INVALID_ARGUMENT);
    }
}

} // namespace

int main() {
    test_binary_ops();
    test_unary_family();
    test_glu();
    test_norms();
    test_rope();
    test_soft_max();
    test_rows_ops();
    test_cpy();
    test_mul_mat();
    test_mul_mat_id_and_add_id();
    test_misc_ops();
    test_errors();
    return 0;
}
