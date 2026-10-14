// XeStrata decode-op fixtures: correct head pairing/state layout, causal conv,
// L2 (not RMS) normalization, and captured-QSA step metadata.
//
// No GPU or XeStrata installation required. CPU backend tests the same C ABI
// used by future OpenCL/Vulkan/SYCL implementations.
#include "scrapgpu/api.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/cpu.h"
#include "scrapgpu/device.h"
#include "scrapgpu/op.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); std::exit(1); } } while (0)
#define OK(x) do { const auto s = (x); if (s != SCRAPGPU_STATUS_OK) { \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, scrapgpu_status_string(s), scrapgpu_last_error()); \
    std::exit(1); } } while (0)

static void near(float x, float expected, float tol = 1e-5f) {
    CHECK(std::fabs(x - expected) <= tol);
}

static scrapgpu_tensor_desc tensor(scrapgpu_dtype type, int64_t a, int64_t b = 1, int64_t c = 1) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = type;
    t.ne[0] = a; t.ne[1] = b; t.ne[2] = c; t.ne[3] = 1;
    t.nb[0] = scrapgpu_dtype_type_size(type);
    t.nb[1] = t.nb[0] * static_cast<uint64_t>(a);
    t.nb[2] = t.nb[1] * static_cast<uint64_t>(b);
    t.nb[3] = t.nb[2] * static_cast<uint64_t>(c);
    t.layout = SCRAPGPU_LAYOUT_NATURAL;
    return t;
}
static scrapgpu_op_desc op(scrapgpu_op which, const scrapgpu_tensor_desc * in, size_t nin,
                          scrapgpu_tensor_desc * out, size_t nout,
                          const void * params = nullptr, size_t nparams = 0) {
    scrapgpu_op_desc d{};
    d.struct_size = sizeof(d);
    d.op = which;
    d.inputs = in; d.input_count = nin;
    d.outputs = out; d.output_count = nout;
    d.params = params; d.params_size = nparams;
    return d;
}
static void ref(const scrapgpu_op_desc & d, void * const * in, void * const * out) {
    OK(scrapgpu_ref_execute(&d, in, out));
}

struct Cpu {
    scrapgpu_device_t * device = nullptr;
    scrapgpu_stream_t * stream = nullptr;
    std::vector<scrapgpu_buffer_t *> buffers;
    Cpu() {
        OK(scrapgpu_cpu_register());
        for (size_t i = 0; i < scrapgpu_get_device_count(); ++i) {
            scrapgpu_device_t * dev = nullptr;
            OK(scrapgpu_get_device(i, &dev));
            scrapgpu_device_info info{};
            OK(scrapgpu_device_info_init(&info, sizeof(info)));
            OK(scrapgpu_device_get_info(dev, &info));
            if (info.backend_name && std::strcmp(info.backend_name, "cpu") == 0) { device = dev; break; }
        }
        CHECK(device);
        OK(scrapgpu_stream_create(device, &stream));
    }
    ~Cpu() {
        for (scrapgpu_buffer_t * b : buffers) OK(scrapgpu_buffer_free(b));
        OK(scrapgpu_stream_free(stream));
    }
    template<class T> scrapgpu_tensor_desc upload(scrapgpu_tensor_desc d, const std::vector<T> & data) {
        scrapgpu_buffer_t * b = nullptr;
        OK(scrapgpu_buffer_alloc(device, data.size() * sizeof(T), &b));
        buffers.push_back(b);
        d.buffer = b;
        OK(scrapgpu_buffer_write(stream, b, 0, data.data(), data.size() * sizeof(T), 1));
        return d;
    }
    template<class T> std::vector<T> read(const scrapgpu_tensor_desc & d, size_t n) {
        std::vector<T> result(n);
        OK(scrapgpu_buffer_read(stream, d.buffer, 0, result.data(), result.size() * sizeof(T), 1));
        return result;
    }
    void run(const scrapgpu_op_desc & d) {
        const char * reason = nullptr;
        OK(scrapgpu_op_supported(device, &d, &reason));
        OK(scrapgpu_op_execute(stream, &d));
    }
};

static void gdn_step() {
    // S=2, Hk=2, Hv=4 forces modulo head mapping: heads 0,2 use key 0.
    const scrapgpu_tensor_desc base[6] = {
        tensor(SCRAPGPU_DTYPE_F32, 2, 4, 2), // state[j,h,i], i slowest
        tensor(SCRAPGPU_DTYPE_F32, 2, 2),    // q
        tensor(SCRAPGPU_DTYPE_F32, 2, 2),    // k
        tensor(SCRAPGPU_DTYPE_F32, 2, 4),    // v
        tensor(SCRAPGPU_DTYPE_F32, 4),       // pre-exp gate
        tensor(SCRAPGPU_DTYPE_F32, 4),       // sigmoid(beta)
    };
    const scrapgpu_tensor_desc outbase[2] = {base[0], base[3]};
    std::vector<float> state(16, 0.0f);
    std::vector<float> q = {1, 0, 0, 1}, k = {1, 0, 0, 1};
    std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> gate(4, 0.0f);
    std::vector<float> beta = {1, 0.5f, 0.25f, 0.75f};
    std::vector<float> next(16, -1.0f), y(8, -1.0f);
    scrapgpu_tensor_desc outs[2] = {outbase[0], outbase[1]};
    auto d = op(SCRAPGPU_OP_GDN_STEP, base, 6, outs, 2);
    void * in_host[6] = {state.data(), q.data(), k.data(), v.data(), gate.data(), beta.data()};
    void * out_host[2] = {next.data(), y.data()};
    ref(d, in_host, out_host);
    for (int64_t h = 0; h < 4; ++h) {
        for (int64_t j = 0; j < 2; ++j) {
            near(y[h * 2 + j], v[h * 2 + j] * beta[h]);
            for (int64_t i = 0; i < 2; ++i) {
                const float want = (i == h % 2) ? v[h * 2 + j] * beta[h] : 0.0f;
                near(next[(i * 4 + h) * 2 + j], want);
            }
        }
    }
    Cpu cpu;
    scrapgpu_tensor_desc ins[6];
    ins[0] = cpu.upload(base[0], state);
    ins[1] = cpu.upload(base[1], q);
    ins[2] = cpu.upload(base[2], k);
    ins[3] = cpu.upload(base[3], v);
    ins[4] = cpu.upload(base[4], gate);
    ins[5] = cpu.upload(base[5], beta);
    scrapgpu_tensor_desc outputs[2];
    outputs[0] = cpu.upload(outbase[0], std::vector<float>(16, -1));
    outputs[1] = cpu.upload(outbase[1], std::vector<float>(8, -1));
    d = op(SCRAPGPU_OP_GDN_STEP, ins, 6, outputs, 2);
    cpu.run(d);
    const auto got_next = cpu.read<float>(outputs[0], 16);
    const auto got_y = cpu.read<float>(outputs[1], 8);
    for (size_t i = 0; i < 16; ++i) near(got_next[i], next[i]);
    for (size_t i = 0; i < 8; ++i) near(got_y[i], y[i]);

    // A second step from non-zero state with decay tests update ordering.
    // In-place state must agree with the reference out-of-place result.
    std::vector<float> state2 = next;
    std::vector<float> gate2(4, std::log(0.5f));
    scrapgpu_tensor_desc ins2[6];
    std::memcpy(ins2, ins, sizeof(ins));
    ins2[0] = cpu.upload(base[0], state2);
    ins2[4] = cpu.upload(base[4], gate2);
    outputs[0] = ins2[0]; // exact alias is permitted for state only
    d = op(SCRAPGPU_OP_GDN_STEP, ins2, 6, outputs, 2);
    void * host2[6] = {state2.data(), q.data(), k.data(), v.data(), gate2.data(), beta.data()};
    std::vector<float> want_state(16), want_y(8);
    void * ref2[2] = {want_state.data(), want_y.data()};
    ref(d, host2, ref2);
    cpu.run(d);
    const auto got2 = cpu.read<float>(outputs[0], 16);
    const auto got2y = cpu.read<float>(outputs[1], 8);
    for (size_t i = 0; i < 16; ++i) near(got2[i], want_state[i]);
    for (size_t i = 0; i < 8; ++i) near(got2y[i], want_y[i]);

    // Even an exact alias of an unrelated input is unsafe and rejected.
    outputs[0] = cpu.upload(outbase[0], std::vector<float>(16, 0));
    outputs[1] = ins2[3]; // aliases v input
    d = op(SCRAPGPU_OP_GDN_STEP, ins2, 6, outputs, 2);
    CHECK(scrapgpu_op_execute(cpu.stream, &d) == SCRAPGPU_STATUS_INVALID_ARGUMENT);

    // Bad head geometry must fail before touching tensors.
    ins2[1].ne[1] = 3; // keep a real backing buffer; only geometry is wrong
    d = op(SCRAPGPU_OP_GDN_STEP, ins2, 6, outputs, 2);
    CHECK(scrapgpu_op_desc_validate(&d) == SCRAPGPU_STATUS_OK);
    CHECK(scrapgpu_op_supported(cpu.device, &d, nullptr) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
}

static void qsa_step() {
    const auto pdesc = tensor(SCRAPGPU_DTYPE_I32, 1);
    auto stepdesc = tensor(SCRAPGPU_DTYPE_I32, 4);
    scrapgpu_qsa_step_params params{sizeof(params), 4, 2048};
    auto d = op(SCRAPGPU_OP_QSA_STEP, &pdesc, 1, &stepdesc, 1, &params, sizeof(params));
    Cpu cpu;
    auto x = cpu.upload(pdesc, std::vector<int32_t>{0});
    auto y = cpu.upload(stepdesc, std::vector<int32_t>(4, -1));
    d = op(SCRAPGPU_OP_QSA_STEP, &x, 1, &y, 1, &params, sizeof(params));
    for (int32_t pos : {0, 3, 4, 2047, 2050, 2051}) {
        OK(scrapgpu_buffer_write(cpu.stream, x.buffer, 0, &pos, sizeof(pos), 1));
        int32_t reference[4] = {};
        void * in_host[1] = {&pos}, * out_host[1] = {reference};
        ref(d, in_host, out_host);
        cpu.run(d);
        auto step = cpu.read<int32_t>(y, 4);
        for (int i = 0; i < 4; ++i) CHECK(step[i] == reference[i]);
        CHECK(step[0] == pos && step[1] == pos + 1);
        CHECK(step[2] == (pos + 1) / 4);
        CHECK(step[3] == (pos + 1 < 2051 ? pos + 1 : 2051));
    }
    int32_t negative = -1;
    OK(scrapgpu_buffer_write(cpu.stream, x.buffer, 0, &negative, sizeof(negative), 1));
    CHECK(scrapgpu_op_execute(cpu.stream, &d) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    params.idx_block = 0;
    CHECK(scrapgpu_op_supported(cpu.device, &d, nullptr) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
    params.idx_block = 4;
    params.idx_top_k = std::numeric_limits<uint32_t>::max();
    CHECK(scrapgpu_op_supported(cpu.device, &d, nullptr) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
}

static void gdn_l2_norm() {
    const auto xdesc = tensor(SCRAPGPU_DTYPE_F32, 2, 2);
    auto ydesc = xdesc;
    scrapgpu_rms_norm_params eps{sizeof(eps), 1e-6f};
    auto d = op(SCRAPGPU_OP_GDN_L2_NORM, &xdesc, 1, &ydesc, 1, &eps, sizeof(eps));
    std::vector<float> x = {3, 4, 0, 0};
    std::vector<float> y(4, -1);
    void * ip[1] = {x.data()}, * yp[1] = {y.data()};
    ref(d, ip, yp);
    near(y[0], 0.6f);
    near(y[1], 0.8f);
    near(y[2], 0.f);
    near(y[3], 0.f);
    // RMS_NORM would use 1/sqrt(25/2); this must use 1/sqrt(25).
    CHECK(y[0] < 0.7f);
    Cpu cpu;
    auto in = cpu.upload(xdesc, x);
    auto out = cpu.upload(ydesc, std::vector<float>(4, 0));
    d = op(SCRAPGPU_OP_GDN_L2_NORM, &in, 1, &out, 1, &eps, sizeof(eps));
    cpu.run(d);
    auto got = cpu.read<float>(out, 4);
    for (size_t i = 0; i < 4; ++i) near(got[i], y[i]);
    // An identical view can be normalized in place.
    d = op(SCRAPGPU_OP_GDN_L2_NORM, &in, 1, &in, 1, &eps, sizeof(eps));
    cpu.run(d);
    got = cpu.read<float>(in, 4);
    for (size_t i = 0; i < 4; ++i) near(got[i], y[i]);
    eps.eps = -1.0f;
    CHECK(scrapgpu_op_supported(cpu.device, &d, nullptr) == SCRAPGPU_STATUS_INVALID_ARGUMENT);
}

static void gdn_conv_step() {
    const scrapgpu_tensor_desc bases[3] = {
        tensor(SCRAPGPU_DTYPE_F32, 3, 2), // state[oldest ... newest,channel]
        tensor(SCRAPGPU_DTYPE_F32, 2),    // x
        tensor(SCRAPGPU_DTYPE_F32, 4, 2), // channel-major convolution weights
    };
    scrapgpu_tensor_desc results[2] = {bases[0], bases[1]};
    std::vector<float> state = {1, 2, 3, 4, 5, 6};
    std::vector<float> x = {10, 20};
    std::vector<float> w = {0.1f, 0.2f, 0.3f, 0.4f, 1, 1, 1, 1};
    std::vector<float> next(6, -1), out(2, -1);
    auto d = op(SCRAPGPU_OP_GDN_CONV_STEP, bases, 3, results, 2);
    void * in[3] = {state.data(), x.data(), w.data()}, * o[2] = {next.data(), out.data()};
    ref(d, in, o);
    near(out[0], 5.4f);
    near(out[1], 35.0f);
    for (int i = 0; i < 6; ++i) near(next[i], std::vector<float>{2, 3, 10, 5, 6, 20}[i]);
    Cpu cpu;
    scrapgpu_tensor_desc inputs[3] = {
        cpu.upload(bases[0], state), cpu.upload(bases[1], x), cpu.upload(bases[2], w)
    };
    scrapgpu_tensor_desc outputs[2] = {
        inputs[0], // in-place state update, identical view
        cpu.upload(results[1], std::vector<float>(2, -1))
    };
    d = op(SCRAPGPU_OP_GDN_CONV_STEP, inputs, 3, outputs, 2);
    cpu.run(d);
    const auto got_state = cpu.read<float>(outputs[0], 6), got_out = cpu.read<float>(outputs[1], 2);
    for (int i = 0; i < 6; ++i) near(got_state[i], next[i]);
    for (int i = 0; i < 2; ++i) near(got_out[i], out[i]);
}

int main() {
    gdn_step();
    qsa_step();
    gdn_l2_norm();
    gdn_conv_step();
    std::puts("PASS: XeStrata decode primitive reference / CPU backend");
    return 0;
}
