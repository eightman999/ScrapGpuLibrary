// Stage A PyTorch adapter: CPU float tensors in/out; each op copies through the host.
#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/opencl11.h"
#include "scrapgpu/op.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/status.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#include <ATen/ATen.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::mutex g_mu;
bool g_ready = false;
scrapgpu_device_t * g_dev = nullptr;
scrapgpu_stream_t * g_stream = nullptr;

void check_status(scrapgpu_status st, const char * what) {
    if (st != SCRAPGPU_STATUS_OK) {
        throw std::runtime_error(std::string(what) + ": " + scrapgpu_status_string(st) + " (" + scrapgpu_last_error() + ")");
    }
}

void ensure_gpu() {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_ready) {
        return;
    }
    check_status(scrapgpu_opencl11_register(), "scrapgpu_opencl11_register");
    if (scrapgpu_get_device_count() == 0) {
        throw std::runtime_error("no ScrapGpu OpenCL device");
    }
    check_status(scrapgpu_get_device(0, &g_dev), "scrapgpu_get_device");
    check_status(scrapgpu_stream_create(g_dev, &g_stream), "scrapgpu_stream_create");
    g_ready = true;
}

scrapgpu_tensor_desc make_desc(
    scrapgpu_dtype dt,
    scrapgpu_layout layout,
    int64_t n0,
    int64_t n1,
    int64_t n2,
    int64_t n3,
    scrapgpu_buffer_t * buf) {
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
    t.buffer = buf;
    t.offset = 0;
    t.layout = layout;
    return t;
}

size_t tensor_bytes(const scrapgpu_tensor_desc & t) {
    return static_cast<size_t>(scrapgpu_tensor_row_size(t.dtype, t.ne[0]) * t.ne[1] * t.ne[2] * t.ne[3]);
}

void run_op(scrapgpu_op op, const std::vector<scrapgpu_tensor_desc> & ins, std::vector<scrapgpu_tensor_desc> & outs, const void * params, size_t params_size) {
    ensure_gpu();
    std::vector<scrapgpu_tensor_desc> in_copy = ins;
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = op;
    desc.inputs = in_copy.data();
    desc.input_count = in_copy.size();
    desc.outputs = outs.data();
    desc.output_count = outs.size();
    desc.params = params;
    desc.params_size = params_size;
    check_status(scrapgpu_op_desc_validate(&desc), "scrapgpu_op_desc_validate");
    const char * reason = nullptr;
    check_status(scrapgpu_op_supported(g_dev, &desc, &reason), reason ? reason : "scrapgpu_op_supported");
    check_status(scrapgpu_op_execute(g_stream, &desc), "scrapgpu_op_execute");
    check_status(scrapgpu_stream_sync(g_stream), "scrapgpu_stream_sync");
}

at::Tensor require_cpu_f32(const at::Tensor & t, const char * name) {
    TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t;
}

uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t e = static_cast<int32_t>((x >> 23) & 0xff) - 127 + 15;
    const uint32_t m = x & 0x7fffffu;
    if (e <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (e >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t h = sign | (static_cast<uint32_t>(e) << 10) | (m >> 13);
    if ((m & 0x1fffu) > 0x1000u || ((m & 0x3fffu) == 0x3000u)) {
        ++h;
    }
    return static_cast<uint16_t>(h);
}

scrapgpu_dtype parse_quant(const std::string & q) {
    if (q == "f32") {
        return SCRAPGPU_DTYPE_F32;
    }
    if (q == "f16") {
        return SCRAPGPU_DTYPE_F16;
    }
    if (q == "q4_0") {
        return SCRAPGPU_DTYPE_Q4_0;
    }
    if (q == "q8_0") {
        return SCRAPGPU_DTYPE_Q8_0;
    }
    if (q == "q4_k") {
        return SCRAPGPU_DTYPE_Q4_K;
    }
    if (q == "q6_k") {
        return SCRAPGPU_DTYPE_Q6_K;
    }
    throw std::runtime_error("unknown quantization type: " + q);
}

bool is_quantized(scrapgpu_dtype dt) {
    return dt == SCRAPGPU_DTYPE_Q4_0 || dt == SCRAPGPU_DTYPE_Q8_0 || dt == SCRAPGPU_DTYPE_Q4_K || dt == SCRAPGPU_DTYPE_Q6_K;
}

class WeightImpl : public torch::CustomClassHolder {
public:
    WeightImpl(at::Tensor weight_f32, std::string quant_type) : quant_type_(std::move(quant_type)) {
        at::Tensor w = require_cpu_f32(weight_f32, "weight");
        TORCH_CHECK(w.dim() == 2, "weight must be 2-D [out_features, in_features]");
        const int64_t m = w.size(0);
        const int64_t k = w.size(1);
        out_features_ = m;
        in_features_ = k;
        dtype_ = parse_quant(quant_type_);
        layout_ = is_quantized(dtype_) ? SCRAPGPU_LAYOUT_R32 : SCRAPGPU_LAYOUT_NATURAL;

        // ScrapGpu MUL_MAT: src0 is [K, M] (ne0=K, ne1=M); PyTorch Linear weight is [M, K].
        std::vector<float> km(static_cast<size_t>(k * m));
        for (int64_t mi = 0; mi < m; ++mi) {
            for (int64_t ki = 0; ki < k; ++ki) {
                km[static_cast<size_t>(ki + mi * k)] = w[mi][ki].item<float>();
            }
        }

        std::vector<uint8_t> host;
        if (dtype_ == SCRAPGPU_DTYPE_F32) {
            host.resize(static_cast<size_t>(k * m) * sizeof(float));
            std::memcpy(host.data(), km.data(), host.size());
        } else if (dtype_ == SCRAPGPU_DTYPE_F16) {
            host.resize(static_cast<size_t>(k * m) * 2);
            for (int64_t i = 0; i < k * m; ++i) {
                const uint16_t h = f32_to_f16(km[static_cast<size_t>(i)]);
                std::memcpy(host.data() + static_cast<size_t>(i) * 2, &h, 2);
            }
        } else {
            host.resize(tensor_bytes(make_desc(dtype_, SCRAPGPU_LAYOUT_NATURAL, k, m, 1, 1, nullptr)));
            check_status(scrapgpu_ref_quantize(dtype_, km.data(), host.data(), k * m), "scrapgpu_ref_quantize");
        }

        ensure_gpu();
        check_status(scrapgpu_buffer_alloc(g_dev, host.size(), &buf_), "scrapgpu_buffer_alloc");
        desc_ = make_desc(dtype_, layout_, k, m, 1, 1, buf_);
        check_status(scrapgpu_tensor_upload(g_stream, &desc_, host.data(), host.size()), "weight upload");
        check_status(scrapgpu_stream_sync(g_stream), "stream_sync after weight upload");
    }

    ~WeightImpl() override {
        if (buf_) {
            scrapgpu_buffer_free(buf_);
            buf_ = nullptr;
        }
    }

    const scrapgpu_tensor_desc & desc() const {
        return desc_;
    }
    int64_t in_features() const {
        return in_features_;
    }
    int64_t out_features() const {
        return out_features_;
    }
    const std::string & quant_type() const {
        return quant_type_;
    }

private:
    scrapgpu_buffer_t * buf_ = nullptr;
    scrapgpu_tensor_desc desc_{};
    scrapgpu_dtype dtype_ = SCRAPGPU_DTYPE_F32;
    scrapgpu_layout layout_ = SCRAPGPU_LAYOUT_NATURAL;
    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
    std::string quant_type_;
};

at::Tensor linear_cpu(const at::Tensor & x, const c10::intrusive_ptr<WeightImpl> & weight) {
    at::Tensor xt = require_cpu_f32(x, "x");
    const int64_t k = weight->in_features();
    const int64_t m = weight->out_features();
    bool vec = false;
    if (xt.dim() == 1) {
        TORCH_CHECK(xt.size(0) == k, "x length must match in_features");
        xt = xt.unsqueeze(0);
        vec = true;
    } else if (xt.dim() == 2) {
        TORCH_CHECK(xt.size(1) == k, "x.shape[1] must match in_features");
    } else {
        TORCH_CHECK(false, "x must be [N,K] or [K]");
    }
    const int64_t n = xt.size(0);

    scrapgpu_buffer_t * bx = nullptr;
    scrapgpu_buffer_t * by = nullptr;
    check_status(scrapgpu_buffer_alloc(g_dev, static_cast<size_t>(k * n) * sizeof(float), &bx), "buffer_alloc x");
    check_status(scrapgpu_buffer_alloc(g_dev, static_cast<size_t>(m * n) * sizeof(float), &by), "buffer_alloc y");

    scrapgpu_tensor_desc xdesc = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, k, n, 1, 1, bx);
    scrapgpu_tensor_desc ydesc = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, m, n, 1, 1, by);

    check_status(scrapgpu_tensor_upload(g_stream, &xdesc, xt.data_ptr<float>(), static_cast<size_t>(k * n) * sizeof(float)), "upload x");

    scrapgpu_mul_mat_params mm{};
    mm.struct_size = sizeof(mm);
    mm.allow_broadcast = 1;
    std::vector<scrapgpu_tensor_desc> ins = {weight->desc(), xdesc};
    std::vector<scrapgpu_tensor_desc> outs = {ydesc};
    run_op(SCRAPGPU_OP_MUL_MAT, ins, outs, &mm, sizeof(mm));

    at::Tensor y = at::empty({n, m}, xt.options());
    check_status(scrapgpu_tensor_download(g_stream, &ydesc, y.data_ptr<float>(), static_cast<size_t>(m * n) * sizeof(float)), "download y");

    scrapgpu_buffer_free(bx);
    scrapgpu_buffer_free(by);

    if (vec) {
        return y.squeeze(0);
    }
    return y;
}

at::Tensor unary_f32(scrapgpu_unary_kind kind, const at::Tensor & x) {
    at::Tensor xt = require_cpu_f32(x, "x");
    scrapgpu_buffer_t * bx = nullptr;
    scrapgpu_buffer_t * by = nullptr;
    const int64_t n0 = xt.size(-1);
    const int64_t n1 = xt.numel() / n0;
    check_status(scrapgpu_buffer_alloc(g_dev, static_cast<size_t>(xt.numel()) * sizeof(float), &bx), "buffer_alloc");
    check_status(scrapgpu_buffer_alloc(g_dev, static_cast<size_t>(xt.numel()) * sizeof(float), &by), "buffer_alloc");

    scrapgpu_tensor_desc in = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, n1, 1, 1, bx);
    scrapgpu_tensor_desc out = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, n1, 1, 1, by);
    check_status(scrapgpu_tensor_upload(g_stream, &in, xt.data_ptr<float>(), static_cast<size_t>(xt.numel()) * sizeof(float)), "upload");

    scrapgpu_unary_params p{};
    p.struct_size = sizeof(p);
    p.kind = kind;
    std::vector<scrapgpu_tensor_desc> outs = {out};
    run_op(SCRAPGPU_OP_UNARY, {in}, outs, &p, sizeof(p));

    at::Tensor y = at::empty_like(xt);
    check_status(scrapgpu_tensor_download(g_stream, &out, y.data_ptr<float>(), static_cast<size_t>(xt.numel()) * sizeof(float)), "download");
    scrapgpu_buffer_free(bx);
    scrapgpu_buffer_free(by);
    return y;
}

at::Tensor add_cpu(const at::Tensor & a, const at::Tensor & b) {
    at::Tensor ta = require_cpu_f32(a, "a");
    at::Tensor tb = require_cpu_f32(b, "b");
    TORCH_CHECK(ta.sizes() == tb.sizes(), "add: shape mismatch");
    scrapgpu_buffer_t * ba = nullptr;
    scrapgpu_buffer_t * bb = nullptr;
    scrapgpu_buffer_t * bd = nullptr;
    const size_t bytes = static_cast<size_t>(ta.numel()) * sizeof(float);
    check_status(scrapgpu_buffer_alloc(g_dev, bytes, &ba), "buffer_alloc");
    check_status(scrapgpu_buffer_alloc(g_dev, bytes, &bb), "buffer_alloc");
    check_status(scrapgpu_buffer_alloc(g_dev, bytes, &bd), "buffer_alloc");

    const int64_t n0 = ta.size(-1);
    const int64_t n1 = ta.numel() / n0;
    scrapgpu_tensor_desc da = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, n1, 1, 1, ba);
    scrapgpu_tensor_desc db = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, n1, 1, 1, bb);
    scrapgpu_tensor_desc dd = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, n1, 1, 1, bd);
    check_status(scrapgpu_tensor_upload(g_stream, &da, ta.data_ptr<float>(), bytes), "upload a");
    check_status(scrapgpu_tensor_upload(g_stream, &db, tb.data_ptr<float>(), bytes), "upload b");
    std::vector<scrapgpu_tensor_desc> outs = {dd};
    run_op(SCRAPGPU_OP_ADD, {da, db}, outs, nullptr, 0);
    at::Tensor out = at::empty_like(ta);
    check_status(scrapgpu_tensor_download(g_stream, &dd, out.data_ptr<float>(), bytes), "download");
    scrapgpu_buffer_free(ba);
    scrapgpu_buffer_free(bb);
    scrapgpu_buffer_free(bd);
    return out;
}

at::Tensor rms_norm_cpu(const at::Tensor & x, const at::Tensor & w, double eps) {
    at::Tensor xt = require_cpu_f32(x, "x");
    at::Tensor wt = require_cpu_f32(w, "w");
    TORCH_CHECK(xt.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(wt.dim() == 1, "w must be 1-D");
    TORCH_CHECK(wt.size(0) == xt.size(-1), "w length must match x last dim");
    const int64_t n0 = xt.size(-1);
    const int64_t n1 = xt.numel() / n0;

    scrapgpu_buffer_t * bx = nullptr;
    scrapgpu_buffer_t * bw = nullptr;
    scrapgpu_buffer_t * by = nullptr;
    const size_t xbytes = static_cast<size_t>(xt.numel()) * sizeof(float);
    const size_t wbytes = static_cast<size_t>(wt.numel()) * sizeof(float);
    check_status(scrapgpu_buffer_alloc(g_dev, xbytes, &bx), "buffer_alloc x");
    check_status(scrapgpu_buffer_alloc(g_dev, wbytes, &bw), "buffer_alloc w");
    check_status(scrapgpu_buffer_alloc(g_dev, xbytes, &by), "buffer_alloc y");

    scrapgpu_tensor_desc tx = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, n1, 1, 1, bx);
    scrapgpu_tensor_desc tw = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, 1, 1, 1, bw);
    scrapgpu_tensor_desc ty = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, n0, n1, 1, 1, by);
    check_status(scrapgpu_tensor_upload(g_stream, &tx, xt.data_ptr<float>(), xbytes), "upload x");
    check_status(scrapgpu_tensor_upload(g_stream, &tw, wt.data_ptr<float>(), wbytes), "upload w");

    scrapgpu_rms_norm_params rp{};
    rp.struct_size = sizeof(rp);
    rp.eps = static_cast<float>(eps);
    std::vector<scrapgpu_tensor_desc> outs = {ty};
    run_op(SCRAPGPU_OP_RMS_NORM_MUL, {tx, tw}, outs, &rp, sizeof(rp));

    at::Tensor y = at::empty_like(xt);
    check_status(scrapgpu_tensor_download(g_stream, &ty, y.data_ptr<float>(), xbytes), "download");
    scrapgpu_buffer_free(bx);
    scrapgpu_buffer_free(bw);
    scrapgpu_buffer_free(by);
    return y;
}

at::Tensor rope_cpu(
    const at::Tensor & x,
    const at::Tensor & pos,
    int64_t n_dims,
    int64_t n_ctx_orig,
    double freq_base,
    double freq_scale,
    double ext_factor,
    double attn_factor,
    double beta_fast,
    double beta_slow,
    int64_t mode) {
    at::Tensor xt = require_cpu_f32(x, "x");
    TORCH_CHECK(xt.dim() == 3, "x must be [head_dim, n_heads, n_tokens] (ScrapGpu layout)");
    TORCH_CHECK(pos.device().is_cpu() && pos.scalar_type() == at::kInt, "pos must be CPU int32");
    TORCH_CHECK(pos.is_contiguous(), "pos must be contiguous");
    const int64_t hd = xt.size(0);
    const int64_t nh = xt.size(1);
    const int64_t nt = xt.size(2);
    TORCH_CHECK(pos.size(0) >= nt, "pos must have at least n_tokens entries");

    scrapgpu_buffer_t * bx = nullptr;
    scrapgpu_buffer_t * bp = nullptr;
    scrapgpu_buffer_t * by = nullptr;
    const size_t xbytes = static_cast<size_t>(xt.numel()) * sizeof(float);
    const size_t pbytes = static_cast<size_t>(pos.numel()) * sizeof(int32_t);
    check_status(scrapgpu_buffer_alloc(g_dev, xbytes, &bx), "buffer_alloc x");
    check_status(scrapgpu_buffer_alloc(g_dev, pbytes, &bp), "buffer_alloc pos");
    check_status(scrapgpu_buffer_alloc(g_dev, xbytes, &by), "buffer_alloc y");

    scrapgpu_tensor_desc tx = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, hd, nh, nt, 1, bx);
    scrapgpu_tensor_desc tp = make_desc(SCRAPGPU_DTYPE_I32, SCRAPGPU_LAYOUT_NATURAL, pos.size(0), 1, 1, 1, bp);
    scrapgpu_tensor_desc ty = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, hd, nh, nt, 1, by);
    check_status(scrapgpu_tensor_upload(g_stream, &tx, xt.data_ptr<float>(), xbytes), "upload x");
    check_status(scrapgpu_tensor_upload(g_stream, &tp, pos.data_ptr<int32_t>(), pbytes), "upload pos");

    scrapgpu_rope_params rp{};
    rp.struct_size = sizeof(rp);
    rp.mode = mode ? SCRAPGPU_ROPE_MODE_NEOX : SCRAPGPU_ROPE_MODE_NORMAL;
    rp.n_dims = static_cast<int32_t>(n_dims > 0 ? n_dims : hd);
    rp.n_ctx_orig = static_cast<int32_t>(n_ctx_orig);
    rp.freq_base = static_cast<float>(freq_base);
    rp.freq_scale = static_cast<float>(freq_scale);
    rp.ext_factor = static_cast<float>(ext_factor);
    rp.attn_factor = static_cast<float>(attn_factor);
    rp.beta_fast = static_cast<float>(beta_fast);
    rp.beta_slow = static_cast<float>(beta_slow);
    std::vector<scrapgpu_tensor_desc> outs = {ty};
    run_op(SCRAPGPU_OP_ROPE, {tx, tp}, outs, &rp, sizeof(rp));

    at::Tensor y = at::empty_like(xt);
    check_status(scrapgpu_tensor_download(g_stream, &ty, y.data_ptr<float>(), xbytes), "download");
    scrapgpu_buffer_free(bx);
    scrapgpu_buffer_free(bp);
    scrapgpu_buffer_free(by);
    return y;
}

// CPU reference rope for tests (same layout as x).
at::Tensor ref_rope_cpu(
    const at::Tensor & x,
    const at::Tensor & pos,
    int64_t n_dims,
    int64_t n_ctx_orig,
    double freq_base,
    double freq_scale,
    double ext_factor,
    double attn_factor,
    double beta_fast,
    double beta_slow,
    int64_t mode) {
    at::Tensor xt = require_cpu_f32(x, "x").clone();
    at::Tensor pt = pos.to(at::kInt).contiguous();
    scrapgpu_tensor_desc tx = make_desc(SCRAPGPU_DTYPE_F32, SCRAPGPU_LAYOUT_NATURAL, xt.size(0), xt.size(1), xt.size(2), 1, nullptr);
    scrapgpu_tensor_desc tp = make_desc(SCRAPGPU_DTYPE_I32, SCRAPGPU_LAYOUT_NATURAL, pt.size(0), 1, 1, 1, nullptr);
    scrapgpu_tensor_desc ty = tx;
    scrapgpu_rope_params rp{};
    rp.struct_size = sizeof(rp);
    rp.mode = mode ? SCRAPGPU_ROPE_MODE_NEOX : SCRAPGPU_ROPE_MODE_NORMAL;
    rp.n_dims = static_cast<int32_t>(n_dims > 0 ? n_dims : xt.size(0));
    rp.n_ctx_orig = static_cast<int32_t>(n_ctx_orig);
    rp.freq_base = static_cast<float>(freq_base);
    rp.freq_scale = static_cast<float>(freq_scale);
    rp.ext_factor = static_cast<float>(ext_factor);
    rp.attn_factor = static_cast<float>(attn_factor);
    rp.beta_fast = static_cast<float>(beta_fast);
    rp.beta_slow = static_cast<float>(beta_slow);
    at::Tensor y = at::empty_like(xt);
    const scrapgpu_tensor_desc ins[2] = {tx, tp};
    scrapgpu_op_desc od{};
    od.struct_size = sizeof(od);
    od.op = SCRAPGPU_OP_ROPE;
    od.inputs = ins;
    od.input_count = 2;
    od.outputs = &ty;
    od.output_count = 1;
    od.params = &rp;
    od.params_size = sizeof(rp);
    void * in_ptr[2] = {xt.data_ptr<float>(), pt.data_ptr<int32_t>()};
    void * out_ptr[1] = {y.data_ptr<float>()};
    check_status(scrapgpu_ref_execute(&od, in_ptr, out_ptr), "scrapgpu_ref_execute ROPE");
    return y;
}

int64_t device_count() {
    if (scrapgpu_opencl11_register() != SCRAPGPU_STATUS_OK) {
        return 0;
    }
    return static_cast<int64_t>(scrapgpu_get_device_count());
}

std::string device_name(int64_t index) {
    check_status(scrapgpu_opencl11_register(), "register");
    scrapgpu_device_t * dev = nullptr;
    check_status(scrapgpu_get_device(static_cast<size_t>(index), &dev), "get_device");
    scrapgpu_device_info info{};
    check_status(scrapgpu_device_info_init(&info, sizeof(info)), "device_info_init");
    check_status(scrapgpu_device_get_info(dev, &info), "device_get_info");
    return info.name ? info.name : "";
}

void load_library() {
    ensure_gpu();
}

// Dequantize a PyTorch-shaped [out,in] weight for torch reference tests.
at::Tensor ref_dequant_linear_weight(at::Tensor weight_f32, std::string quant_type) {
    at::Tensor w = require_cpu_f32(weight_f32, "weight");
    TORCH_CHECK(w.dim() == 2, "weight must be 2-D");
    const int64_t m = w.size(0);
    const int64_t k = w.size(1);
    const scrapgpu_dtype dt = parse_quant(quant_type);
    TORCH_CHECK(is_quantized(dt), "ref_dequant_linear_weight expects a quantized type");
    std::vector<float> km(static_cast<size_t>(k * m));
    for (int64_t mi = 0; mi < m; ++mi) {
        for (int64_t ki = 0; ki < k; ++ki) {
            km[static_cast<size_t>(ki + mi * k)] = w[mi][ki].item<float>();
        }
    }
    scrapgpu_tensor_desc td = make_desc(dt, SCRAPGPU_LAYOUT_NATURAL, k, m, 1, 1, nullptr);
    std::vector<uint8_t> qhost(tensor_bytes(td));
    check_status(scrapgpu_ref_quantize(dt, km.data(), qhost.data(), k * m), "quantize");
    std::vector<float> km_deq(static_cast<size_t>(k * m));
    check_status(scrapgpu_ref_dequantize(dt, qhost.data(), km_deq.data(), k * m), "dequantize");
    at::Tensor out = at::empty({m, k}, w.options());
    for (int64_t mi = 0; mi < m; ++mi) {
        for (int64_t ki = 0; ki < k; ++ki) {
            out[mi][ki] = km_deq[static_cast<size_t>(ki + mi * k)];
        }
    }
    return out;
}

} // namespace

TORCH_LIBRARY(scrapgpu, m) {
    m.class_<WeightImpl>("Weight")
        .def(torch::init<at::Tensor, std::string>())
        .def("in_features", &WeightImpl::in_features)
        .def("out_features", &WeightImpl::out_features)
        .def("quant_type", &WeightImpl::quant_type);
    m.def("device_count", &device_count);
    m.def("device_name", &device_name);
    m.def("load", &load_library);
    m.def("linear(Tensor x, __torch__.torch.classes.scrapgpu.Weight weight) -> Tensor", &linear_cpu);
    m.def("silu(Tensor x) -> Tensor", [](const at::Tensor & x) { return unary_f32(SCRAPGPU_UNARY_SILU, x); });
    m.def("add(Tensor a, Tensor b) -> Tensor", &add_cpu);
    m.def(
        "rms_norm(Tensor x, Tensor w, float eps) -> Tensor",
        [](const at::Tensor & x, const at::Tensor & w, double eps) { return rms_norm_cpu(x, w, eps); });
    m.def(
        "rope(Tensor x, Tensor pos, int n_dims, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, "
        "float attn_factor, float beta_fast, float beta_slow, int mode) -> Tensor",
        &rope_cpu);
    m.def(
        "ref_rope(Tensor x, Tensor pos, int n_dims, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, "
        "float attn_factor, float beta_fast, float beta_slow, int mode) -> Tensor",
        &ref_rope_cpu);
    m.def("ref_dequant_linear_weight(Tensor weight, str quant_type) -> Tensor", &ref_dequant_linear_weight);
}
