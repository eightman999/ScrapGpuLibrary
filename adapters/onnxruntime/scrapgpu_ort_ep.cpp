// ScrapGpuLibrary ONNX Runtime plugin execution provider (compile-based, first slice).
// SPDX-License-Identifier: GPL-3.0-or-later

#include "onnxruntime_c_api.h"

#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/opencl11.h"
#include "scrapgpu/op.h"
#include "scrapgpu/reference.h"
#include "scrapgpu/status.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define SGL_ORT_EP_NAME "ScrapGpuExecutionProvider"
#define SGL_ORT_EP_VENDOR "ScrapGpu"
#define SGL_ORT_EP_VERSION "0.1.0"

#if defined(__GNUC__) || defined(__clang__)
#  define SGL_ORT_EP_EXPORT __attribute__((visibility("default")))
#else
#  define SGL_ORT_EP_EXPORT
#endif

namespace {

const OrtApi * g_api = nullptr;
const OrtEpApi * g_ep_api = nullptr;
std::once_flag g_register_once;
std::atomic<uint64_t> g_claimed_nodes{0};

void publish_claimed_count() {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(g_claimed_nodes.load()));
    setenv("SCRAPGPU_ORT_EP_CLAIMED_NODES", buf, 1);
}

void note_claimed(const char * op) {
    g_claimed_nodes.fetch_add(1);
    publish_claimed_count();
    if (std::getenv("SCRAPGPU_ORT_EP_LOG_CLAIMS") != nullptr) {
        std::fprintf(stderr, "[ScrapGpuExecutionProvider] claimed %s\n", op);
    }
}

OrtStatus * ort_ok() { return nullptr; }

OrtStatus * ort_fail(const char * msg) {
    return g_api->CreateStatus(ORT_FAIL, msg);
}

scrapgpu_dtype parse_weight_dtype(const OrtSessionOptions * session_options) {
    const char * keys[] = {
        "ep.scrapgpu.weight_type",
        "ep.ScrapGpuExecutionProvider.weight_type",
    };
    OrtKeyValuePairs * entries = nullptr;
    if (session_options == nullptr ||
        g_api->GetSessionOptionsConfigEntries(session_options, &entries) != nullptr) {
        return SCRAPGPU_DTYPE_F32;
    }
    scrapgpu_dtype dt = SCRAPGPU_DTYPE_F32;
    for (const char * key : keys) {
        const char * val = g_api->GetKeyValue(entries, key);
        if (val == nullptr) {
            continue;
        }
        if (std::strcmp(val, "Q8_0") == 0 || std::strcmp(val, "q8_0") == 0) {
            dt = SCRAPGPU_DTYPE_Q8_0;
        } else if (std::strcmp(val, "Q4_0") == 0 || std::strcmp(val, "q4_0") == 0) {
            dt = SCRAPGPU_DTYPE_Q4_0;
        } else {
            dt = SCRAPGPU_DTYPE_F32;
        }
        break;
    }
    g_api->ReleaseKeyValuePairs(entries);
    return dt;
}

void ensure_scrapgpu_registered() {
    std::call_once(g_register_once, []() {
        const scrapgpu_status st = scrapgpu_opencl11_register();
        if (st != SCRAPGPU_STATUS_OK) {
            std::fprintf(stderr, "scrapgpu_opencl11_register: %s\n", scrapgpu_last_error());
        }
    });
}

// ONNX row-major [rows, cols] -> ScrapGpu tensor with ne0=cols, ne1=rows (inner dim cols).
void transpose_f32_to_scrap(const float * onnx, float * dst, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            dst[c + r * cols] = onnx[r * cols + c];
        }
    }
}

// ONNX MatMul weight [k, n] row-major -> ScrapGpu ne0=k, ne1=n (inner dim k).
void transpose_weight_f32_to_scrap(const float * onnx, float * dst, int64_t k, int64_t n) {
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            dst[i + j * k] = onnx[i * n + j];
        }
    }
}

void transpose_f32_from_scrap(const float * src, float * onnx, int64_t rows, int64_t cols) {
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            onnx[r * cols + c] = src[c + r * cols];
        }
    }
}

bool tensor_dims_f32(const OrtTypeInfo * type_info, std::vector<int64_t> & dims) {
    const OrtTensorTypeAndShapeInfo * tinfo = nullptr;
    if (g_api->CastTypeInfoToTensorInfo(type_info, &tinfo) != nullptr) {
        return false;
    }
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (g_api->GetTensorElementType(tinfo, &et) != nullptr || et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        return false;
    }
    size_t n = 0;
    if (g_api->GetDimensionsCount(tinfo, &n) != nullptr) {
        return false;
    }
    dims.resize(n);
    if (n > 0 && g_api->GetDimensions(tinfo, dims.data(), n) != nullptr) {
        return false;
    }
    return true;
}

bool get_value_info_dims(const OrtValueInfo * vi, std::vector<int64_t> & dims) {
    const OrtTypeInfo * ti = nullptr;
    if (g_api->GetValueInfoTypeInfo(vi, &ti) != nullptr) {
        return false;
    }
    return tensor_dims_f32(ti, dims);
}

bool dims_equal(const std::vector<int64_t> & a, const std::vector<int64_t> & b) {
    return a == b;
}

bool is_empty_domain(const char * domain) { return domain == nullptr || domain[0] == '\0'; }

enum class NodeKind {
    MatMul,
    Add,
    Mul,
    RmsNorm,
};

struct ScrapGpuEp;

struct ComputeState {
    NodeKind kind = NodeKind::MatMul;
    ScrapGpuEp * ep = nullptr;
    scrapgpu_device_t * device = nullptr;
    scrapgpu_stream_t * stream = nullptr;

    scrapgpu_buffer_t * weight_buf = nullptr;
    scrapgpu_dtype weight_dtype = SCRAPGPU_DTYPE_F32;
    scrapgpu_layout weight_layout = SCRAPGPU_LAYOUT_NATURAL;
    size_t weight_host_bytes = 0;
    int64_t weight_k = 0;
    int64_t weight_n = 0;
    int64_t weight_k_padded = 0;
    int64_t weight_k_in = 0; // the K of the ONNX weight (weight_k is the padded K for quant)

    scrapgpu_buffer_t * scale_buf = nullptr;
    int64_t norm_cols = 0;
    float rms_eps = 1e-5f;
};

struct ScrapGpuEpFactory {
    OrtEpFactory iface{};
    std::vector<OrtHardwareDevice *> hardware_devices;
};

struct ScrapGpuEp {
    OrtEp iface{};
    ScrapGpuEpFactory * factory = nullptr;
    uint32_t device_index = 0;
    scrapgpu_device_t * device = nullptr;
    scrapgpu_stream_t * stream = nullptr;
    scrapgpu_dtype session_weight_dtype = SCRAPGPU_DTYPE_F32;
};

scrapgpu_tensor_desc make_tensor_desc(scrapgpu_dtype dt, scrapgpu_buffer_t * buf, int64_t n0, int64_t n1,
                                      scrapgpu_layout layout) {
    scrapgpu_tensor_desc t{};
    t.struct_size = sizeof(t);
    t.dtype = dt;
    t.ne[0] = n0;
    t.ne[1] = n1;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = scrapgpu_dtype_type_size(dt);
    t.nb[1] = scrapgpu_tensor_row_size(dt, n0);
    t.nb[2] = t.nb[1] * static_cast<uint64_t>(n1);
    t.nb[3] = t.nb[2];
    t.buffer = buf;
    t.offset = 0;
    t.layout = layout;
    return t;
}

static int64_t pad_dim_to_block(int64_t k, size_t block) {
    return static_cast<int64_t>(((static_cast<uint64_t>(k) + block - 1u) / block) * block);
}

scrapgpu_status upload_weight(ScrapGpuEp * ep, ComputeState * st, const float * onnx_weight, int64_t k, int64_t n) {
    st->weight_k = k;
    st->weight_n = n;
    st->weight_k_in = k;
    st->weight_k_padded = k;
    st->weight_dtype = ep->session_weight_dtype;
    if (st->weight_dtype == SCRAPGPU_DTYPE_F32) {
        st->weight_layout = SCRAPGPU_LAYOUT_NATURAL;
        st->weight_host_bytes = static_cast<size_t>(k * n) * sizeof(float);
        std::vector<float> transposed(static_cast<size_t>(k * n));
        transpose_weight_f32_to_scrap(onnx_weight, transposed.data(), k, n);
        if (scrapgpu_buffer_alloc(ep->device, st->weight_host_bytes, &st->weight_buf) != SCRAPGPU_STATUS_OK) {
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        scrapgpu_tensor_desc w = make_tensor_desc(SCRAPGPU_DTYPE_F32, st->weight_buf, k, n, SCRAPGPU_LAYOUT_NATURAL);
        return scrapgpu_tensor_upload(ep->stream, &w, transposed.data(), st->weight_host_bytes);
    }

    st->weight_layout = SCRAPGPU_LAYOUT_R32;
    const size_t block = scrapgpu_dtype_block_size(st->weight_dtype);
    const int64_t k_pad = pad_dim_to_block(k, block);
    st->weight_k_padded = k_pad;
    st->weight_k = k_pad;
    const size_t row_bytes = scrapgpu_tensor_row_size(st->weight_dtype, k_pad);
    if (row_bytes == 0) {
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    std::vector<float> transposed(static_cast<size_t>(k_pad * n));
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < k; ++i) {
            transposed[i + j * static_cast<size_t>(k_pad)] = onnx_weight[i * n + j];
        }
    }
    const size_t qbytes = row_bytes * static_cast<size_t>(n);
    std::vector<uint8_t> qhost(qbytes);
    if (scrapgpu_ref_quantize(st->weight_dtype, transposed.data(), qhost.data(), k_pad * n) !=
        SCRAPGPU_STATUS_OK) {
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    st->weight_host_bytes = qbytes;
    if (scrapgpu_buffer_alloc(ep->device, qbytes, &st->weight_buf) != SCRAPGPU_STATUS_OK) {
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    scrapgpu_tensor_desc w =
        make_tensor_desc(st->weight_dtype, st->weight_buf, k_pad, n, SCRAPGPU_LAYOUT_R32);
    return scrapgpu_tensor_upload(ep->stream, &w, qhost.data(), qbytes);
}

void free_compute_state(ComputeState * st) {
    if (st == nullptr) {
        return;
    }
    if (st->weight_buf != nullptr) {
        scrapgpu_buffer_free(st->weight_buf);
    }
    if (st->scale_buf != nullptr) {
        scrapgpu_buffer_free(st->scale_buf);
    }
    delete st;
}

OrtStatus * run_mul_mat(ScrapGpuEp * ep, ComputeState * st, const float * act_onnx, int64_t m, int64_t k, float * out_onnx,
                        int64_t n) {
    const int64_t k_act = st->weight_k_padded > k ? st->weight_k_padded : k;
    const size_t act_bytes = static_cast<size_t>(m * k_act) * sizeof(float);
    std::vector<float> act_sgl(static_cast<size_t>(k_act * m), 0.0f);
    if (k_act == k) {
        transpose_f32_to_scrap(act_onnx, act_sgl.data(), m, k);
    } else {
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t c = 0; c < k; ++c) {
                act_sgl[c + r * k_act] = act_onnx[r * k + c];
            }
        }
    }

    scrapgpu_buffer_t * bx = nullptr;
    scrapgpu_buffer_t * by = nullptr;
    if (scrapgpu_buffer_alloc(ep->device, static_cast<size_t>(k_act * m) * sizeof(float), &bx) != SCRAPGPU_STATUS_OK) {
        return ort_fail("MUL_MAT activation buffer alloc");
    }
    if (scrapgpu_buffer_alloc(ep->device, static_cast<size_t>(n * m) * sizeof(float), &by) != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(bx);
        return ort_fail("MUL_MAT output buffer alloc");
    }

    scrapgpu_tensor_desc w =
        make_tensor_desc(st->weight_dtype, st->weight_buf, st->weight_k, st->weight_n, st->weight_layout);
    scrapgpu_tensor_desc x = make_tensor_desc(SCRAPGPU_DTYPE_F32, bx, k_act, m, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_tensor_desc y = make_tensor_desc(SCRAPGPU_DTYPE_F32, by, n, m, SCRAPGPU_LAYOUT_NATURAL);

    scrapgpu_status st_u = scrapgpu_tensor_upload(ep->stream, &x, act_sgl.data(), act_bytes);
    if (st_u != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(bx);
        scrapgpu_buffer_free(by);
        return ort_fail(scrapgpu_last_error());
    }

    scrapgpu_mul_mat_params mm{};
    mm.struct_size = sizeof(mm);
    mm.allow_broadcast = 1;
    const scrapgpu_tensor_desc ins[2] = {w, x};
    scrapgpu_op_desc op{};
    op.struct_size = sizeof(op);
    op.op = SCRAPGPU_OP_MUL_MAT;
    op.inputs = ins;
    op.input_count = 2;
    op.outputs = &y;
    op.output_count = 1;
    op.params = &mm;
    op.params_size = sizeof(mm);

    const char * reason = nullptr;
    st_u = scrapgpu_op_supported(ep->device, &op, &reason);
    if (st_u != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(bx);
        scrapgpu_buffer_free(by);
        return ort_fail(reason != nullptr ? reason : "MUL_MAT unsupported");
    }
    st_u = scrapgpu_op_execute(ep->stream, &op);
    if (st_u != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(bx);
        scrapgpu_buffer_free(by);
        return ort_fail(scrapgpu_last_error());
    }
    std::vector<float> y_sgl(static_cast<size_t>(n * m));
    st_u = scrapgpu_tensor_download(ep->stream, &y, y_sgl.data(), static_cast<size_t>(n * m) * sizeof(float));
    scrapgpu_buffer_free(bx);
    scrapgpu_buffer_free(by);
    if (st_u != SCRAPGPU_STATUS_OK) {
        return ort_fail(scrapgpu_last_error());
    }
    transpose_f32_from_scrap(y_sgl.data(), out_onnx, m, n);
    return ort_ok();
}

OrtStatus * run_binary(ScrapGpuEp * ep, NodeKind kind, const float * a, const float * b, float * out, size_t count) {
    scrapgpu_buffer_t * ba = nullptr;
    scrapgpu_buffer_t * bb = nullptr;
    scrapgpu_buffer_t * bo = nullptr;
    const size_t bytes = count * sizeof(float);
    if (scrapgpu_buffer_alloc(ep->device, bytes, &ba) != SCRAPGPU_STATUS_OK ||
        scrapgpu_buffer_alloc(ep->device, bytes, &bb) != SCRAPGPU_STATUS_OK ||
        scrapgpu_buffer_alloc(ep->device, bytes, &bo) != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(ba);
        scrapgpu_buffer_free(bb);
        scrapgpu_buffer_free(bo);
        return ort_fail("binary buffer alloc");
    }
    scrapgpu_tensor_desc ta = make_tensor_desc(SCRAPGPU_DTYPE_F32, ba, static_cast<int64_t>(count), 1, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_tensor_desc tb = make_tensor_desc(SCRAPGPU_DTYPE_F32, bb, static_cast<int64_t>(count), 1, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_tensor_desc to = make_tensor_desc(SCRAPGPU_DTYPE_F32, bo, static_cast<int64_t>(count), 1, SCRAPGPU_LAYOUT_NATURAL);
    if (scrapgpu_tensor_upload(ep->stream, &ta, a, bytes) != SCRAPGPU_STATUS_OK ||
        scrapgpu_tensor_upload(ep->stream, &tb, b, bytes) != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(ba);
        scrapgpu_buffer_free(bb);
        scrapgpu_buffer_free(bo);
        return ort_fail(scrapgpu_last_error());
    }
    const scrapgpu_tensor_desc ins[2] = {ta, tb};
    scrapgpu_op_desc op{};
    op.struct_size = sizeof(op);
    op.op = kind == NodeKind::Add ? SCRAPGPU_OP_ADD : SCRAPGPU_OP_MUL;
    op.inputs = ins;
    op.input_count = 2;
    op.outputs = &to;
    op.output_count = 1;
    op.params = nullptr;
    op.params_size = 0;
    scrapgpu_status stx = scrapgpu_op_execute(ep->stream, &op);
    if (stx != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(ba);
        scrapgpu_buffer_free(bb);
        scrapgpu_buffer_free(bo);
        return ort_fail(scrapgpu_last_error());
    }
    stx = scrapgpu_tensor_download(ep->stream, &to, out, bytes);
    scrapgpu_buffer_free(ba);
    scrapgpu_buffer_free(bb);
    scrapgpu_buffer_free(bo);
    if (stx != SCRAPGPU_STATUS_OK) {
        return ort_fail(scrapgpu_last_error());
    }
    return ort_ok();
}

OrtStatus * run_rms_norm(ScrapGpuEp * ep, ComputeState * st, const float * x, float * y, int64_t rows, int64_t cols) {
    const size_t bytes = static_cast<size_t>(rows * cols) * sizeof(float);
    scrapgpu_buffer_t * bx = nullptr;
    scrapgpu_buffer_t * by = nullptr;
    if (scrapgpu_buffer_alloc(ep->device, bytes, &bx) != SCRAPGPU_STATUS_OK ||
        scrapgpu_buffer_alloc(ep->device, bytes, &by) != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(bx);
        scrapgpu_buffer_free(by);
        return ort_fail("rmsnorm buffer alloc");
    }
    scrapgpu_tensor_desc tx = make_tensor_desc(SCRAPGPU_DTYPE_F32, bx, cols, rows, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_tensor_desc ty = make_tensor_desc(SCRAPGPU_DTYPE_F32, by, cols, rows, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_tensor_desc scale =
        make_tensor_desc(SCRAPGPU_DTYPE_F32, st->scale_buf, st->norm_cols, 1, SCRAPGPU_LAYOUT_NATURAL);
    if (scrapgpu_tensor_upload(ep->stream, &tx, x, bytes) != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(bx);
        scrapgpu_buffer_free(by);
        return ort_fail(scrapgpu_last_error());
    }
    scrapgpu_rms_norm_params rp{};
    rp.struct_size = sizeof(rp);
    rp.eps = st->rms_eps;
    const scrapgpu_tensor_desc ins[2] = {tx, scale};
    scrapgpu_op_desc op{};
    op.struct_size = sizeof(op);
    op.op = SCRAPGPU_OP_RMS_NORM_MUL; // x normalized * scale (RMS_NORM takes a single input)
    op.inputs = ins;
    op.input_count = 2;
    op.outputs = &ty;
    op.output_count = 1;
    op.params = &rp;
    op.params_size = sizeof(rp);
    scrapgpu_status stx = scrapgpu_op_execute(ep->stream, &op);
    if (stx != SCRAPGPU_STATUS_OK) {
        scrapgpu_buffer_free(bx);
        scrapgpu_buffer_free(by);
        return ort_fail(scrapgpu_last_error());
    }
    stx = scrapgpu_tensor_download(ep->stream, &ty, y, bytes);
    scrapgpu_buffer_free(bx);
    scrapgpu_buffer_free(by);
    if (stx != SCRAPGPU_STATUS_OK) {
        return ort_fail(scrapgpu_last_error());
    }
    return ort_ok();
}

bool find_initializer_tensor(const OrtGraph * graph, const char * name, const OrtValue ** out_value) {
    size_t n_init = 0;
    if (g_api->Graph_GetNumInitializers(graph, &n_init) != nullptr) {
        return false;
    }
    std::vector<const OrtValueInfo *> inits(n_init);
    if (n_init > 0 && g_api->Graph_GetInitializers(graph, inits.data(), n_init) != nullptr) {
        return false;
    }
    for (const OrtValueInfo * vi : inits) {
        const char * iname = nullptr;
        if (g_api->GetValueInfoName(vi, &iname) != nullptr) {
            continue;
        }
        if (std::strcmp(iname, name) != 0) {
            continue;
        }
        const OrtValue * val = nullptr;
        if (g_api->ValueInfo_GetInitializerValue(vi, &val) != nullptr) {
            return false;
        }
        *out_value = val;
        return true;
    }
    return false;
}

bool get_ort_value_dims(const OrtValue * value, std::vector<int64_t> & dims) {
    OrtTensorTypeAndShapeInfo * tinfo = nullptr;
    if (g_api->GetTensorTypeAndShape(const_cast<OrtValue *>(value), &tinfo) != nullptr) {
        return false;
    }
    size_t n = 0;
    if (g_api->GetDimensionsCount(tinfo, &n) != nullptr) {
        g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
        return false;
    }
    dims.resize(n);
    if (n > 0 && g_api->GetDimensions(tinfo, dims.data(), n) != nullptr) {
        g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
        return false;
    }
    g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
    return true;
}

bool read_f32_tensor(const OrtValue * value, std::vector<float> & data, std::vector<int64_t> & dims) {
    OrtTensorTypeAndShapeInfo * tinfo = nullptr;
    if (g_api->GetTensorTypeAndShape(const_cast<OrtValue *>(value), &tinfo) != nullptr) {
        return false;
    }
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (g_api->GetTensorElementType(tinfo, &et) != nullptr || et != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
        return false;
    }
    size_t n = 0;
    if (g_api->GetDimensionsCount(tinfo, &n) != nullptr) {
        g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
        return false;
    }
    dims.resize(n);
    if (n > 0 && g_api->GetDimensions(tinfo, dims.data(), n) != nullptr) {
        g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
        return false;
    }
    size_t count = 0;
    if (g_api->GetTensorShapeElementCount(tinfo, &count) != nullptr) {
        g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
        return false;
    }
    g_api->ReleaseTensorTypeAndShapeInfo(tinfo);
    data.resize(count);
    float * raw = nullptr;
    if (g_api->GetTensorMutableData(const_cast<OrtValue *>(value), reinterpret_cast<void **>(&raw)) != nullptr) {
        return false;
    }
    std::memcpy(data.data(), raw, count * sizeof(float));
    return true;
}

bool try_claim_matmul(const OrtGraph * graph, const OrtNode * node) {
    const char * domain = nullptr;
    const char * op = nullptr;
    if (g_api->Node_GetDomain(node, &domain) != nullptr || !is_empty_domain(domain)) {
        return false;
    }
    if (g_api->Node_GetOperatorType(node, &op) != nullptr || std::strcmp(op, "MatMul") != 0) {
        return false;
    }
    size_t n_in = 0;
    if (g_api->Node_GetNumInputs(node, &n_in) != nullptr || n_in != 2) {
        return false;
    }
    std::vector<const OrtValueInfo *> inputs(2);
    if (g_api->Node_GetInputs(node, inputs.data(), 2) != nullptr) {
        return false;
    }
    std::vector<int64_t> da, db;
    if (!get_value_info_dims(inputs[0], da) || !get_value_info_dims(inputs[1], db)) {
        return false;
    }
    const char * bname = nullptr;
    if (g_api->GetValueInfoName(inputs[1], &bname) != nullptr) {
        return false;
    }
    bool is_const = false;
    if (g_api->ValueInfo_IsConstantInitializer(inputs[1], &is_const) != nullptr || !is_const) {
        return false;
    }
    const OrtValue * bval = nullptr;
    if (!find_initializer_tensor(graph, bname, &bval)) {
        return false;
    }
    std::vector<float> bdata;
    std::vector<int64_t> bd;
    if (!read_f32_tensor(bval, bdata, bd) || bd.size() != 2) {
        return false;
    }
    if (da.size() == 1) {
        if (da[0] != bd[0]) {
            return false;
        }
    } else if (da.size() == 2) {
        if (da[1] != bd[0]) {
            return false;
        }
    } else {
        return false;
    }
    if (bd[1] <= 0 || bd[0] <= 0) {
        return false;
    }
    return true;
}

bool try_claim_binary(const OrtNode * node, const char * op_name) {
    const char * domain = nullptr;
    const char * op = nullptr;
    if (g_api->Node_GetDomain(node, &domain) != nullptr || !is_empty_domain(domain)) {
        return false;
    }
    if (g_api->Node_GetOperatorType(node, &op) != nullptr || std::strcmp(op, op_name) != 0) {
        return false;
    }
    size_t n_in = 0, n_out = 0;
    if (g_api->Node_GetNumInputs(node, &n_in) != nullptr || n_in != 2) {
        return false;
    }
    if (g_api->Node_GetNumOutputs(node, &n_out) != nullptr || n_out != 1) {
        return false;
    }
    std::vector<const OrtValueInfo *> inputs(2);
    std::vector<const OrtValueInfo *> outputs(1);
    if (g_api->Node_GetInputs(node, inputs.data(), 2) != nullptr ||
        g_api->Node_GetOutputs(node, outputs.data(), 1) != nullptr) {
        return false;
    }
    std::vector<int64_t> a, b, o;
    if (!get_value_info_dims(inputs[0], a) || !get_value_info_dims(inputs[1], b) ||
        !get_value_info_dims(outputs[0], o)) {
        return false;
    }
    return dims_equal(a, b) && dims_equal(a, o);
}

bool try_claim_rmsnorm(const OrtGraph * graph, const OrtNode * node) {
    const char * domain = nullptr;
    const char * op = nullptr;
    if (g_api->Node_GetDomain(node, &domain) != nullptr || !is_empty_domain(domain)) {
        return false;
    }
    if (g_api->Node_GetOperatorType(node, &op) != nullptr) {
        return false;
    }
    if (std::strcmp(op, "SimplifiedLayerNormalization") != 0 && std::strcmp(op, "RMSNormalization") != 0) {
        return false;
    }
    size_t n_in = 0;
    if (g_api->Node_GetNumInputs(node, &n_in) != nullptr || n_in < 2) {
        return false;
    }
    std::vector<const OrtValueInfo *> inputs(n_in);
    if (g_api->Node_GetInputs(node, inputs.data(), n_in) != nullptr) {
        return false;
    }
    std::vector<int64_t> xdims, sdims;
    if (!get_value_info_dims(inputs[0], xdims) || xdims.empty()) {
        return false;
    }
    const char * sname = nullptr;
    if (g_api->GetValueInfoName(inputs[1], &sname) != nullptr) {
        return false;
    }
    bool is_const = false;
    if (g_api->ValueInfo_IsConstantInitializer(inputs[1], &is_const) != nullptr || !is_const) {
        return false;
    }
    const OrtValue * sval = nullptr;
    if (!find_initializer_tensor(graph, sname, &sval)) {
        return false;
    }
    std::vector<float> sdata;
    if (!read_f32_tensor(sval, sdata, sdims) || sdims.size() != 1) {
        return false;
    }
    if (sdims[0] != xdims.back()) {
        return false;
    }
    return true;
}

struct NodeComputeInfoEx {
    OrtNodeComputeInfo info{};
    ComputeState * state = nullptr;
};

const char * ORT_API_CALL ep_get_name(_In_ const OrtEp * /*this_ptr*/) noexcept { return SGL_ORT_EP_NAME; }

OrtStatus * ORT_API_CALL ep_get_capability(_In_ OrtEp * /*this_ptr*/, _In_ const OrtGraph * graph,
                                           _Inout_ OrtEpGraphSupportInfo * graph_support_info) noexcept {
    size_t n_nodes = 0;
    if (g_api->Graph_GetNumNodes(graph, &n_nodes) != nullptr) {
        return ort_fail("Graph_GetNumNodes");
    }
    std::vector<const OrtNode *> nodes(n_nodes);
    if (n_nodes > 0 && g_api->Graph_GetNodes(graph, nodes.data(), n_nodes) != nullptr) {
        return ort_fail("Graph_GetNodes");
    }
    OrtNodeFusionOptions fuse_opts{};
    fuse_opts.ort_version_supported = ORT_API_VERSION;
    fuse_opts.drop_constant_initializers = true;
    for (const OrtNode * node : nodes) {
        const char * op = nullptr;
        if (g_api->Node_GetOperatorType(node, &op) != nullptr) {
            continue;
        }
        bool claim = false;
        if (try_claim_matmul(graph, node)) {
            claim = true;
        } else if (op != nullptr && std::strcmp(op, "Add") == 0 && try_claim_binary(node, "Add")) {
            claim = true;
        } else if (op != nullptr && std::strcmp(op, "Mul") == 0 && try_claim_binary(node, "Mul")) {
            claim = true;
        } else if (try_claim_rmsnorm(graph, node)) {
            claim = true;
        }
        if (!claim) {
            continue;
        }
        note_claimed(op != nullptr ? op : "?");
        OrtStatus * st =
            g_ep_api->EpGraphSupportInfo_AddNodesToFuse(graph_support_info, &node, 1, &fuse_opts);
        if (st != nullptr) {
            return st;
        }
    }
    return ort_ok();
}

OrtStatus * compile_one(ScrapGpuEp * ep, const OrtGraph * graph, OrtNodeComputeInfo ** out_info) {
    size_t n_nodes = 0;
    if (g_api->Graph_GetNumNodes(graph, &n_nodes) != nullptr || n_nodes != 1) {
        return ort_fail("expected single-node graph");
    }
    const OrtNode * nodes_arr[1] = {nullptr};
    if (g_api->Graph_GetNodes(graph, nodes_arr, 1) != nullptr) {
        return ort_fail("Graph_GetNodes");
    }
    const OrtNode * node = nodes_arr[0];
    const char * op = nullptr;
    if (g_api->Node_GetOperatorType(node, &op) != nullptr) {
        return ort_fail("Node_GetOperatorType");
    }

    auto * state = new ComputeState();
    state->ep = ep;
    state->device = ep->device;
    state->stream = ep->stream;

    if (std::strcmp(op, "MatMul") == 0) {
        state->kind = NodeKind::MatMul;
        size_t n_in = 0;
        if (g_api->Node_GetNumInputs(node, &n_in) != nullptr || n_in != 2) {
            delete state;
            return ort_fail("MatMul input count");
        }
        std::vector<const OrtValueInfo *> inputs(2);
        if (g_api->Node_GetInputs(node, inputs.data(), 2) != nullptr) {
            delete state;
            return ort_fail("MatMul inputs");
        }
        const char * bname = nullptr;
        if (g_api->GetValueInfoName(inputs[1], &bname) != nullptr) {
            delete state;
            return ort_fail("MatMul weight name");
        }
        const OrtValue * bval = nullptr;
        if (!find_initializer_tensor(graph, bname, &bval)) {
            delete state;
            return ort_fail("MatMul missing weight initializer");
        }
        std::vector<float> bdata;
        std::vector<int64_t> bd;
        if (!read_f32_tensor(bval, bdata, bd) || bd.size() != 2) {
            delete state;
            return ort_fail("MatMul weight read");
        }
        const int64_t k = bd[0];
        const int64_t n = bd[1];
        if (upload_weight(ep, state, bdata.data(), k, n) != SCRAPGPU_STATUS_OK) {
            free_compute_state(state);
            return ort_fail(scrapgpu_last_error());
        }
    } else if (std::strcmp(op, "Add") == 0) {
        state->kind = NodeKind::Add;
    } else if (std::strcmp(op, "Mul") == 0) {
        state->kind = NodeKind::Mul;
    } else if (std::strcmp(op, "SimplifiedLayerNormalization") == 0 || std::strcmp(op, "RMSNormalization") == 0) {
        state->kind = NodeKind::RmsNorm;
        size_t n_in = 0;
        if (g_api->Node_GetNumInputs(node, &n_in) != nullptr || n_in < 2) {
            delete state;
            return ort_fail("RMSNorm input count");
        }
        std::vector<const OrtValueInfo *> inputs(n_in);
        if (g_api->Node_GetInputs(node, inputs.data(), n_in) != nullptr) {
            delete state;
            return ort_fail("RMSNorm inputs");
        }
        const char * sname = nullptr;
        if (g_api->GetValueInfoName(inputs[1], &sname) != nullptr) {
            delete state;
            return ort_fail("RMSNorm scale name");
        }
        const OrtValue * sval = nullptr;
        if (!find_initializer_tensor(graph, sname, &sval)) {
            delete state;
            return ort_fail("RMSNorm missing scale");
        }
        std::vector<float> sdata;
        std::vector<int64_t> sd;
        if (!read_f32_tensor(sval, sdata, sd) || sd.size() != 1) {
            delete state;
            return ort_fail("RMSNorm scale read");
        }
        state->norm_cols = sd[0];
        state->rms_eps = 1e-5f;
        const OrtOpAttr * eps_attr = nullptr;
        if (g_api->Node_GetAttributeByName(node, "epsilon", &eps_attr) == nullptr && eps_attr != nullptr) {
            OrtValue * eps_tensor = nullptr;
            if (g_api->OpAttr_GetTensorAttributeAsOrtValue(eps_attr, &eps_tensor) == nullptr && eps_tensor != nullptr) {
                float * p = nullptr;
                if (g_api->GetTensorMutableData(eps_tensor, reinterpret_cast<void **>(&p)) == nullptr && p != nullptr) {
                    state->rms_eps = p[0];
                }
                g_api->ReleaseValue(eps_tensor);
            }
        }
        if (scrapgpu_buffer_alloc(ep->device, sdata.size() * sizeof(float), &state->scale_buf) != SCRAPGPU_STATUS_OK) {
            free_compute_state(state);
            return ort_fail("RMSNorm scale buffer");
        }
        scrapgpu_tensor_desc scale =
            make_tensor_desc(SCRAPGPU_DTYPE_F32, state->scale_buf, state->norm_cols, 1, SCRAPGPU_LAYOUT_NATURAL);
        if (scrapgpu_tensor_upload(ep->stream, &scale, sdata.data(), sdata.size() * sizeof(float)) !=
            SCRAPGPU_STATUS_OK) {
            free_compute_state(state);
            return ort_fail(scrapgpu_last_error());
        }
    } else {
        delete state;
        return ort_fail("unsupported op in compile");
    }

    auto * info_ex = new NodeComputeInfoEx();
    info_ex->state = state;
    info_ex->info.ort_version_supported = ORT_API_VERSION;
    info_ex->info.CreateState = [](OrtNodeComputeInfo * self, OrtNodeComputeContext * /*ctx*/, void ** compute_state) -> OrtStatus * {
        auto * ex = reinterpret_cast<NodeComputeInfoEx *>(self);
        *compute_state = ex->state;
        return ort_ok();
    };
    info_ex->info.Compute = [](OrtNodeComputeInfo * /*self*/, void * compute_state, OrtKernelContext * context) -> OrtStatus * {
        auto * st = static_cast<ComputeState *>(compute_state);
        ScrapGpuEp * ep = st->ep;
        size_t n_in = 0, n_out = 0;
        if (g_api->KernelContext_GetInputCount(context, &n_in) != nullptr ||
            g_api->KernelContext_GetOutputCount(context, &n_out) != nullptr) {
            return ort_fail("kernel IO count");
        }
        if (st->kind == NodeKind::MatMul) {
            if (n_in < 1 || n_out < 1) {
                return ort_fail("MatMul IO count");
            }
            const OrtValue * in0 = nullptr;
            if (g_api->KernelContext_GetInput(context, 0, &in0) != nullptr) {
                return ort_fail("MatMul GetInput");
            }
            std::vector<int64_t> ad;
            if (!get_ort_value_dims(in0, ad)) {
                return ort_fail("MatMul input dims");
            }
            int64_t m = 1;
            int64_t k = st->weight_k;
            const int64_t n = st->weight_n;
            if (ad.size() == 1) {
                k = ad[0];
            } else if (ad.size() == 2) {
                m = ad[0];
                k = ad[1];
            } else {
                return ort_fail("MatMul bad input rank");
            }
            // the runtime shape is the contract of the claimed op (audit M-1): the K the
            // node runs with must be the K the weight was uploaded for, and no dim may be
            // negative (a symbolic dim that leaked through)
            if (m < 0 || k < 0) {
                return ort_fail("MatMul input has an invalid dim");
            }
            if (k != st->weight_k_in) {
                return ort_fail("MatMul input K does not match the weight");
            }
            std::vector<int64_t> out_dims = ad;
            if (out_dims.size() == 1) {
                out_dims = {n};
            } else {
                out_dims.back() = n;
            }
            OrtValue * out0 = nullptr;
            if (g_api->KernelContext_GetOutput(context, 0, out_dims.data(), out_dims.size(), &out0) != nullptr) {
                return ort_fail("MatMul GetOutput");
            }
            float * act = nullptr;
            float * out = nullptr;
            if (g_api->GetTensorMutableData(const_cast<OrtValue *>(in0), reinterpret_cast<void **>(&act)) != nullptr ||
                g_api->GetTensorMutableData(out0, reinterpret_cast<void **>(&out)) != nullptr) {
                return ort_fail("MatMul tensor data");
            }
            OrtStatus * r = run_mul_mat(ep, st, act, m, k, out, n);
            if (r == nullptr) {
                scrapgpu_stream_sync(ep->stream);
            }
            return r;
        }
        if (st->kind == NodeKind::Add || st->kind == NodeKind::Mul) {
            if (n_in < 1 || n_out < 1) {
                return ort_fail("binary IO count");
            }
            const OrtValue * in0 = nullptr;
            const OrtValue * in1 = nullptr;
            if (g_api->KernelContext_GetInput(context, 0, &in0) != nullptr) {
                return ort_fail("binary GetInput");
            }
            if (n_in >= 2) {
                if (g_api->KernelContext_GetInput(context, 1, &in1) != nullptr) {
                    return ort_fail("binary GetInput");
                }
            } else {
                in1 = in0;
            }
            std::vector<int64_t> dims, dims1;
            if (!get_ort_value_dims(in0, dims) || !get_ort_value_dims(in1, dims1)) {
                return ort_fail("binary dims");
            }
            // the claim compared the *declared* shapes, which can be symbolic: the second
            // operand's runtime shape must really match the first's before count elements are
            // read from it (audit M-1)
            if (!dims_equal(dims, dims1)) {
                return ort_fail("binary input shapes do not match at run time");
            }
            size_t count = 1;
            for (int64_t d : dims) {
                if (d < 0 || (d > 0 && count > std::numeric_limits<size_t>::max() / static_cast<size_t>(d))) {
                    return ort_fail("binary input has an invalid dim");
                }
                count *= static_cast<size_t>(d);
            }
            OrtValue * out0 = nullptr;
            if (g_api->KernelContext_GetOutput(context, 0, dims.data(), dims.size(), &out0) != nullptr) {
                return ort_fail("binary GetOutput");
            }
            float * a = nullptr;
            float * b = nullptr;
            float * o = nullptr;
            if (g_api->GetTensorMutableData(const_cast<OrtValue *>(in0), reinterpret_cast<void **>(&a)) != nullptr ||
                g_api->GetTensorMutableData(const_cast<OrtValue *>(in1), reinterpret_cast<void **>(&b)) != nullptr ||
                g_api->GetTensorMutableData(out0, reinterpret_cast<void **>(&o)) != nullptr) {
                return ort_fail("binary tensor data");
            }
            OrtStatus * r = run_binary(ep, st->kind, a, b, o, count);
            if (r == nullptr) {
                scrapgpu_stream_sync(ep->stream);
            }
            return r;
        }
        if (st->kind == NodeKind::RmsNorm) {
            if (n_in < 1 || n_out < 1) {
                return ort_fail("rmsnorm IO count");
            }
            const OrtValue * in0 = nullptr;
            if (g_api->KernelContext_GetInput(context, 0, &in0) != nullptr) {
                return ort_fail("rmsnorm GetInput");
            }
            std::vector<int64_t> dims;
            if (!get_ort_value_dims(in0, dims) || dims.empty()) {
                return ort_fail("rmsnorm dims");
            }
            // the runtime shape is the contract of the claimed op (audit M-1): the last dim must
            // match the uploaded scale and no dim may be negative
            for (int64_t d : dims) {
                if (d < 0) {
                    return ort_fail("rmsnorm input has an invalid dim");
                }
            }
            const int64_t cols = dims.back();
            if (cols != st->norm_cols) {
                return ort_fail("rmsnorm last dim does not match the scale");
            }
            int64_t rows = 1;
            for (size_t i = 0; i + 1 < dims.size(); ++i) {
                if (dims[i] > 0 && rows > std::numeric_limits<int64_t>::max() / dims[i]) {
                    return ort_fail("rmsnorm input is too large");
                }
                rows *= dims[i];
            }
            OrtValue * out0 = nullptr;
            if (g_api->KernelContext_GetOutput(context, 0, dims.data(), dims.size(), &out0) != nullptr) {
                return ort_fail("rmsnorm GetOutput");
            }
            float * x = nullptr;
            float * y = nullptr;
            if (g_api->GetTensorMutableData(const_cast<OrtValue *>(in0), reinterpret_cast<void **>(&x)) != nullptr ||
                g_api->GetTensorMutableData(out0, reinterpret_cast<void **>(&y)) != nullptr) {
                return ort_fail("rmsnorm tensor data");
            }
            OrtStatus * r = run_rms_norm(ep, st, x, y, rows, cols);
            if (r == nullptr) {
                scrapgpu_stream_sync(ep->stream);
            }
            return r;
        }
        return ort_fail("unknown compute kind");
    };
    info_ex->info.ReleaseState = [](OrtNodeComputeInfo * /*self*/, void * /*compute_state*/) {};
    *out_info = &info_ex->info;
    return ort_ok();
}

OrtStatus * ORT_API_CALL ep_compile(_In_ OrtEp * this_ptr, _In_ const OrtGraph ** graphs, _In_ const OrtNode ** /*fused_nodes*/,
                                    _In_ size_t count, _Out_writes_all_(count) OrtNodeComputeInfo ** node_compute_infos,
                                    _Out_writes_(count) OrtNode ** /*ep_context_nodes*/) noexcept {
    auto * ep = reinterpret_cast<ScrapGpuEp *>(this_ptr);
    for (size_t i = 0; i < count; ++i) {
        OrtStatus * st = compile_one(ep, graphs[i], &node_compute_infos[i]);
        if (st != nullptr) {
            return st;
        }
    }
    return ort_ok();
}

void ORT_API_CALL ep_release_node_compute_infos(_In_ OrtEp * /*this_ptr*/, OrtNodeComputeInfo ** node_compute_infos,
                                                _In_ size_t num_node_compute_infos) noexcept {
    for (size_t i = 0; i < num_node_compute_infos; ++i) {
        auto * ex = reinterpret_cast<NodeComputeInfoEx *>(node_compute_infos[i]);
        free_compute_state(ex->state);
        delete ex;
    }
}

void ORT_API_CALL factory_release_ep(OrtEpFactory * /*this_ptr*/, OrtEp * ep) noexcept {
    auto * sgl = reinterpret_cast<ScrapGpuEp *>(ep);
    if (sgl->stream != nullptr) {
        scrapgpu_stream_free(sgl->stream);
    }
    delete sgl;
}

const char * ORT_API_CALL factory_get_name(const OrtEpFactory * /*this_ptr*/) noexcept { return SGL_ORT_EP_NAME; }

const char * ORT_API_CALL factory_get_vendor(const OrtEpFactory * /*this_ptr*/) noexcept { return SGL_ORT_EP_VENDOR; }

uint32_t ORT_API_CALL factory_get_vendor_id(const OrtEpFactory * /*this_ptr*/) noexcept { return 0; }

const char * ORT_API_CALL factory_get_version(const OrtEpFactory * /*this_ptr*/) noexcept { return SGL_ORT_EP_VERSION; }

bool ORT_API_CALL factory_is_stream_aware(const OrtEpFactory * /*this_ptr*/) noexcept { return false; }

OrtStatus * ORT_API_CALL factory_validate_compatibility(
    OrtEpFactory * /*this_ptr*/, const OrtHardwareDevice * const * /*devices*/, size_t /*num_devices*/,
    const char * /*compatibility_info*/, OrtCompiledModelCompatibility * model_compatibility) noexcept {
    *model_compatibility = OrtCompiledModelCompatibility_EP_NOT_APPLICABLE;
    return ort_ok();
}

OrtStatus * ORT_API_CALL factory_get_num_custom_op_domains(OrtEpFactory * /*this_ptr*/, size_t * num_domains) noexcept {
    *num_domains = 0;
    return ort_ok();
}

OrtStatus * ORT_API_CALL factory_get_custom_op_domains(OrtEpFactory * /*this_ptr*/, OrtCustomOpDomain ** /*domains*/,
                                                       size_t /*num_domains*/) noexcept {
    return ort_ok();
}

OrtStatus * ORT_API_CALL factory_get_supported_devices(OrtEpFactory * this_ptr,
                                                       _In_reads_(num_devices) const OrtHardwareDevice * const * devices,
                                                       _In_ size_t num_devices, _Inout_ OrtEpDevice ** ep_devices,
                                                       _In_ size_t max_ep_devices, _Out_ size_t * num_ep_devices) noexcept {
    *num_ep_devices = 0;
    if (g_api == nullptr || g_ep_api == nullptr || num_devices == 0) {
        return ort_ok();
    }
    ensure_scrapgpu_registered();
    const size_t n = scrapgpu_get_device_count();
    if (n == 0) {
        return ort_ok();
    }
    const OrtHardwareDevice * hw = devices[0];
    for (size_t i = 0; i < n && *num_ep_devices < max_ep_devices; ++i) {
        char idx[16];
        std::snprintf(idx, sizeof(idx), "%zu", i);
        OrtKeyValuePairs * meta = nullptr;
        g_api->CreateKeyValuePairs(&meta);
        g_api->AddKeyValuePair(meta, "scrapgpu.device_index", idx);
        OrtEpDevice * ep_dev = nullptr;
        OrtStatus * cst = g_ep_api->CreateEpDevice(this_ptr, hw, meta, nullptr, &ep_dev);
        g_api->ReleaseKeyValuePairs(meta);
        if (cst != nullptr) {
            g_api->ReleaseStatus(cst);
            continue;
        }
        ep_devices[*num_ep_devices] = ep_dev;
        (*num_ep_devices)++;
    }
    return ort_ok();
}

OrtStatus * ORT_API_CALL factory_create_ep(_In_ OrtEpFactory * this_ptr,
                                           _In_reads_(num_devices) const OrtHardwareDevice * const * devices,
                                           _In_reads_(num_devices) const OrtKeyValuePairs * const * ep_metadata_pairs,
                                           _In_ size_t num_devices, _In_ const OrtSessionOptions * session_options,
                                           _In_ const OrtLogger * /*logger*/, _Outptr_ OrtEp ** ep) noexcept {
    if (num_devices == 0) {
        return ort_fail("no devices");
    }
    uint32_t device_index = devices[0] != nullptr ? g_api->HardwareDevice_DeviceId(devices[0]) : 0;
    if (ep_metadata_pairs != nullptr && ep_metadata_pairs[0] != nullptr) {
        const char * idx = g_api->GetKeyValue(ep_metadata_pairs[0], "scrapgpu.device_index");
        if (idx != nullptr) {
            device_index = static_cast<uint32_t>(std::strtoul(idx, nullptr, 10));
        }
    }
    ensure_scrapgpu_registered();
    scrapgpu_device_t * dev = nullptr;
    if (scrapgpu_get_device(device_index, &dev) != SCRAPGPU_STATUS_OK) {
        return ort_fail("scrapgpu_get_device");
    }
    scrapgpu_stream_t * stream = nullptr;
    if (scrapgpu_stream_create(dev, &stream) != SCRAPGPU_STATUS_OK) {
        return ort_fail(scrapgpu_last_error());
    }
    auto * sgl = new ScrapGpuEp();
    sgl->factory = reinterpret_cast<ScrapGpuEpFactory *>(this_ptr);
    sgl->device_index = device_index;
    sgl->device = dev;
    sgl->stream = stream;
    sgl->session_weight_dtype = parse_weight_dtype(session_options);
    sgl->iface.ort_version_supported = ORT_API_VERSION;
    sgl->iface.GetName = ep_get_name;
    sgl->iface.GetCapability = ep_get_capability;
    sgl->iface.Compile = ep_compile;
    sgl->iface.ReleaseNodeComputeInfos = ep_release_node_compute_infos;
    *ep = &sgl->iface;
    return ort_ok();
}

} // namespace


// The optional parts of OrtEpFactory. ORT 1.31 calls some of them without a null check (CreateDataTransfer right after
// the factory is registered), so every member gets a function: no allocator, no data transfer and no stream (the EP keeps
// the activations in CPU memory and copies them itself), not implemented for the rest.
OrtStatus * ORT_API_CALL factory_create_allocator(OrtEpFactory * /*this_ptr*/, const OrtMemoryInfo * /*memory_info*/,
                                                  const OrtKeyValuePairs * /*allocator_options*/, OrtAllocator ** allocator) noexcept {
    *allocator = nullptr;
    return nullptr;
}

void ORT_API_CALL factory_release_allocator(OrtEpFactory * /*this_ptr*/, OrtAllocator * /*allocator*/) noexcept {}

OrtStatus * ORT_API_CALL factory_create_data_transfer(OrtEpFactory * /*this_ptr*/, OrtDataTransferImpl ** data_transfer) noexcept {
    *data_transfer = nullptr;
    return nullptr;
}

OrtStatus * ORT_API_CALL factory_create_sync_stream(OrtEpFactory * /*this_ptr*/, const OrtMemoryDevice * /*memory_device*/,
                                                    const OrtKeyValuePairs * /*stream_options*/, OrtSyncStreamImpl ** stream) noexcept {
    *stream = nullptr;
    return nullptr;
}

OrtStatus * ORT_API_CALL factory_get_hw_incompatibility(OrtEpFactory * /*this_ptr*/, const OrtHardwareDevice * /*hw*/,
                                                        OrtDeviceEpIncompatibilityDetails * /*details*/) noexcept {
    return nullptr; // no details: the device list of GetSupportedDevices is the answer
}

OrtStatus * ORT_API_CALL factory_create_importer(OrtEpFactory * /*this_ptr*/, const OrtEpDevice * /*ep_device*/,
                                                 OrtExternalResourceImporterImpl ** out_importer) noexcept {
    *out_importer = nullptr;
    return nullptr;
}

OrtStatus * ORT_API_CALL factory_init_graphics_interop(OrtEpFactory * /*this_ptr*/, const OrtEpDevice * /*ep_device*/,
                                                       const OrtGraphicsInteropConfig * /*config*/) noexcept {
    return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, "ScrapGpu EP: no graphics interop");
}

OrtStatus * ORT_API_CALL factory_deinit_graphics_interop(OrtEpFactory * /*this_ptr*/, const OrtEpDevice * /*ep_device*/) noexcept {
    return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, "ScrapGpu EP: no graphics interop");
}

OrtStatus * ORT_API_CALL factory_select_best_model(OrtEpFactory * /*this_ptr*/, const OrtHardwareDevice * /*device*/,
                                                   const OrtKeyValuePairs * const * /*candidates*/, size_t /*num_candidates*/,
                                                   const OrtSessionOptions * /*session_options*/, size_t * selected_index) noexcept {
    *selected_index = 0;
    return g_api->CreateStatus(ORT_NOT_IMPLEMENTED, "ScrapGpu EP: no model candidate selection");
}

extern "C" {

SGL_ORT_EP_EXPORT OrtStatus * ORT_API_CALL CreateEpFactories(const char * /*registered_name*/, const OrtApiBase * ort_api_base,
                                           const OrtLogger * /*default_logger*/, OrtEpFactory ** factories,
                                           size_t max_factories, size_t * num_factories) noexcept {
    if (max_factories < 1) {
        return nullptr;
    }
    g_api = ort_api_base->GetApi(ORT_API_VERSION);
    if (g_api == nullptr) {
        return nullptr;
    }
    g_ep_api = g_api->GetEpApi();
    if (g_ep_api == nullptr) {
        return g_api->CreateStatus(ORT_FAIL, "GetEpApi returned null");
    }
    auto * factory = new ScrapGpuEpFactory();
    std::memset(&factory->iface, 0, sizeof(factory->iface));
    factory->iface.ort_version_supported = ORT_API_VERSION;
    factory->iface.GetName = factory_get_name;
    factory->iface.GetVendor = factory_get_vendor;
    factory->iface.GetVendorId = factory_get_vendor_id;
    factory->iface.GetVersion = factory_get_version;
    factory->iface.GetSupportedDevices = factory_get_supported_devices;
    factory->iface.CreateEp = factory_create_ep;
    factory->iface.ReleaseEp = factory_release_ep;
    factory->iface.ValidateCompiledModelCompatibilityInfo = factory_validate_compatibility;
    factory->iface.IsStreamAware = factory_is_stream_aware;
    factory->iface.GetNumCustomOpDomains = factory_get_num_custom_op_domains;
    factory->iface.GetCustomOpDomains = factory_get_custom_op_domains;
    factory->iface.CreateAllocator = factory_create_allocator;
    factory->iface.ReleaseAllocator = factory_release_allocator;
    factory->iface.CreateDataTransfer = factory_create_data_transfer;
    factory->iface.CreateSyncStreamForDevice = factory_create_sync_stream;
    factory->iface.GetHardwareDeviceIncompatibilityDetails = factory_get_hw_incompatibility;
    factory->iface.CreateExternalResourceImporterForDevice = factory_create_importer;
    factory->iface.InitGraphicsInterop = factory_init_graphics_interop;
    factory->iface.DeinitGraphicsInterop = factory_deinit_graphics_interop;
    factory->iface.SelectBestModelCandidate = factory_select_best_model;
    factories[0] = &factory->iface;
    *num_factories = 1;
    return nullptr;
}

SGL_ORT_EP_EXPORT OrtStatus * ORT_API_CALL ReleaseEpFactory(OrtEpFactory * factory) noexcept {
    if (factory == nullptr) {
        return nullptr;
    }
    delete reinterpret_cast<ScrapGpuEpFactory *>(factory);
    return nullptr;
}

} // extern "C"
