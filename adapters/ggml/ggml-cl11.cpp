// CL11: ggml backend that only needs OpenCL 1.1 (host API and OpenCL C 1.1).
//
// Target: old GPUs such as the NVIDIA GeForce GT 430 (Fermi, "OpenCL 1.1 CUDA"):
//   - no sub-groups, no cl_khr_fp16, no doubles in the kernels,
//   - clCreateBuffer is limited to CL_DEVICE_MAX_MEM_ALLOC_SIZE (1/4 of the memory),
//   - PCIe transfers are slow: the whole graph is enqueued on one in-order queue and the host only
//     synchronizes in synchronize() and get_tensor().
//
// Only the OpenCL 1.1 host API is used (CL_TARGET_OPENCL_VERSION 110): no clEnqueueFillBuffer,
// no clEnqueueMarkerWithWaitList, no clCreateProgramWithBuiltInKernels, no clCreateImage, no SVM.

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

#include "ggml-cl11.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "opencl11/buffer.h"
#include "opencl11/context.h"
#include "opencl11/device.h"
#include "opencl11/device_state.h"
#include "opencl11/executor.h"
#include "opencl11/kernels.h"
#include "opencl11/ops.h"
#include "opencl11/r32.h"
#include "opencl11/tensor.h"
#include "tensor/tensor.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <mutex>
#include <string>
#include <vector>

// ggml-alloc asks for the base address of a buffer: the tensors of a CL11 buffer are addressed as
// offsets from this dummy (non-null, page aligned) pointer.
#define CL11_BASE ((void *) 0x1000)

// Device information, context creation, the programs / kernels and the executor state live in src/backends/opencl11/
// (cl11_device). ggml keeps the API objects: the buffer wrapper, the op translation / dispatch and the registry.

// The descriptor passed to the kernels by value (cl11_td / cl11_bcast / cl11_padp) is built from a
// sgt::TensorDesc by the inline helpers of src/backends/opencl11/tensor_abi.h; the tensor math
// (fast division, broadcast factors, contiguity, the R32 eligibility) is in src/tensor.

// The ggml device wrapper: only the ggml objects (buffer type, device, registry) and the runtime state of the device, which lives in
// src/backends/opencl11 (cl11_device: OpenCL context / queue, programs / kernels, tuning, caches, scratch buffers, executor).
struct cl11_device_context {
    std::unique_ptr<cl11_device> device;
    ggml_backend_buffer_type buft = {};
    ggml_backend_device      dev  = {};
};

static std::vector<std::unique_ptr<cl11_device_context>> g_cl11_devices;

static bool cl11_debug() {
    static const bool debug = getenv("GGML_CL11_DEBUG") != nullptr;
    return debug;
}

#define CL11_CHECK(expr)                                                                   \
    do {                                                                                   \
        const cl_int _err = (expr);                                                        \
        if (_err != CL_SUCCESS) {                                                          \
            GGML_LOG_ERROR("%s: %s failed: %s (%d)\n", __func__, #expr, cl11_err_str(_err), (int) _err); \
            return false;                                                                  \
        }                                                                                  \
    } while (0)

//
// device enumeration compatibility shim
//

static void cl11_probe_devices() {
    (void) cl11_probe_device_infos([](const cl11_device_info & info, bool strict_alloc) {
        auto ctx = std::make_unique<cl11_device_context>();
        ctx->device = std::make_unique<cl11_device>();
        ctx->device->info = info;
        ctx->device->strict_alloc_dev = strict_alloc;
        ctx->device->index = (int) g_cl11_devices.size();
        ctx->device->name = "CL11_" + std::to_string(ctx->device->index);
        g_cl11_devices.push_back(std::move(ctx));
    });
}

//
// buffers
//

struct cl11_buffer_context {
    cl11_device * dev = nullptr;
    cl_mem        mem = nullptr;
    size_t        size = 0; // allocated size
};

static ggml_backend_buffer_t cl11_tensor_buffer(const ggml_tensor * t) {
    return t->view_src ? t->view_src->buffer : t->buffer;
}

// byte offset of the tensor data inside of its cl_mem
static size_t cl11_tensor_offset(const ggml_tensor * t) {
    return (size_t) ((const char *) t->data - (const char *) CL11_BASE);
}

// the ggml types this backend handles, as the dtype of the tensor module (the other types are only
// met by tensors of ops the backend does not support)
static scrapgpu_dtype cl11_dtype_of(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return SCRAPGPU_DTYPE_F32;
        case GGML_TYPE_F16:  return SCRAPGPU_DTYPE_F16;
        case GGML_TYPE_I32:  return SCRAPGPU_DTYPE_I32;
        case GGML_TYPE_I64:  return SCRAPGPU_DTYPE_I64;
        case GGML_TYPE_I16:  return SCRAPGPU_DTYPE_I16;
        case GGML_TYPE_BF16: return SCRAPGPU_DTYPE_BF16;
        case GGML_TYPE_Q4_0: return SCRAPGPU_DTYPE_Q4_0;
        case GGML_TYPE_Q8_0: return SCRAPGPU_DTYPE_Q8_0;
        case GGML_TYPE_Q4_K: return SCRAPGPU_DTYPE_Q4_K;
        case GGML_TYPE_Q6_K: return SCRAPGPU_DTYPE_Q6_K;
        default:             return (scrapgpu_dtype) -1;
    }
}

// the tensor descriptor (on the stack) of a ggml_tensor; layout stays NATURAL: whether the tensor is
// stored repacked is decided per use by cl11_repacked()
static sgt::TensorDesc cl11_desc(const ggml_tensor * t) {
    sgt::TensorDesc d;
    d.dtype = cl11_dtype_of(t->type);
    for (int i = 0; i < 4; ++i) {
        d.ne[i] = t->ne[i];
        d.nb[i] = t->nb[i];
    }
    d.offset = (uint64_t) cl11_tensor_offset(t);
    return d;
}

// the backend view (on the stack) of a ggml_tensor: the cl_mem of its buffer and the natural-layout
// descriptor. `repacked` is filled where the kernels need it (the src0 of MUL_MAT / MUL_MAT_ID).
static cl11_tensor cl11_view(const ggml_tensor * t) {
    cl11_tensor v;
    auto * bctx = (cl11_buffer_context *) cl11_tensor_buffer(t)->context;
    v.mem = bctx->mem;
    v.d   = cl11_desc(t);
    return v;
}

// the desc of `t` into `out`, nullptr for a missing input
static const sgt::TensorDesc * cl11_pdesc(const ggml_tensor * t, sgt::TensorDesc & out) {
    if (t == nullptr) {
        return nullptr;
    }
    out = cl11_desc(t);
    return &out;
}

//
// Repacked layout "R32" of the Q4_0 / Q8_0 / Q4_K / Q6_K tensors
//
// set_tensor stores the repackable tensors (when cl11_repacked() is true, that is a property of the tensor alone, so
// supports_op / the kernels / get_tensor agree) in a layout made for coalesced 128-bit loads, get_tensor converts back.
// The layout itself and the host-side pack / unpack / partial transfer are in src/backends/opencl11/r32.{h,cpp}
// (cl11_rp_pack, cl11_rp_unpack, cl11_rp_transfer).
//

static bool cl11_rp_type(ggml_type type) {
    return sgt::dtype_is_repackable(cl11_dtype_of(type));
}

static bool cl11_repacked(const ggml_tensor * t) {
    return t != nullptr && t->view_src == nullptr && sgt::r32_eligible(cl11_desc(t));
}

// a view of a repacked tensor cannot be addressed (the kernels and the transfers know the layout of the whole tensor only)
static void cl11_rp_check_view(const ggml_tensor * t) {
    if (t->view_src != nullptr && cl11_repacked(t->view_src)) {
        GGML_ABORT("CL11: the tensor %s is a view of a repacked quantized tensor (%s), which is not supported\n", t->name, t->view_src->name);
    }
}

static void ggml_backend_cl11_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    {
        std::lock_guard<std::mutex> lock(bctx->dev->mutex);
        cl11_release_const_subs_for_parent(bctx->dev->const_subs, bctx->mem);
    }
    if (bctx->mem) {
        clReleaseMemObject(bctx->mem);
    }
    delete bctx;
}

static void * ggml_backend_cl11_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return CL11_BASE;
}

static void ggml_backend_cl11_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    if (!cl11_ensure_program(bctx->dev) ||
            !cl11_fill(bctx->dev->queue, bctx->dev->exec, bctx->dev->kernels[K_FILL_U32], bctx->dev->kernels[K_FILL_U8],
                       bctx->dev->cfg, bctx->mem, cl11_tensor_offset(tensor) + offset, size, value)) {
        GGML_ABORT("%s: memset failed\n", __func__);
    }
}

static void ggml_backend_cl11_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    cl11_rp_check_view(tensor);
    if (cl11_repacked(tensor)) {
        if (!cl11_rp_transfer(bctx->dev, bctx->mem, cl11_desc(tensor), (uint8_t *) const_cast<void *>(data), offset, size, true)) {
            GGML_ABORT("%s: transfer of the repacked tensor %s failed\n", __func__, tensor->name);
        }
        return;
    }
    // blocking write: it also waits for the kernels that are already enqueued
    const cl_int err = clEnqueueWriteBuffer(bctx->dev->queue, bctx->mem, CL_TRUE, cl11_tensor_offset(tensor) + offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        GGML_ABORT("%s: clEnqueueWriteBuffer failed: %s\n", __func__, cl11_err_str(err));
    }
}

static void ggml_backend_cl11_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    if (size == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    cl11_rp_check_view(tensor);
    if (cl11_repacked(tensor)) {
        if (!cl11_rp_transfer(bctx->dev, bctx->mem, cl11_desc(tensor), (uint8_t *) data, offset, size, false)) {
            GGML_ABORT("%s: transfer of the repacked tensor %s failed\n", __func__, tensor->name);
        }
        return;
    }
    const cl_int err = clEnqueueReadBuffer(bctx->dev->queue, bctx->mem, CL_TRUE, cl11_tensor_offset(tensor) + offset, size, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        GGML_ABORT("%s: clEnqueueReadBuffer failed: %s\n", __func__, cl11_err_str(err));
    }
}

static bool ggml_backend_cl11_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    auto * dctx_buf = (cl11_buffer_context *) buffer->context;
    cl11_device * c = dctx_buf->dev;
    const size_t size = ggml_nbytes(src);
    if (size == 0) {
        return true;
    }

    ggml_backend_buffer_t src_buf = cl11_tensor_buffer(src);
    if (src_buf == nullptr) {
        return false;
    }

    if (ggml_backend_buffer_is_host(src_buf)) {
        // host -> device
        std::lock_guard<std::mutex> lock(c->mutex);
        if (cl11_repacked(dst)) {
            return cl11_rp_transfer(c, dctx_buf->mem, cl11_desc(dst), (uint8_t *) src->data, 0, size, true);
        }
        return clEnqueueWriteBuffer(c->queue, dctx_buf->mem, CL_TRUE, cl11_tensor_offset(dst), size, src->data, 0, nullptr, nullptr) == CL_SUCCESS;
    }

    if (src_buf->iface.free_buffer == ggml_backend_cl11_buffer_free_buffer) {
        if (cl11_repacked(src) != cl11_repacked(dst)) {
            return false; // different layouts: let ggml copy through the host
        }
        auto * sctx = (cl11_buffer_context *) src_buf->context;
        if (sctx->dev != c) {
            return false; // another cl_context: let ggml copy through the host
        }
        const size_t so = cl11_tensor_offset(src);
        const size_t d_o = cl11_tensor_offset(dst);
        if (sctx->mem == dctx_buf->mem && so < d_o + size && d_o < so + size) {
            return false; // overlapping regions are not allowed in clEnqueueCopyBuffer
        }
        std::lock_guard<std::mutex> lock(c->mutex);
        return clEnqueueCopyBuffer(c->queue, sctx->mem, dctx_buf->mem, so, d_o, size, 0, nullptr, nullptr) == CL_SUCCESS;
    }

    return false;
}

static void ggml_backend_cl11_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * bctx = (cl11_buffer_context *) buffer->context;
    std::lock_guard<std::mutex> lock(bctx->dev->mutex);
    if (!cl11_ensure_program(bctx->dev) ||
            !cl11_fill(bctx->dev->queue, bctx->dev->exec, bctx->dev->kernels[K_FILL_U32], bctx->dev->kernels[K_FILL_U8],
                       bctx->dev->cfg, bctx->mem, 0, bctx->size, value)) {
        GGML_ABORT("%s: clear failed\n", __func__);
    }
}

static const ggml_backend_buffer_i ggml_backend_cl11_buffer_i = {
    /* .free_buffer   = */ ggml_backend_cl11_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_cl11_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_cl11_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_cl11_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_cl11_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ ggml_backend_cl11_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_cl11_buffer_clear,
    /* .reset         = */ nullptr,
};

//
// buffer type
//

static const char * ggml_backend_cl11_buft_get_name(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return c->device->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_cl11_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * c = (cl11_device_context *) buft->context;
    cl11_device * dev = c->device.get();
    std::lock_guard<std::mutex> lock(dev->mutex);
    if (!cl11_ensure_context(dev)) {
        return nullptr;
    }

    cl_mem mem = nullptr;
    size_t alloc_size = 0;
    if (!cl11_alloc_buffer(dev->context, dev->queue, dev->info, dev->name.c_str(), __func__, dev->strict_alloc_dev, size, mem, alloc_size)) {
        return nullptr;
    }

    auto * bctx = new cl11_buffer_context{ dev, mem, alloc_size };
    return ggml_backend_buffer_init(buft, ggml_backend_cl11_buffer_i, bctx, size);
}

static size_t ggml_backend_cl11_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return cl11_buffer_alignment(c->device->info);
}

static size_t ggml_backend_cl11_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * c = (cl11_device_context *) buft->context;
    return cl11_max_buffer_size(c->device->info);
}

static bool ggml_backend_cl11_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static const ggml_backend_buffer_type_i ggml_backend_cl11_buft_i = {
    /* .get_name         = */ ggml_backend_cl11_buft_get_name,
    /* .alloc_buffer     = */ ggml_backend_cl11_buft_alloc_buffer,
#if GGML_BACKEND_API_VERSION >= 3 // llama.cpp b11384: alloc_buffer_n / get_alloc_size_n (absent in older trees, e.g. mesh-llm's Skippy)
    /* .alloc_buffer_n   = */ nullptr,
#endif
    /* .get_alignment    = */ ggml_backend_cl11_buft_get_alignment,
    /* .get_max_size     = */ ggml_backend_cl11_buft_get_max_size,
    /* .get_alloc_size   = */ nullptr,
#if GGML_BACKEND_API_VERSION >= 3
    /* .get_alloc_size_n = */ nullptr,
#endif
    /* .is_host          = */ ggml_backend_cl11_buft_is_host,
};

//
// supported operations
//

// the offsets and strides are passed as 32-bit integers
static bool cl11_fits_int(const ggml_tensor * t) {
    if (t == nullptr) {
        return true;
    }
    if (cl11_dtype_of(t->type) == (scrapgpu_dtype) -1) {
        // a type the tensor module does not know (its nbytes would be 0): the same limits with the ggml sizes
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (t->ne[i] > INT_MAX || t->nb[i] > INT_MAX) {
                return false;
            }
        }
        return ggml_nelements(t) <= INT_MAX - 65536 && ggml_nbytes(t) <= (size_t) INT_MAX - 65536;
    }
    return sgt::fits_int32(cl11_desc(t));
}

// GGML_CL11_NO_OPS=NAME,NAME,...: supports_op is false for these operations (ggml_op_desc names, e.g. MUL_MAT_ID,ARGSORT); debugging of mixed CPU / GPU graphs
static bool cl11_op_disabled(const ggml_tensor * op) {
    static const std::string list = [] {
        const char * env = getenv("GGML_CL11_NO_OPS");
        return env != nullptr ? "," + std::string(env) + "," : std::string();
    }();
    return !list.empty() && list.find("," + std::string(ggml_op_desc(op)) + ",") != std::string::npos;
}

// ---- ggml op_params / enums -> the plain values the backend ops take ----

// the k_unary op code of a ggml op (a UNARY kind, SQR, SQRT, SCALE, CLAMP): -1 for anything else
static int32_t cl11_unary_code(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_SILU:    return CL11_UC_SILU;
                case GGML_UNARY_OP_RELU:    return CL11_UC_RELU;
                case GGML_UNARY_OP_NEG:     return CL11_UC_NEG;
                case GGML_UNARY_OP_ABS:     return CL11_UC_ABS;
                case GGML_UNARY_OP_SIGMOID: return CL11_UC_SIGMOID;
                case GGML_UNARY_OP_TANH:    return CL11_UC_TANH;
                default:                    return -1;
            }
        case GGML_OP_SQR:   return CL11_UC_SQR;
        case GGML_OP_SQRT:  return CL11_UC_SQRT;
        case GGML_OP_SCALE: return CL11_UC_SCALE;
        case GGML_OP_CLAMP: return CL11_UC_CLAMP;
        default:            return -1;
    }
}

static cl11_binary_kind cl11_binary_kind_of(ggml_op op) {
    switch (op) {
        case GGML_OP_ADD: return CL11_BINARY_ADD;
        case GGML_OP_SUB: return CL11_BINARY_SUB;
        case GGML_OP_MUL: return CL11_BINARY_MUL;
        default:          return CL11_BINARY_DIV;
    }
}

static scrapgpu_glu_kind cl11_glu_kind(const ggml_tensor * op) {
    switch (ggml_get_glu_op(op)) {
        case GGML_GLU_OP_SWIGLU: return SCRAPGPU_GLU_SWIGLU;
        case GGML_GLU_OP_REGLU:  return SCRAPGPU_GLU_REGLU;
        default:                 return (scrapgpu_glu_kind) -1;
    }
}

// the eps of a RMS_NORM node
static float cl11_eps(const ggml_tensor * op) {
    float eps = 0.0f;
    memcpy(&eps, (const int32_t *) op->op_params + 0, sizeof(float));
    return eps;
}

// the ROPE params decoded from ggml's op_params; the yarn corr dims are filled by the launch path
// (ggml_rope_yarn_corr_dims), the support check does not need them
static void cl11_rope_decode(const ggml_tensor * op, cl11_rope_params & p) {
    const int32_t * params = (const int32_t *) op->op_params;
    p.mode       = params[2] == GGML_ROPE_TYPE_NORMAL ? SCRAPGPU_ROPE_MODE_NORMAL
                 : params[2] == GGML_ROPE_TYPE_NEOX   ? SCRAPGPU_ROPE_MODE_NEOX : (scrapgpu_rope_mode) -1;
    p.n_dims     = params[1];
    p.n_ctx_orig = params[4];
    memcpy(&p.freq_base,   params +  5, sizeof(float));
    memcpy(&p.freq_scale,  params +  6, sizeof(float));
    memcpy(&p.ext_factor,  params +  7, sizeof(float));
    memcpy(&p.attn_factor, params +  8, sizeof(float));
    memcpy(&p.beta_fast,   params +  9, sizeof(float));
    memcpy(&p.beta_slow,   params + 10, sizeof(float));
    p.n_offs = params[15];
    p.corr_dims[0] = 0.0f;
    p.corr_dims[1] = 0.0f;
}

static bool cl11_supports_op_impl(const cl11_device * c, const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        default:
            break;
    }

    if (cl11_op_disabled(op)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (!cl11_fits_int(op->src[i])) {
            return false;
        }
    }
    if (!cl11_fits_int(op)) {
        return false;
    }
    if (c->strict_alloc_dev) {
        // the driver cannot allocate a tensor larger than CL_DEVICE_MAX_MEM_ALLOC_SIZE
        if (ggml_nbytes(op) > cl11_max_buffer_size(c->info)) {
            return false;
        }
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (op->src[i] != nullptr && ggml_nbytes(op->src[i]) > cl11_max_buffer_size(c->info)) {
                return false;
            }
        }
    }

    // the shape / type / size conditions live in the backend (ops_support.cpp); this decodes the
    // op_params and decides the things a descriptor cannot carry (type mapping, repacked layout)
    switch (op->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return cl11_supports_binary(cl11_desc(src0), cl11_desc(src1), cl11_desc(op));
        case GGML_OP_UNARY:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
            return cl11_supports_unary(cl11_desc(src0), cl11_desc(op), cl11_unary_code(op));
        case GGML_OP_GLU: {
            const scrapgpu_glu_params p = { sizeof(p), cl11_glu_kind(op), src1 != nullptr, ggml_get_op_params_i32(op, 1) };
            sgt::TensorDesc d1;
            return cl11_supports_glu(cl11_desc(src0), cl11_pdesc(src1, d1), cl11_desc(op), p);
        }
        case GGML_OP_RMS_NORM:
            return cl11_supports_rms_norm(cl11_desc(src0), cl11_desc(op));
        case GGML_OP_SOFT_MAX: {
            cl11_soft_max_params p;
            memcpy(&p.scale,     (const int32_t *) op->op_params + 0, sizeof(float));
            memcpy(&p.max_bias,  (const int32_t *) op->op_params + 1, sizeof(float));
            sgt::TensorDesc d1, d2;
            return cl11_supports_soft_max(cl11_desc(src0), cl11_pdesc(src1, d1), cl11_pdesc(src2, d2), cl11_desc(op), p);
        }
        case GGML_OP_ROPE: {
            cl11_rope_params p;
            cl11_rope_decode(op, p);
            sgt::TensorDesc d1, d2;
            return cl11_supports_rope(cl11_desc(src0), cl11_pdesc(src1, d1), cl11_pdesc(src2, d2), cl11_desc(op), p);
        }
        case GGML_OP_GET_ROWS:
            return cl11_supports_get_rows(cl11_desc(src0), cl11_repacked(src0), cl11_desc(src1), cl11_desc(op));
        case GGML_OP_SET_ROWS:
            return cl11_supports_set_rows(cl11_desc(src0), cl11_desc(src1), cl11_desc(op));
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return cl11_supports_cpy(cl11_desc(src0), cl11_desc(op));
        case GGML_OP_MUL_MAT:
            return cl11_supports_mul_mat(cl11_desc(src0), cl11_repacked(src0), cl11_desc(src1), cl11_desc(op));
        case GGML_OP_ADD_ID: {
            sgt::TensorDesc d2;
            return cl11_supports_add_id(cl11_desc(src0), cl11_desc(src1), cl11_pdesc(src2, d2), cl11_desc(op));
        }
        case GGML_OP_SSM_CONV:
            return cl11_supports_ssm_conv(cl11_desc(src0), cl11_desc(src1), cl11_desc(op));
        case GGML_OP_CONCAT: {
            const scrapgpu_concat_params p = { sizeof(p), (uint32_t) ggml_get_op_params_i32(op, 0) };
            return cl11_supports_concat(cl11_desc(src0), cl11_desc(src1), cl11_desc(op), p);
        }
        case GGML_OP_PAD:
            return cl11_supports_pad(cl11_desc(src0), cl11_desc(op));
        case GGML_OP_SUM_ROWS:
            return cl11_supports_sum_rows(cl11_desc(src0), cl11_desc(op));
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
            return cl11_supports_argsort(c, cl11_desc(src0), cl11_desc(op));
        case GGML_OP_MUL_MAT_ID: {
            sgt::TensorDesc d2;
            return cl11_supports_mul_mat_id(c, cl11_desc(src0), cl11_repacked(src0), cl11_desc(src1), cl11_pdesc(src2, d2), cl11_desc(op));
        }
        default:
            return false;
    }
}

//
// graph compute
//

// RMS_NORM (F32) followed by a MUL with a weight vector: the weights `w` (ne0 contiguous floats) of the MUL `mul` whose other operand is the
// RMS_NORM node i, which has no other user
static const ggml_tensor * cl11_norm_mul_weights(const cl11_device * c, const ggml_cgraph * cgraph, int i) {
    if (c->no_fuse || i + 1 >= cgraph->n_nodes) {
        return nullptr;
    }
    const ggml_tensor * norm = cgraph->nodes[i];
    const ggml_tensor * mul  = cgraph->nodes[i + 1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || (mul->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        !ggml_node_has_n_uses(cgraph, i, 1) || norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return nullptr;
    }
    const ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : (mul->src[1] == norm ? mul->src[0] : nullptr);
    if (w == nullptr || w == norm || w->type != GGML_TYPE_F32 || ggml_nelements(w) != norm->ne[0] || w->ne[0] != norm->ne[0] ||
        w->nb[0] != sizeof(float) || !ggml_are_same_shape(mul, norm) || mul->nb[0] != sizeof(float) || norm->src[0]->nb[0] != sizeof(float) ||
        norm->src[0]->type != GGML_TYPE_F32 || !cl11_fits_int(w)) {
        return nullptr;
    }
    return w;
}

// ADD (F32, one row, same shapes, no broadcast) + RMS_NORM + MUL: nodes i, i + 1, i + 2 (the ADD result has other users)
static bool cl11_can_fuse_add_norm_mul(const cl11_device * c, const ggml_cgraph * cgraph, int i, const ggml_tensor ** w) {
    if (c->no_fuse || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * add = cgraph->nodes[i];
    const ggml_tensor * norm = cgraph->nodes[i + 1];
    if (add->op != GGML_OP_ADD || norm->op != GGML_OP_RMS_NORM || norm->src[0] != add || (add->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (norm->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const ggml_tensor * a = add->src[0], * b = add->src[1];
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, b) || !ggml_are_same_shape(a, add) ||
        ggml_nelements(add) != add->ne[0] || a->nb[0] != sizeof(float) || b->nb[0] != sizeof(float) || add->nb[0] != sizeof(float) ||
        !cl11_fits_int(a) || !cl11_fits_int(b) || !cl11_fits_int(add)) {
        return false;
    }
    *w = cl11_norm_mul_weights(c, cgraph, i + 1);
    return *w != nullptr && cgraph->nodes[i + 2]->ne[0] == add->ne[0] && ggml_nelements(cgraph->nodes[i + 2]) == add->ne[0];
}

// can the SET_ROWS `b` be computed in the same launch as `a` (which comes before it, only nodes without computation in between)?
static bool cl11_can_pair_set_rows(const cl11_device * c, const ggml_tensor * a, const ggml_tensor * b) {
    if (!c->fuse_set_rows || b->op != GGML_OP_SET_ROWS || (b->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        c->kernels[K_SET_ROWS2 + (a->src[0]->type == GGML_TYPE_F16 ? 2 : 0) + (a->type == GGML_TYPE_F16 ? 1 : 0)] == nullptr) {
        return false;
    }
    if (a->src[0]->type != b->src[0]->type || a->type != b->type) {
        return false;
    }
    auto base = [](const ggml_tensor * t) { return t->view_src ? t->view_src : t; };
    // b must not read what a writes, and they must not write the same tensor
    if (base(a) == base(b)) {
        return false;
    }
    for (int k = 0; k < 2; ++k) {
        if (b->src[k] == a || base(b->src[k]) == base(a)) {
            return false;
        }
    }
    return true;
}

// a node that writes into a column the scratch buffers were made from makes them stale
static void cl11_xprep_note_write(cl11_device * c, const ggml_tensor * node) {
    if (node->data == nullptr) {
        return;
    }
    const cl11_tensor d = cl11_view(node);
    cl11_xprep_note_write(c, d.mem, (size_t) d.d.offset, ggml_nbytes(node));
}

// ---- several MUL_MATs of the same src1 in one launch (k_mvm_*, mul_mv_multi.cl) ----

// static properties only (types and shapes, no buffers: the graph is also looked at before it is allocated, see graph_optimize):
// a MUL_MAT of a repacked weight matrix with a single column of src1 and no batch. The launch-time checks (alignment, work-group limits) are in cl11_op_mul_mat_multi.
static bool cl11_mvm_candidate(const cl11_device * c, const ggml_tensor * node) {
    if (!c->multi_mv || c->mvr_const_x || c->no_fuse || node->op != GGML_OP_MUL_MAT || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return false;
    }
    const ggml_tensor * a = node->src[0];
    const ggml_tensor * b = node->src[1];
    if (a == nullptr || b == nullptr || !cl11_rp_type(a->type) || a->op != GGML_OP_NONE || a->view_src != nullptr || b->type != GGML_TYPE_F32) {
        return false;
    }
    return c->mv_max_cols >= 1 && b->ne[1] == 1 && b->ne[2] == 1 && b->ne[3] == 1 && a->ne[2] == 1 && a->ne[3] == 1 &&
           node->type == GGML_TYPE_F32 && node->ne[2] == 1 && node->ne[3] == 1 && c->cfg.wg_mvr[cl11_rk_of(cl11_dtype_of(a->type))] % 32 == 0;
}

// can `other` be computed in the same launch as `head`? (both are candidates)
static bool cl11_mvm_same_group(const ggml_tensor * head, const ggml_tensor * other) {
    return other->src[1] == head->src[1] && other->src[0]->type == head->src[0]->type && other->src[0]->ne[0] == head->src[0]->ne[0] &&
           other != head && other->src[0] != head->src[0];
}

// can the CONT node i be skipped because the (only) MUL_MAT that uses it reads the transposed view directly?
static bool cl11_can_skip_cont(const cl11_device * c, const ggml_cgraph * cgraph, int i) {
    if (c->no_mvt || c->cfg.wg_row % 32 != 0) {
        return false;
    }
    const ggml_tensor * cont = cgraph->nodes[i];
    if (cont->op != GGML_OP_CONT || !ggml_node_has_n_uses(cgraph, i, 1)) {
        return false;
    }
    const ggml_tensor * a = cont->src[0];
    if (a == nullptr || a->type != GGML_TYPE_F16 || cont->type != GGML_TYPE_F16 || !ggml_are_same_shape(a, cont) ||
        a->nb[1] != ggml_type_size(GGML_TYPE_F16) || !sgt::is_transposed(cl11_desc(a)) || !cl11_fits_int(a)) {
        return false;
    }
    // the consumer
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * mm = cgraph->nodes[j];
        if (mm->src[1] == cont) {
            return false; // used as src1
        }
        if (mm->src[0] != cont) {
            continue;
        }
        const ggml_tensor * src1 = mm->src[1];
        return mm->op == GGML_OP_MUL_MAT && (mm->flags & GGML_TENSOR_FLAG_COMPUTE) != 0 &&
               src1->type == GGML_TYPE_F32 && src1->ne[1] == 1 && src1->nb[0] == sizeof(float) &&
               mm->type == GGML_TYPE_F32 && mm->nb[0] == sizeof(float) &&
               a->ne[0] == src1->ne[0] && src1->ne[2] % a->ne[2] == 0 &&
               (c->cfg.mvt_groups ? src1->ne[3] % a->ne[3] == 0
                                  : src1->ne[3] == a->ne[3] &&
                                    cl11_attn_ok(c, cl11_tref_of(cl11_view(a)).td, cl11_tref_of(cl11_view(src1)).td,
                                                 a->ne[1], a->ne[2], src1->ne[2] / a->ne[2], a->ne[0], true));
    }
    return false;
}

// the dispatch: decode op_params / build the views / call the backend op (src/backends/opencl11/ops*.cpp)
static bool cl11_compute_node(cl11_device * c, const ggml_tensor * node) {
    switch (node->op) {
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return cl11_op_binary(c, cl11_view(node), cl11_view(node->src[0]), cl11_view(node->src[1]), cl11_binary_kind_of(node->op));
        case GGML_OP_UNARY:
        case GGML_OP_SCALE:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT: {
            const int32_t code = cl11_unary_code(node);
            if (code < 0) {
                return false;
            }
            float p0 = 0.0f, p1 = 0.0f;
            if (node->op == GGML_OP_SCALE || node->op == GGML_OP_CLAMP) {
                memcpy(&p0, (const int32_t *) node->op_params + 0, sizeof(float));
                memcpy(&p1, (const int32_t *) node->op_params + 1, sizeof(float));
            }
            return cl11_op_unary(c, cl11_view(node), cl11_view(node->src[0]), code, p0, p1);
        }
        case GGML_OP_GLU: {
            const scrapgpu_glu_params p = { sizeof(p), cl11_glu_kind(node), node->src[1] != nullptr, ggml_get_op_params_i32(node, 1) };
            if (p.kind == (scrapgpu_glu_kind) -1) {
                return false;
            }
            const cl11_tensor s1 = node->src[1] != nullptr ? cl11_view(node->src[1]) : cl11_tensor();
            return cl11_op_glu(c, cl11_view(node), cl11_view(node->src[0]), node->src[1] != nullptr ? &s1 : nullptr, p);
        }
        case GGML_OP_RMS_NORM: {
            const scrapgpu_rms_norm_params p = { sizeof(p), cl11_eps(node) };
            return cl11_op_rms_norm(c, cl11_view(node), cl11_view(node->src[0]), p);
        }
        case GGML_OP_SOFT_MAX: {
            float scale = 1.0f;
            memcpy(&scale, (const int32_t *) node->op_params + 0, sizeof(float));
            const cl11_tensor m = node->src[1] != nullptr ? cl11_view(node->src[1]) : cl11_tensor();
            return cl11_op_soft_max(c, cl11_view(node), cl11_view(node->src[0]), node->src[1] != nullptr ? &m : nullptr, scale);
        }
        case GGML_OP_ROPE: {
            cl11_rope_params p;
            cl11_rope_decode(node, p);
            ggml_rope_yarn_corr_dims(p.n_dims, p.n_ctx_orig, p.freq_base, p.beta_fast, p.beta_slow, p.corr_dims);
            const cl11_tensor f = node->src[2] != nullptr ? cl11_view(node->src[2]) : cl11_tensor();
            return cl11_op_rope(c, cl11_view(node), cl11_view(node->src[0]), cl11_view(node->src[1]), node->src[2] != nullptr ? &f : nullptr, p);
        }
        case GGML_OP_GET_ROWS:
            return cl11_op_get_rows(c, cl11_view(node), cl11_view(node->src[0]), cl11_view(node->src[1]));
        case GGML_OP_SET_ROWS:
            return cl11_op_set_rows(c, cl11_view(node), cl11_view(node->src[0]), cl11_view(node->src[1]));
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return cl11_op_cpy(c, cl11_view(node), cl11_view(node->src[0]));
        case GGML_OP_MUL_MAT: {
            cl11_tensor s0 = cl11_view(node->src[0]);
            s0.repacked = cl11_repacked(node->src[0]);
            return cl11_op_mul_mat(c, cl11_view(node), s0, cl11_view(node->src[1]));
        }
        case GGML_OP_MUL_MAT_ID: {
            cl11_tensor s0 = cl11_view(node->src[0]);
            s0.repacked = cl11_repacked(node->src[0]);
            return cl11_op_mul_mat_id(c, cl11_view(node), s0, cl11_view(node->src[1]), cl11_view(node->src[2]));
        }
        case GGML_OP_ADD_ID:
            return cl11_op_add_id(c, cl11_view(node), cl11_view(node->src[0]), cl11_view(node->src[1]), cl11_view(node->src[2]));
        case GGML_OP_SSM_CONV:
            return cl11_op_ssm_conv(c, cl11_view(node), cl11_view(node->src[0]), cl11_view(node->src[1]));
        case GGML_OP_CONCAT: {
            const scrapgpu_concat_params p = { sizeof(p), (uint32_t) ggml_get_op_params_i32(node, 0) };
            return cl11_op_concat(c, cl11_view(node), cl11_view(node->src[0]), cl11_view(node->src[1]), p);
        }
        case GGML_OP_PAD: {
            scrapgpu_pad_params p = { sizeof(p), { 0, 0, 0, 0 }, { 0, 0, 0, 0 }, 0 };
            p.circular = ggml_get_op_params_i32(node, 8);
            for (int i = 0; i < 4; ++i) {
                p.before[i] = ggml_get_op_params_i32(node, 2 * i);
                p.after[i]  = ggml_get_op_params_i32(node, 2 * i + 1);
            }
            return cl11_op_pad(c, cl11_view(node), cl11_view(node->src[0]), p);
        }
        case GGML_OP_SUM_ROWS:
            return cl11_op_sum_rows(c, cl11_view(node), cl11_view(node->src[0]));
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K: {
            const scrapgpu_argsort_params p = { sizeof(p),
                ggml_get_op_params_i32(node, 0) == GGML_SORT_ORDER_DESC ? SCRAPGPU_ARGSORT_DESC : SCRAPGPU_ARGSORT_ASC, 0 };
            return cl11_op_argsort(c, cl11_view(node), cl11_view(node->src[0]), p, node->op == GGML_OP_TOP_K);
        }
        default:
            GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            return false;
    }
}

//
// backend
//

struct cl11_backend_context {
    cl11_device * dev;
    std::string   name;
};

static const char * ggml_backend_cl11_get_name(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    return bctx->name.c_str();
}

static void ggml_backend_cl11_free(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    if (bctx->dev->exec.profile) {
        std::lock_guard<std::mutex> lock(bctx->dev->mutex);
        if (bctx->dev->queue) {
            clFinish(bctx->dev->queue);
            cl11_prof_collect(bctx->dev->exec);
        }
        cl11_prof_print(bctx->dev->exec, bctx->dev->name.c_str());
    }
    delete bctx;
    delete backend;
}

static void ggml_backend_cl11_synchronize(ggml_backend_t backend) {
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device * c = bctx->dev;
    std::lock_guard<std::mutex> lock(c->mutex);
    if (c->queue) {
        const cl_int err = clFinish(c->queue);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("%s: clFinish failed: %s\n", __func__, cl11_err_str(err));
        }
    }
}

static enum ggml_status ggml_backend_cl11_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device * c = bctx->dev;

    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_program(c)) {
        return GGML_STATUS_FAILED;
    }

    if (getenv("GGML_CL11_TRACE_GRAPH")) {
        static int traced = 0;
        if (traced++ == 1) { // the first decode graph
            for (int i = 0; i < cgraph->n_nodes && i < 80; ++i) {
                const ggml_tensor * n = cgraph->nodes[i];
                fprintf(stderr, "node %3d %-10s %-28s [%lld,%lld,%lld,%lld]%s src0=%s src1=%s\n", i, ggml_op_name(n->op), n->name, (long long) n->ne[0], (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3],
                        (n->flags & GGML_TENSOR_FLAG_COMPUTE) ? "" : " (skip)", n->src[0] ? n->src[0]->name : "-", n->src[1] ? n->src[1]->name : "-");
            }
        }
    }
    for (auto & x : c->xprep) {
        x.valid = false;
    }
    c->mid_ids.valid = false;
    std::vector<const ggml_tensor *> skipped_conts;
    const size_t launches_before = c->exec.n_launches;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        if (ggml_is_empty(node) || ggml_op_is_empty(node->op)) {
            continue;
        }
        cl11_xprep_note_write(c, node);
        if (node->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes) {
            cl11_xprep_note_write(c, cgraph->nodes[i + 1]);
            cl11_xprep_note_write(c, cgraph->nodes[i + 2]);
        } else if (node->op == GGML_OP_RMS_NORM && i + 1 < cgraph->n_nodes) {
            cl11_xprep_note_write(c, cgraph->nodes[i + 1]);
        }

        // CONT of a transposed view that only feeds a mat-vec: skipped, the MUL_MAT reads the view (k_mvt_f16)
        if (node->op == GGML_OP_CONT && cl11_can_skip_cont(c, cgraph, i)) {
            skipped_conts.push_back(node);
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT && std::find(skipped_conts.begin(), skipped_conts.end(), node->src[0]) != skipped_conts.end()) {
            if (c->exec.profile) {
                c->exec.prof_key = cl11_prof_key(c->exec, c->cfg.mvt_groups ? "MUL_MAT f16 transposed (CONT skipped, many groups)" : "MUL_MAT f16 transposed (CONT skipped)");
                c->exec.prof_stats[c->exec.prof_key].count++;
            }
            if (!cl11_op_mul_mat_t(c, cl11_view(node), cl11_view(node->src[0]->src[0]), cl11_view(node->src[1]))) {
                GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
                return GGML_STATUS_FAILED;
            }
            continue;
        }

        // MUL_MATs of the same src1 and the same type of weights that follow each other (the nodes are made adjacent by graph_optimize): one launch
        if (node->op == GGML_OP_MUL_MAT && cl11_mvm_candidate(c, node)) {
            ggml_tensor * group[3] = { node, nullptr, nullptr };
            int n_group = 1;
            while (n_group < 3 && i + n_group < cgraph->n_nodes) {
                ggml_tensor * next = cgraph->nodes[i + n_group];
                if (next->op != GGML_OP_MUL_MAT || !cl11_mvm_candidate(c, next) || !cl11_mvm_same_group(node, next)) {
                    break;
                }
                group[n_group++] = next;
            }
            if (n_group >= 2) {
                if (c->exec.profile) {
                    std::string key = std::string("MUL_MAT ") + ggml_type_name(node->src[0]->type) + " mv multi x" + std::to_string(n_group) + " (fused)";
                    c->exec.prof_key = cl11_prof_key(c->exec, key);
                    c->exec.prof_stats[c->exec.prof_key].count++;
                    for (int j = 0; j < n_group; ++j) {
                        c->exec.prof_stats[c->exec.prof_key].bytes += (double) ggml_nbytes(group[j]->src[0]);
                    }
                }
                GGML_ASSERT(n_group >= 2 && n_group <= 3);
                cl11_tensor md[3], ms[3];
                for (int j = 0; j < n_group; ++j) {
                    md[j] = cl11_view(group[j]);
                    ms[j] = cl11_view(group[j]->src[0]);
                    ms[j].repacked = cl11_repacked(group[j]->src[0]);
                }
                const int r = cl11_op_mul_mat_multi(c, md, ms, cl11_view(node->src[1]), n_group);
                if (r < 0) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s) with %d fused mat-vec\n", __func__, i, ggml_op_desc(node), node->name, n_group);
                    return GGML_STATUS_FAILED;
                }
                if (r > 0) {
                    for (int j = 1; j < n_group; ++j) {
                        cl11_xprep_note_write(c, group[j]);
                    }
                    i += n_group - 1;
                    continue;
                }
                if (c->exec.profile) {
                    c->exec.prof_stats[c->exec.prof_key].count--; // not fused after all
                }
            }
        }

        // the SET_ROWS of the K cache and the one of the V cache (only views in between): one launch
        if (node->op == GGML_OP_SET_ROWS && c->fuse_set_rows) {
            int j = i + 1;
            while (j < cgraph->n_nodes && ((cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(cgraph->nodes[j]) || ggml_op_is_empty(cgraph->nodes[j]->op))) {
                ++j;
            }
            if (j < cgraph->n_nodes && cl11_can_pair_set_rows(c, node, cgraph->nodes[j])) {
                if (c->exec.profile) {
                    c->exec.prof_key = cl11_prof_key(c->exec, std::string("SET_ROWS ") + ggml_type_name(node->src[0]->type) + "->" + ggml_type_name(node->type) + " x2 (fused)");
                    c->exec.prof_stats[c->exec.prof_key].count++;
                }
                const ggml_tensor * nb = cgraph->nodes[j];
                const cl11_tensor pa[3] = { cl11_view(node->src[0]), cl11_view(node->src[1]), cl11_view(node) };
                const cl11_tensor pb[3] = { cl11_view(nb->src[0]), cl11_view(nb->src[1]), cl11_view(nb) };
                if (!cl11_op_set_rows2(c, pa, pb)) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s) with the next SET_ROWS\n", __func__, i, ggml_op_desc(node), node->name);
                    return GGML_STATUS_FAILED;
                }
                cl11_xprep_note_write(c, cgraph->nodes[j]);
                i = j;
                continue;
            }
        }

        // ADD + RMS_NORM + MUL (norm weights) in one kernel
        if (node->op == GGML_OP_ADD) {
            const ggml_tensor * w = nullptr;
            if (cl11_can_fuse_add_norm_mul(c, cgraph, i, &w)) {
                if (c->exec.profile) {
                    c->exec.prof_key = cl11_prof_key(c->exec, "ADD+RMS_NORM+MUL (fused)");
                    c->exec.prof_stats[c->exec.prof_key].count++;
                }
                const scrapgpu_rms_norm_params p = { sizeof(p), cl11_eps(cgraph->nodes[i + 1]) };
                if (!cl11_op_add_rms_norm_mul(c, cl11_view(cgraph->nodes[i + 2]), cl11_view(node->src[0]), cl11_view(node->src[1]),
                                              cl11_view(w), cl11_view(node), p)) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
                    return GGML_STATUS_FAILED;
                }
                i += 2;
                continue;
            }
        }

        // RMS_NORM + MUL (norm weights) in one kernel
        if (node->op == GGML_OP_RMS_NORM) {
            if (const ggml_tensor * w = cl11_norm_mul_weights(c, cgraph, i)) {
                if (c->exec.profile) {
                    c->exec.prof_key = cl11_prof_key(c->exec, "RMS_NORM+MUL (fused)");
                    c->exec.prof_stats[c->exec.prof_key].count++;
                }
                const scrapgpu_rms_norm_params p = { sizeof(p), cl11_eps(node) };
                if (!cl11_op_rms_norm_mul(c, cl11_view(cgraph->nodes[i + 1]), cl11_view(node->src[0]), cl11_view(w), p)) {
                    GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
                    return GGML_STATUS_FAILED;
                }
                ++i;
                continue;
            }
        }

        if (c->exec.profile) {
            std::string key = ggml_op_desc(node);
            if (node->op == GGML_OP_MUL_MAT) {
                key += std::string(" ") + ggml_type_name(node->src[0]->type) + (node->src[1]->ne[1] <= c->mv_max_cols ? " mv" : " mm");
                key += " n=" + std::to_string(node->src[1]->ne[1] <= c->mv_max_cols ? 1 : node->src[1]->ne[1]);
            } else if (node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_CPY || node->op == GGML_OP_SET_ROWS) {
                key += std::string(" ") + ggml_type_name(node->src[0]->type) + "->" + ggml_type_name(node->type);
            }
            if (getenv("GGML_CL11_PROFILE_SHAPES")) {
                const ggml_tensor * s0 = node->src[0];
                char buf[160];
                snprintf(buf, sizeof(buf), " src0[%lld,%lld,%lld,%lld] nb[%zu,%zu,%zu,%zu]", s0 ? (long long) s0->ne[0] : 0, s0 ? (long long) s0->ne[1] : 0,
                        s0 ? (long long) s0->ne[2] : 0, s0 ? (long long) s0->ne[3] : 0, s0 ? s0->nb[0] : 0, s0 ? s0->nb[1] : 0, s0 ? s0->nb[2] : 0, s0 ? s0->nb[3] : 0);
                key += buf;
            }
            c->exec.prof_key = cl11_prof_key(c->exec, key);
            c->exec.prof_stats[c->exec.prof_key].count++;
            if (node->op == GGML_OP_MUL_MAT) {
                c->exec.prof_stats[c->exec.prof_key].bytes += (double) ggml_nbytes(node->src[0]);
            }
        }

        if (!cl11_compute_node(c, node)) {
            GGML_LOG_ERROR("%s: failed to compute node %d (%s, %s)\n", __func__, i, ggml_op_desc(node), node->name);
            return GGML_STATUS_FAILED;
        }
    }

    if (c->exec.launch_log) {
        // the number of kernels of a graph does not depend on the GPU: the same count can be compared on any device
        int n_mul_mat = 0;
        int64_t n_cols = 0;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            if (cgraph->nodes[i]->op == GGML_OP_MUL_MAT && (cgraph->nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                ++n_mul_mat;
                n_cols = std::max<int64_t>(n_cols, cgraph->nodes[i]->src[1]->ne[1]);
            }
        }
        fprintf(stderr, "cl11 launches: nodes %d, mul_mat %d (max columns %lld), kernels %zu\n", cgraph->n_nodes, n_mul_mat, (long long) n_cols,
                c->exec.n_launches - launches_before);
    }

    // submit the commands, but do not wait for them
    clFlush(c->queue);
    if (c->exec.profile) {
        clFinish(c->queue);
        cl11_prof_collect(c->exec);
    }
    return GGML_STATUS_SUCCESS;
}

// Makes the MUL_MATs of the same src1 and the same type of weights adjacent (Q / K / V of an attention layer: the nodes of the RoPE / norm of Q lie
// between them) so that graph_compute can compute them in one launch. This runs before the graph is allocated (ggml_backend_sched): the buffers of
// the moved nodes are chosen for the new order, and a MUL_MAT whose inputs are a weight tensor and a tensor that the first node of the group reads too
// can be computed at any position after that tensor.
static void ggml_backend_cl11_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_UNUSED(params);
    auto * bctx = (cl11_backend_context *) backend->context;
    cl11_device * c = bctx->dev;

    std::lock_guard<std::mutex> lock(c->mutex);
    if (!cl11_ensure_program(c) || !c->multi_mv || c->no_fuse) {
        return;
    }
    const int n = cgraph->n_nodes;
    const int window = 48; // how far behind the first node the others are looked for
    int moved = 0;
    for (int i = 0; i < n; ++i) {
        ggml_tensor * head = cgraph->nodes[i];
        if (!cl11_mvm_candidate(c, head)) {
            continue;
        }
        int last = i;
        int members = 1;
        for (int j = i + 1; j < n && j < i + window && members < 3; ++j) {
            ggml_tensor * cand = cgraph->nodes[j];
            if (!cl11_mvm_candidate(c, cand) || !cl11_mvm_same_group(head, cand)) {
                continue;
            }
            if (j != last + 1) {
                for (int k = j; k > last + 1; --k) {
                    cgraph->nodes[k] = cgraph->nodes[k - 1];
                }
                cgraph->nodes[last + 1] = cand;
                ++moved;
            }
            ++last;
            ++members;
        }
        i = last;
    }
    if (moved > 0 && cl11_debug()) {
        GGML_LOG_INFO("%s: moved %d mat-vec nodes next to the nodes that read the same src1\n", __func__, moved);
    }
}

static const ggml_backend_i ggml_backend_cl11_i = {
    /* .get_name                = */ ggml_backend_cl11_get_name,
    /* .free                    = */ ggml_backend_cl11_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ ggml_backend_cl11_synchronize,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_cl11_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ ggml_backend_cl11_graph_optimize,
};

static ggml_guid_t ggml_backend_cl11_guid() {
    static ggml_guid guid = { 0x43, 0x4c, 0x31, 0x31, 0x6f, 0x70, 0x65, 0x6e, 0x63, 0x6c, 0x31, 0x2e, 0x31, 0x67, 0x67, 0x6d };
    return &guid;
}

//
// device
//

static cl11_device_context * cl11_dev_ctx(ggml_backend_dev_t dev) {
    return (cl11_device_context *) dev->context;
}

static const char * ggml_backend_cl11_device_get_name(ggml_backend_dev_t dev) {
    return cl11_dev_ctx(dev)->device->name.c_str();
}

static const char * ggml_backend_cl11_device_get_description(ggml_backend_dev_t dev) {
    return cl11_dev_ctx(dev)->device->info.name.c_str();
}

static void ggml_backend_cl11_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    const cl11_device * c = cl11_dev_ctx(dev)->device.get();
    // OpenCL 1.1 cannot report the free memory: assume that the system (display, ...) keeps some of it
    *total = c->info.global_mem;
    const size_t reserve = std::min<size_t>(128u * 1024 * 1024, *total / 4);
    *free = *total - reserve;
}

static enum ggml_backend_dev_type ggml_backend_cl11_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cl11_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_cl11_device_get_name(dev);
    props->description = ggml_backend_cl11_device_get_description(dev);
    props->type        = ggml_backend_cl11_device_get_type(dev);
    props->device_id   = nullptr;
    ggml_backend_cl11_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_cl11_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    cl11_device * c = cl11_dev_ctx(dev)->device.get();

    {
        std::lock_guard<std::mutex> lock(c->mutex);
        if (!cl11_ensure_context(c)) {
            return nullptr;
        }
    }

    auto * bctx = new cl11_backend_context{ c, c->name };
    return new ggml_backend {
        /* .guid    = */ ggml_backend_cl11_guid(),
        /* .iface   = */ ggml_backend_cl11_i,
        /* .device  = */ dev,
        /* .context = */ bctx,
    };
}

static ggml_backend_buffer_type_t ggml_backend_cl11_device_get_buffer_type(ggml_backend_dev_t dev) {
    return &cl11_dev_ctx(dev)->buft;
}

static bool ggml_backend_cl11_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const bool ok = cl11_supports_op_impl(cl11_dev_ctx(dev)->device.get(), op);
    // GGML_CL11_LOG_UNSUPPORTED=1: print every operation (once per name / shape) that is left to another backend
    if (!ok && op->op != GGML_OP_NONE && getenv("GGML_CL11_LOG_UNSUPPORTED") != nullptr) {
        static std::mutex mtx;
        static std::vector<std::string> seen;
        char buf[256];
        snprintf(buf, sizeof(buf), "%s %s [%lld,%lld,%lld,%lld] src0 %s", ggml_op_desc(op), ggml_type_name(op->type), (long long) op->ne[0], (long long) op->ne[1], (long long) op->ne[2],
                (long long) op->ne[3], op->src[0] ? ggml_type_name(op->src[0]->type) : "-");
        std::lock_guard<std::mutex> lock(mtx);
        if (std::find(seen.begin(), seen.end(), buf) == seen.end()) {
            seen.push_back(buf);
            fprintf(stderr, "cl11: not supported: %s (%s)\n", buf, op->name);
        }
    }
    return ok;
}

static bool ggml_backend_cl11_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // only the buffers of the same device: a cl_mem cannot be used with another cl_context
    return buft->iface.get_name == ggml_backend_cl11_buft_get_name && buft->context == dev->context;
}

static const ggml_backend_device_i ggml_backend_cl11_device_i = {
    /* .get_name             = */ ggml_backend_cl11_device_get_name,
    /* .get_description      = */ ggml_backend_cl11_device_get_description,
    /* .get_memory           = */ ggml_backend_cl11_device_get_memory,
    /* .get_type             = */ ggml_backend_cl11_device_get_type,
    /* .get_props            = */ ggml_backend_cl11_device_get_props,
    /* .init_backend         = */ ggml_backend_cl11_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_cl11_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_cl11_device_supports_op,
    /* .supports_buft        = */ ggml_backend_cl11_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

//
// registry
//

static const char * ggml_backend_cl11_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CL11_NAME;
}

static size_t ggml_backend_cl11_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return g_cl11_devices.size();
}

static ggml_backend_dev_t ggml_backend_cl11_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index < g_cl11_devices.size());
    GGML_UNUSED(reg);
    return &g_cl11_devices[index]->dev;
}

static const ggml_backend_reg_i ggml_backend_cl11_reg_i = {
    /* .get_name         = */ ggml_backend_cl11_reg_get_name,
    /* .get_device_count = */ ggml_backend_cl11_reg_get_device_count,
    /* .get_device       = */ ggml_backend_cl11_reg_get_device,
    /* .get_proc_address = */ nullptr,
};

ggml_backend_reg_t ggml_backend_cl11_reg(void) {
    static std::mutex mutex;
    static ggml_backend_reg reg;
    static bool initialized = false;

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized) {
        return &reg;
    }
    initialized = true;

    reg = ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_cl11_reg_i,
        /* .context     = */ nullptr,
    };

    cl11_probe_devices();

    for (auto & c : g_cl11_devices) {
        c->buft = ggml_backend_buffer_type {
            /* .iface   = */ ggml_backend_cl11_buft_i,
            /* .device  = */ &c->dev,
            /* .context = */ c.get(),
        };
        c->dev = ggml_backend_device {
            /* .iface   = */ ggml_backend_cl11_device_i,
            /* .reg     = */ &reg,
            /* .context = */ c.get(),
        };
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cl11_reg)

//
// public API
//

int ggml_backend_cl11_get_device_count(void) {
    return (int) ggml_backend_reg_dev_count(ggml_backend_cl11_reg());
}

void ggml_backend_cl11_get_device_description(int device, char * description, size_t description_size) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg) || description_size == 0) {
        if (description_size > 0) {
            description[0] = '\0';
        }
        return;
    }
    snprintf(description, description_size, "%s", ggml_backend_dev_description(ggml_backend_reg_dev_get(reg, (size_t) device)));
}

ggml_backend_t ggml_backend_cl11_init(int device) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, (size_t) device), nullptr);
}

bool ggml_backend_is_cl11(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_cl11_guid());
}

ggml_backend_buffer_type_t ggml_backend_cl11_buffer_type(int device) {
    ggml_backend_reg_t reg = ggml_backend_cl11_reg();
    if (device < 0 || (size_t) device >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, (size_t) device));
}
