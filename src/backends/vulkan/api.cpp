// The ScrapGpu C API on top of the Vulkan backend: scrapgpu_vulkan_register() registers a backend
// whose devices are the Vulkan 1.2 physical devices that pass the SCRAPGPU_VULKAN_DEVICES filter
// (NVIDIA skipped unless SCRAPGPU_VULKAN_ALLOW_NVIDIA=1, and devices without a compute queue or
// 16-bit storage). Buffers are device-local memory; host transfers go through a per-stream
// host-visible staging buffer. A stream is one command buffer submitted with a fence per call;
// stream_sync waits. Ops are the first slice: ADD / MUL (F32, broadcast), RMS_NORM (F32) and
// MUL_MAT (src0 F32 / F16 / Q4_0 / Q8_0 in the natural GGUF layout, src1 and dst F32; a workgroup
// per row for a single-column src1, a simple 8x8-tile mat-mat otherwise). R32 is not a Vulkan
// layout: quantized weights stay in the natural layout on this backend.

#include "vk.h"

#include "vk-shaders.h"

#include "scrapgpu/backend.h"
#include "scrapgpu/vulkan.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------------------------
// objects
// ---------------------------------------------------------------------------------------------

struct vk_api_device {
    scrapgpu_object_header hdr; // must stay the first member
    sgv::vk_dev            dev;
    // device_get_info scratch: the info strings point into these members (stable until the next call)
    std::string                 generation_reason;
    scrapgpu_device_generation  generation = SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
    bool                        info_done = false;
    std::string                 vendor_name;
    std::string                 api_version;
    std::string                 driver_version;
    std::string                 tuning_overrides;
    std::string                 tuning_profile;
};

struct vk_api_buffer {
    scrapgpu_object_header hdr;
    vk_api_device *        dev = nullptr;
    sgv::vk_buf            buf;
};

struct vk_api_stream {
    scrapgpu_object_header hdr;
    vk_api_device *        dev = nullptr;
    sgv::vk_str            str;
};

struct vk_api_plan_node {
    scrapgpu_op_desc                   desc;
    std::vector<scrapgpu_tensor_desc>  inputs;
    std::vector<scrapgpu_tensor_desc>  outputs;
    std::vector<uint8_t>               params;
};

struct vk_api_plan {
    scrapgpu_object_header           hdr;
    vk_api_device *                  dev = nullptr;
    std::vector<vk_api_plan_node>    nodes;
};

extern const scrapgpu_backend_vtable g_vk_vtable; // defined under "registration" below

std::vector<std::unique_ptr<vk_api_device>> g_devices;

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

scrapgpu_status api_unsupported(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

scrapgpu_status api_fail(const char * what) {
    scrapgpu_set_last_error(what);
    return SCRAPGPU_STATUS_BACKEND_ERROR;
}

scrapgpu_status api_vk_error(const char * what, VkResult r) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s (VkResult %d)", what, (int) r);
    scrapgpu_set_last_error(buf);
    if (r == VK_ERROR_DEVICE_LOST) {
        return SCRAPGPU_STATUS_DEVICE_LOST;
    }
    if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) {
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    return SCRAPGPU_STATUS_BACKEND_ERROR;
}

vk_api_device * api_device(scrapgpu_device_t * device) {
    auto * d = reinterpret_cast<vk_api_device *>(device);
    if (d == nullptr || d->hdr.backend != &g_vk_vtable) {
        return nullptr;
    }
    return d;
}

vk_api_buffer * api_buffer(scrapgpu_buffer_t * buffer) {
    auto * b = reinterpret_cast<vk_api_buffer *>(buffer);
    if (b == nullptr || b->hdr.backend != &g_vk_vtable) {
        return nullptr;
    }
    return b;
}

vk_api_stream * api_stream(scrapgpu_stream_t * stream) {
    auto * s = reinterpret_cast<vk_api_stream *>(stream);
    if (s == nullptr || s->hdr.backend != &g_vk_vtable) {
        return nullptr;
    }
    return s;
}

// the caller-visible fields of a tensor desc (a smaller struct_size keeps the defaults of the tail:
// buffer = NULL, offset = 0, layout = NATURAL)
scrapgpu_tensor_desc api_load_desc(const scrapgpu_tensor_desc & in) {
    scrapgpu_tensor_desc out = {};
    if (in.struct_size >= sizeof(size_t)) {
        std::memcpy(&out, &in, std::min(in.struct_size, sizeof(out)));
        out.struct_size = sizeof(out);
    }
    return out;
}

sgt::TensorDesc api_sgt_desc(const scrapgpu_tensor_desc & in) {
    return sgt::desc_of(api_load_desc(in));
}

template <typename T>
T api_params(const scrapgpu_op_desc * desc) {
    T p = {};
    if (desc->params != nullptr) {
        std::memcpy(&p, desc->params, std::min(desc->params_size, sizeof(T)));
    }
    return p;
}

// the device (lazily) as a VkDevice: every entry point that touches the GPU calls this first
bool api_ensure(vk_api_device * d, std::string & why) {
    if (d->dev.dev != VK_NULL_HANDLE) {
        return true;
    }
    return sgv::vk_device_init(&d->dev, why);
}

// ---------------------------------------------------------------------------------------------
// push-constant blocks (must match the layout(push_constant) structs of the shaders)
// ---------------------------------------------------------------------------------------------

struct pc_binary { // binary.comp
    uint32_t ne0[4];
    uint32_t a_off, a_nb[4];
    uint32_t ne1[4];
    uint32_t b_off, b_nb[4];
    uint32_t d_off, d_nb[4];
};

struct pc_rms { // rms_norm.comp
    uint32_t ne00, ne01, ne02;
    float eps;
    uint32_t a_off, a_nb[4];
    uint32_t d_off, d_nb[4];
};

struct pc_mv { // mul_mat_vec.comp
    uint32_t ncols;
    uint32_t a_off, a_nb1, a_nb2, a_nb3;
    uint32_t b_off, b_nb2, b_nb3;
    uint32_t d_off, d_nb2, d_nb3;
    uint32_t r2, r3, ne12;
};

struct pc_mm { // mul_mat.comp
    uint32_t ncols, m_rows, n_cols, tiles_x;
    uint32_t a_off, a_nb1, a_nb2, a_nb3;
    uint32_t b_off, b_nb1, b_nb2, b_nb3;
    uint32_t d_off, d_nb1, d_nb2, d_nb3;
    uint32_t r2, r3, ne12;
};

// ---------------------------------------------------------------------------------------------
// op_supported
// ---------------------------------------------------------------------------------------------

bool api_shaders_built() {
    return SG_VK_SHADERS_BUILT != 0;
}

// the shader storage unit of a tensor: 4 bytes for the 32-bit types, 2 for F16 and the quant blocks
uint32_t api_units(scrapgpu_dtype dtype) {
    return dtype == SCRAPGPU_DTYPE_F32 || dtype == SCRAPGPU_DTYPE_I32 ? 4 : 2;
}

// offsets and used strides must be whole in the storage units the shader counts with
bool api_unit_ok(const sgt::TensorDesc & t) {
    const uint32_t u = api_units(t.dtype);
    if (t.offset % u != 0) {
        return false;
    }
    for (int i = 0; i < sgt::kMaxDims; ++i) {
        if (t.ne[i] > 1 && t.nb[i] % u != 0) {
            return false;
        }
    }
    return true;
}

scrapgpu_status api_check_tensors(vk_api_device * d, const scrapgpu_op_desc * desc, sgt::TensorDesc * in, sgt::TensorDesc * out, const char ** reason) {
    (void) d;
    auto check_one = [&](const scrapgpu_tensor_desc & raw, sgt::TensorDesc & td) -> scrapgpu_status {
        td = api_sgt_desc(raw);
        if (sgt::dtype_type_size(td.dtype) == 0) {
            *reason = "unknown tensor dtype";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        for (int i = 0; i < sgt::kMaxDims; ++i) {
            if (td.ne[i] < 0) {
                *reason = "negative tensor dim";
                return SCRAPGPU_STATUS_INVALID_ARGUMENT;
            }
        }
        if (!sgt::fits_int32(td)) {
            *reason = "tensor exceeds the 32-bit limits of the kernel descriptors";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        if (td.layout != SCRAPGPU_LAYOUT_NATURAL) {
            *reason = "the Vulkan backend only supports the natural layout (R32 is an OpenCL repacking)";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        if (!api_unit_ok(td)) {
            *reason = "tensor offset or strides are not aligned to the shader storage units";
            return SCRAPGPU_STATUS_UNSUPPORTED;
        }
        return SCRAPGPU_STATUS_OK;
    };
    for (size_t i = 0; i < desc->input_count; ++i) {
        const scrapgpu_status st = check_one(desc->inputs[i], in[i]);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        const scrapgpu_status st = check_one(desc->outputs[i], out[i]);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_no(const char * why, const char ** reason) {
    if (reason != nullptr) {
        *reason = why;
    }
    scrapgpu_set_last_error(why);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

scrapgpu_status api_supports(vk_api_device * d, const scrapgpu_op_desc * desc, const char ** reason) {
    const char * reason_local = nullptr;
    if (reason == nullptr) {
        reason = &reason_local;
    }
    *reason = nullptr;
    const scrapgpu_status valid = scrapgpu_op_desc_validate(desc);
    if (valid != SCRAPGPU_STATUS_OK) {
        if (reason != nullptr) {
            *reason = scrapgpu_last_error();
        }
        return valid;
    }

    switch (desc->op) {
        case SCRAPGPU_OP_NONE:
        case SCRAPGPU_OP_RESHAPE:
        case SCRAPGPU_OP_VIEW:
        case SCRAPGPU_OP_PERMUTE:
        case SCRAPGPU_OP_TRANSPOSE:
            return SCRAPGPU_STATUS_OK;
        default:
            break;
    }

    if (!api_shaders_built()) {
        return api_no("the Vulkan shaders were not built (glslc not found at build time)", reason);
    }

    sgt::TensorDesc in[8] = {}, out[3] = {};
    {
        const scrapgpu_status st = api_check_tensors(d, desc, in, out, reason);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }

    switch (desc->op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_MUL: {
            const sgt::TensorDesc & s0 = in[0];
            const sgt::TensorDesc & s1 = in[1];
            const sgt::TensorDesc & dst = out[0];
            bool ok = s0.dtype == SCRAPGPU_DTYPE_F32 && s1.dtype == SCRAPGPU_DTYPE_F32 && dst.dtype == SCRAPGPU_DTYPE_F32 &&
                      sgt::same_shape(s0, dst);
            for (int i = 0; i < sgt::kMaxDims && ok; ++i) {
                ok = s1.ne[i] > 0 && s1.ne[i] <= dst.ne[i];
            }
            return ok ? SCRAPGPU_STATUS_OK
                      : api_no("ADD/MUL need F32 tensors, dst with the shape of src0 and a broadcastable src1", reason);
        }

        case SCRAPGPU_OP_RMS_NORM: {
            const bool ok = in[0].dtype == SCRAPGPU_DTYPE_F32 && out[0].dtype == SCRAPGPU_DTYPE_F32 &&
                            sgt::same_shape(in[0], out[0]) &&
                            in[0].nb[0] == sizeof(float) && out[0].nb[0] == sizeof(float);
            return ok ? SCRAPGPU_STATUS_OK
                      : api_no("RMS_NORM needs F32 tensors of the same shape with contiguous rows", reason);
        }

        case SCRAPGPU_OP_MUL_MAT: {
            const sgt::TensorDesc & s0 = in[0];
            const sgt::TensorDesc & s1 = in[1];
            const sgt::TensorDesc & dst = out[0];
            const auto p = api_params<scrapgpu_mul_mat_params>(desc);
            const bool quant = sgt::dtype_is_quantized(s0.dtype);
            bool ok = s1.dtype == SCRAPGPU_DTYPE_F32 && dst.dtype == SCRAPGPU_DTYPE_F32 &&
                      (s0.dtype == SCRAPGPU_DTYPE_F32 || s0.dtype == SCRAPGPU_DTYPE_F16 ||
                       s0.dtype == SCRAPGPU_DTYPE_Q4_0 || s0.dtype == SCRAPGPU_DTYPE_Q8_0) &&
                      s0.nb[0] == sgt::dtype_type_size(s0.dtype) &&
                      s1.nb[0] == sizeof(float) && dst.nb[0] == sizeof(float) &&
                      s0.ne[0] > 0 && s0.ne[2] > 0 && s0.ne[3] > 0 &&
                      s0.ne[0] == s1.ne[0] &&
                      s1.ne[2] % s0.ne[2] == 0 && s1.ne[3] % s0.ne[3] == 0 &&
                      dst.ne[0] == s0.ne[1] && dst.ne[1] == s1.ne[1] &&
                      dst.ne[2] == s1.ne[2] && dst.ne[3] == s1.ne[3];
            if (!ok) {
                return api_no("MUL_MAT needs src0 F32/F16/Q4_0/Q8_0 (natural layout), an F32 src1 and dst, contiguous rows and dst = [ne01, ne11, ne12, ne13]", reason);
            }
            if (quant && s0.ne[0] % (int64_t) sgt::dtype_block_size(s0.dtype) != 0) {
                return api_no("MUL_MAT with a quantized src0 needs whole blocks (ne0 a multiple of 32)", reason);
            }
            const int64_t r2 = s1.ne[2] / s0.ne[2];
            const int64_t r3 = s1.ne[3] / s0.ne[3];
            if (p.allow_broadcast == 0 && (r2 != 1 || r3 != 1)) {
                return api_no("MUL_MAT: broadcasting over dims 2/3 is not allowed by the op params", reason);
            }
            const uint64_t batch = (uint64_t) s1.ne[2] * (uint64_t) s1.ne[3];
            if (batch > d->dev.phys.props.limits.maxComputeWorkGroupCount[2]) {
                return api_no("the batch dim exceeds the driver's workgroup count limit", reason);
            }
            return SCRAPGPU_STATUS_OK;
        }

        default:
            return api_no("op is not implemented by the Vulkan backend", reason);
    }
}

// ---------------------------------------------------------------------------------------------
// tensor resolution and dispatch
// ---------------------------------------------------------------------------------------------

struct vk_tref {
    vk_api_buffer *  b = nullptr;
    sgt::TensorDesc  d;
};

scrapgpu_status api_resolve(vk_api_stream * s, const scrapgpu_tensor_desc & raw, vk_tref & out) {
    const scrapgpu_tensor_desc td = api_load_desc(raw);
    vk_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr) {
        return api_invalid("tensor desc has no Vulkan buffer");
    }
    if (b->dev != s->dev) {
        return api_invalid("tensor buffer and stream belong to different devices");
    }
    out.d = sgt::desc_of(td);
    const uint64_t nb = sgt::nbytes(out.d);
    if (out.d.offset > b->buf.size || nb > b->buf.size - out.d.offset) {
        return api_invalid("tensor exceeds its buffer");
    }
    out.b = b;
    return SCRAPGPU_STATUS_OK;
}

// one dispatch into the stream's command buffer (the caller submits); the pipeline was created
scrapgpu_status api_record_dispatch(vk_api_stream * s, int shader, const void * pc, uint32_t pc_size,
                                    const vk_tref & a, const vk_tref & b, const vk_tref & dst,
                                    uint32_t gx, uint32_t gy, uint32_t gz) {
    vk_api_device * d = s->dev;
    if (sgv::vk_record(&d->dev, &s->str) != VK_SUCCESS) {
        return api_fail("vkBeginCommandBuffer failed");
    }
    if (!sgv::vk_dispatch(&d->dev, &s->str, shader, pc, pc_size, &a.b->buf, &b.b->buf, &dst.b->buf, gx, gy, gz)) {
        return api_fail("the dispatch could not be recorded");
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_submit(vk_api_stream * s) {
    const VkResult r = sgv::vk_submit(&s->dev->dev, &s->str);
    if (r != VK_SUCCESS) {
        return api_vk_error("vkQueueSubmit failed", r);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_dispatch_binary(vk_api_stream * s, const scrapgpu_op_desc * desc, const vk_tref * in, const vk_tref * out) {
    pc_binary pc = {};
    const sgt::TensorDesc & s0 = in[0].d, & s1 = in[1].d, & dst = out[0].d;
    for (int i = 0; i < 4; ++i) {
        pc.ne0[i] = (uint32_t) dst.ne[i];
        pc.a_nb[i] = (uint32_t) (s0.nb[i] / 4);
        pc.ne1[i] = (uint32_t) s1.ne[i];
        pc.b_nb[i] = (uint32_t) (s1.nb[i] / 4);
        pc.d_nb[i] = (uint32_t) (dst.nb[i] / 4);
    }
    pc.a_off = (uint32_t) (s0.offset / 4);
    pc.b_off = (uint32_t) (s1.offset / 4);
    pc.d_off = (uint32_t) (dst.offset / 4);
    const uint64_t total = (uint64_t) sgt::nelements(dst);
    if (total == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    const uint64_t groups = (total + 127) / 128;
    const uint32_t gx = (uint32_t) std::min<uint64_t>(groups, 65535);
    const uint32_t gy = (uint32_t) ((groups + gx - 1) / gx);
    return api_record_dispatch(s, desc->op == SCRAPGPU_OP_MUL ? sgv::VK_S_MUL : sgv::VK_S_ADD,
                               &pc, sizeof(pc), in[0], in[1], out[0], gx, gy, 1);
}

scrapgpu_status api_dispatch_rms(vk_api_stream * s, const scrapgpu_op_desc * desc, const vk_tref * in, const vk_tref * out) {
    const auto p = api_params<scrapgpu_rms_norm_params>(desc);
    pc_rms pc = {};
    const sgt::TensorDesc & a = in[0].d, & dst = out[0].d;
    pc.ne00 = (uint32_t) a.ne[0];
    pc.ne01 = (uint32_t) a.ne[1];
    pc.ne02 = (uint32_t) a.ne[2];
    pc.eps = p.eps;
    pc.a_off = (uint32_t) (a.offset / 4);
    pc.d_off = (uint32_t) (dst.offset / 4);
    for (int i = 0; i < 4; ++i) {
        pc.a_nb[i] = (uint32_t) (a.nb[i] / 4);
        pc.d_nb[i] = (uint32_t) (dst.nb[i] / 4);
    }
    const uint64_t rows = (uint64_t) (a.ne[1] * a.ne[2] * a.ne[3]);
    if (rows == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    const uint32_t gx = (uint32_t) std::min<uint64_t>(rows, 65535);
    const uint32_t gy = (uint32_t) ((rows + gx - 1) / gx);
    return api_record_dispatch(s, sgv::VK_S_RMS_NORM, &pc, sizeof(pc), in[0], in[0], out[0], gx, gy, 1);
}

int api_mv_shader(scrapgpu_dtype dt) {
    switch (dt) {
        case SCRAPGPU_DTYPE_F32:  return sgv::VK_S_MV_F32;
        case SCRAPGPU_DTYPE_F16:  return sgv::VK_S_MV_F16;
        case SCRAPGPU_DTYPE_Q4_0: return sgv::VK_S_MV_Q4_0;
        default:                  return sgv::VK_S_MV_Q8_0;
    }
}

int api_mm_shader(scrapgpu_dtype dt) {
    return api_mv_shader(dt) + (sgv::VK_S_MM_F32 - sgv::VK_S_MV_F32);
}

// dst[n] offsets and strides in shader units for src0 (u16 for F16/quant, floats for F32)
void api_fill_a(pc_mv & pc, const sgt::TensorDesc & a) {
    const uint32_t u = api_units(a.dtype);
    pc.a_off = (uint32_t) (a.offset / u);
    pc.a_nb1 = (uint32_t) (a.nb[1] / u);
    pc.a_nb2 = (uint32_t) (a.nb[2] / u);
    pc.a_nb3 = (uint32_t) (a.nb[3] / u);
}

scrapgpu_status api_dispatch_mul_mat(vk_api_stream * s, const vk_tref * in, const vk_tref * out) {
    const sgt::TensorDesc & s0 = in[0].d, & s1 = in[1].d, & dst = out[0].d;
    const int64_t r2 = s1.ne[2] / s0.ne[2];
    const int64_t r3 = s1.ne[3] / s0.ne[3];
    const uint64_t batch = (uint64_t) s1.ne[2] * (uint64_t) s1.ne[3];
    if (batch == 0 || s0.ne[1] == 0 || s1.ne[1] == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (s1.ne[1] == 1) {
        pc_mv pc = {};
        pc.ncols = (uint32_t) s0.ne[0];
        api_fill_a(pc, s0);
        pc.b_off = (uint32_t) (s1.offset / 4);
        pc.b_nb2 = (uint32_t) (s1.nb[2] / 4);
        pc.b_nb3 = (uint32_t) (s1.nb[3] / 4);
        pc.d_off = (uint32_t) (dst.offset / 4);
        pc.d_nb2 = (uint32_t) (dst.nb[2] / 4);
        pc.d_nb3 = (uint32_t) (dst.nb[3] / 4);
        pc.r2 = (uint32_t) r2;
        pc.r3 = (uint32_t) r3;
        pc.ne12 = (uint32_t) s1.ne[2];
        const uint64_t rows = (uint64_t) s0.ne[1];
        const uint32_t gx = (uint32_t) std::min<uint64_t>(rows, 65535);
        const uint32_t gy = (uint32_t) ((rows + gx - 1) / gx);
        return api_record_dispatch(s, api_mv_shader(s0.dtype), &pc, sizeof(pc), in[0], in[1], out[0], gx, gy, (uint32_t) batch);
    }
    pc_mm pc = {};
    pc.ncols = (uint32_t) s0.ne[0];
    pc.m_rows = (uint32_t) s0.ne[1];
    pc.n_cols = (uint32_t) s1.ne[1];
    pc.tiles_x = (uint32_t) ((s0.ne[1] + 7) / 8);
    {
        const uint32_t u = api_units(s0.dtype);
        pc.a_off = (uint32_t) (s0.offset / u);
        pc.a_nb1 = (uint32_t) (s0.nb[1] / u);
        pc.a_nb2 = (uint32_t) (s0.nb[2] / u);
        pc.a_nb3 = (uint32_t) (s0.nb[3] / u);
    }
    pc.b_off = (uint32_t) (s1.offset / 4);
    pc.b_nb1 = (uint32_t) (s1.nb[1] / 4);
    pc.b_nb2 = (uint32_t) (s1.nb[2] / 4);
    pc.b_nb3 = (uint32_t) (s1.nb[3] / 4);
    pc.d_off = (uint32_t) (dst.offset / 4);
    pc.d_nb1 = (uint32_t) (dst.nb[1] / 4);
    pc.d_nb2 = (uint32_t) (dst.nb[2] / 4);
    pc.d_nb3 = (uint32_t) (dst.nb[3] / 4);
    pc.r2 = (uint32_t) r2;
    pc.r3 = (uint32_t) r3;
    pc.ne12 = (uint32_t) s1.ne[2];
    const uint64_t tiles = (uint64_t) pc.tiles_x * (uint64_t) ((s1.ne[1] + 7) / 8);
    const uint32_t gx = (uint32_t) std::min<uint64_t>(tiles, 65535);
    const uint32_t gy = (uint32_t) ((tiles + gx - 1) / gx);
    return api_record_dispatch(s, api_mm_shader(s0.dtype), &pc, sizeof(pc), in[0], in[1], out[0], gx, gy, (uint32_t) batch);
}

// records one validated + supported op (buffers resolved by the caller)
scrapgpu_status api_dispatch(vk_api_stream * s, const scrapgpu_op_desc * desc, const vk_tref * in, vk_tref * out) {
    switch (desc->op) {
        case SCRAPGPU_OP_ADD:
        case SCRAPGPU_OP_MUL:
            return api_dispatch_binary(s, desc, in, out);
        case SCRAPGPU_OP_RMS_NORM:
            return api_dispatch_rms(s, desc, in, out);
        case SCRAPGPU_OP_MUL_MAT:
            return api_dispatch_mul_mat(s, in, out);
        default:
            return api_unsupported("op is not implemented by the Vulkan backend");
    }
}

scrapgpu_status api_exec_desc(vk_api_stream * s, const scrapgpu_op_desc * desc) {
    switch (desc->op) {
        case SCRAPGPU_OP_NONE:
        case SCRAPGPU_OP_RESHAPE:
        case SCRAPGPU_OP_VIEW:
        case SCRAPGPU_OP_PERMUTE:
        case SCRAPGPU_OP_TRANSPOSE:
            return SCRAPGPU_STATUS_OK; // metadata ops: nothing to compute
        default:
            break;
    }

    const char * why = nullptr;
    const scrapgpu_status st = api_supports(s->dev, desc, &why);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }

    vk_tref in[8] = {}, out[3] = {};
    for (size_t i = 0; i < desc->input_count; ++i) {
        const scrapgpu_status r = api_resolve(s, desc->inputs[i], in[i]);
        if (r != SCRAPGPU_STATUS_OK) {
            return r;
        }
    }
    for (size_t i = 0; i < desc->output_count; ++i) {
        const scrapgpu_status r = api_resolve(s, desc->outputs[i], out[i]);
        if (r != SCRAPGPU_STATUS_OK) {
            return r;
        }
    }

    return api_dispatch(s, desc, in, out);
}

// ---------------------------------------------------------------------------------------------
// devices
// ---------------------------------------------------------------------------------------------

size_t api_get_device_count(void * userdata) {
    (void) userdata;
    return g_devices.size();
}

scrapgpu_status api_get_device(void * userdata, size_t index, scrapgpu_device_t ** out_device) {
    (void) userdata;
    if (index >= g_devices.size()) {
        return api_invalid("device index out of range");
    }
    *out_device = reinterpret_cast<scrapgpu_device_t *>(g_devices[index].get());
    return SCRAPGPU_STATUS_OK;
}

std::string vk_version_string(uint32_t v) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%u.%u.%u", VK_VERSION_MAJOR(v), VK_VERSION_MINOR(v), VK_VERSION_PATCH(v));
    return buf;
}

scrapgpu_device_generation api_generation(const sgv::vk_phys & p, std::string & reason) {
    const std::string name = [&] {
        std::string l = p.name;
        std::transform(l.begin(), l.end(), l.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        return l;
    }();
    if (p.vendor_id == 0x10DE) {
        reason = "NVIDIA device (allowed by SCRAPGPU_VULKAN_ALLOW_NVIDIA): " + p.name;
        return SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
    }
    if (p.vendor_id == 0x1002) {
        // the RADV device name carries the chip's code name: polarisNN / vegaNN / tonga / fiji are
        // GCN, naviNN / gfx10NN are RDNA
        static const char * const gcn[] = {
            "polaris", "vega", "tonga", "fiji", "hawaii", "bonaire", "kaveri", "kabini", "mullins", "gfx6", "gfx7", "gfx8", "gfx9",
        };
        for (const char * c : gcn) {
            if (name.find(c) != std::string::npos) {
                reason = "AMD GCN (" + p.name + ")";
                return SCRAPGPU_DEVICE_GENERATION_AMD_GCN;
            }
        }
        if (name.find("navi") != std::string::npos || name.find("gfx10") != std::string::npos || name.find("gfx11") != std::string::npos || name.find("gfx12") != std::string::npos) {
            reason = "AMD RDNA (" + p.name + ")";
            return SCRAPGPU_DEVICE_GENERATION_AMD_RDNA;
        }
        reason = "AMD GPU, generation not determined (" + p.name + ")";
        return SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
    }
    if (p.vendor_id == 0x106B || p.name.find("Apple") != std::string::npos) {
        reason = "vendor " + p.name;
        return SCRAPGPU_DEVICE_GENERATION_APPLE;
    }
    reason = "unrecognised device (" + p.name + ")";
    return SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
}

const char * api_tuning_profile(scrapgpu_device_generation gen) {
    switch (gen) {
        case SCRAPGPU_DEVICE_GENERATION_AMD_GCN:  return "amd-gcn";
        case SCRAPGPU_DEVICE_GENERATION_AMD_RDNA: return "amd-rdna";
        default:                                  return "generic";
    }
}

scrapgpu_status api_device_get_info(scrapgpu_device_t * device, scrapgpu_device_info * info) {
    vk_api_device * d = api_device(device);
    if (d == nullptr) {
        return api_invalid("not a Vulkan device");
    }
    if (info == nullptr || info->struct_size < sizeof(size_t)) {
        return api_invalid("device info struct is too small");
    }
    const sgv::vk_phys & p = d->dev.phys;
    if (!d->info_done) {
        switch (p.vendor_id) {
            case 0x1002: d->vendor_name = "AMD"; break;
            case 0x8086: d->vendor_name = "Intel"; break;
            case 0x10DE: d->vendor_name = "NVIDIA"; break;
            default: {
                char buf[24];
                snprintf(buf, sizeof(buf), "vendor 0x%04x", p.vendor_id);
                d->vendor_name = buf;
            }
        }
        d->generation = api_generation(p, d->generation_reason);
        d->tuning_profile = api_tuning_profile(d->generation);
        d->api_version = vk_version_string(p.api_version);
        d->driver_version = vk_version_string(p.driver_version);
        static const char * const vars[] = {
            "SCRAPGPU_VULKAN_DEVICES", "SCRAPGPU_VULKAN_ALLOW_NVIDIA",
        };
        for (const char * v : vars) {
            if (getenv(v) == nullptr) {
                continue;
            }
            if (!d->tuning_overrides.empty()) {
                d->tuning_overrides += ' ';
            }
            d->tuning_overrides += v;
        }
        d->info_done = true;
    }

    uint64_t local_heap = 0, total_heap = 0;
    for (uint32_t i = 0; i < p.mem.memoryHeapCount; ++i) {
        total_heap += p.mem.memoryHeaps[i].size;
        if ((p.mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            local_heap += p.mem.memoryHeaps[i].size;
        }
    }

    scrapgpu_device_info full = {};
    full.struct_size            = sizeof(full);
    full.name                   = p.name.c_str();
    full.vendor                 = d->vendor_name.c_str();
    full.platform_name          = "Vulkan";
    full.driver_version         = d->driver_version.c_str();
    full.backend_name           = "vulkan";
    full.reported_api_version   = d->api_version.c_str();
    full.used_api_version       = "Vulkan 1.2";
    full.generation             = d->generation;
    full.generation_reason      = d->generation_reason.c_str();
    full.total_memory_bytes     = local_heap != 0 ? local_heap : total_heap;
    full.max_allocation_reported_bytes = p.max_allocation != 0 ? p.max_allocation : full.total_memory_bytes;
    full.max_allocation_used_bytes     = full.max_allocation_reported_bytes;
    full.max_work_group_size    = p.props.limits.maxComputeWorkGroupInvocations;
    full.local_memory_bytes     = p.props.limits.maxComputeSharedMemorySize;
    full.compute_units          = p.compute_units;
    full.warp_or_wave_width     = p.subgroup_size;
    full.quirks                 = 0;
    full.probed_quirks          = 0;
    full.tuning_profile_name    = d->tuning_profile.c_str();
    full.tuning_overrides       = d->tuning_overrides.c_str();

    std::memcpy(info, &full, std::min(info->struct_size, sizeof(full)));
    info->struct_size = std::min(info->struct_size, sizeof(full));
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// buffers
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer) {
    vk_api_device * d = api_device(device);
    if (d == nullptr || out_buffer == nullptr) {
        return api_invalid("device or out_buffer is null");
    }
    *out_buffer = nullptr;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    std::string why;
    if (!api_ensure(d, why)) {
        scrapgpu_set_last_error(why.c_str());
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    sgv::vk_buf buf = {};
    if (!sgv::vk_buf_create(&d->dev, size,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            0, &buf)) {
        return api_fail("vkCreateBuffer / vkAllocateMemory failed");
    }
    std::unique_ptr<vk_api_buffer> b(new (std::nothrow) vk_api_buffer());
    if (b == nullptr) {
        sgv::vk_buf_destroy(&d->dev, &buf);
        return api_fail("out of host memory");
    }
    b->hdr.backend = &g_vk_vtable;
    b->dev = d;
    b->buf = buf;
    *out_buffer = reinterpret_cast<scrapgpu_buffer_t *>(b.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_free(scrapgpu_buffer_t * buffer) {
    vk_api_buffer * b = api_buffer(buffer);
    if (b == nullptr) {
        return api_invalid("not a Vulkan buffer");
    }
    vk_api_device * d = b->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    // the queue may still run work that references the memory; the caller frees only after sync,
    // but draining the queue here keeps an early free from freeing in-flight memory
    if (d->dev.queue != VK_NULL_HANDLE) {
        vkQueueWaitIdle(d->dev.queue);
    }
    sgv::vk_buf_destroy(&d->dev, &b->buf);
    delete b;
    return SCRAPGPU_STATUS_OK;
}

// record a host -> device write through the staging buffer; submits and optionally waits
scrapgpu_status api_buffer_write(vk_api_stream * s, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking) {
    vk_api_buffer * b = api_buffer(buffer);
    if (s == nullptr || b == nullptr) {
        return api_invalid("stream or buffer is not a Vulkan object");
    }
    if (b->dev != s->dev) {
        return api_invalid("buffer and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (data == nullptr) {
        return api_invalid("write source is null");
    }
    if (offset > b->buf.size || size > b->buf.size - offset) {
        return api_invalid("write exceeds the buffer");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    std::string why;
    if (!api_ensure(d, why)) {
        scrapgpu_set_last_error(why.c_str());
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    void * stage = sgv::vk_staging_map(&d->dev, &s->str, size);
    if (stage == nullptr) {
        return api_fail("the staging buffer could not be allocated");
    }
    std::memcpy(stage, data, size);
    if (sgv::vk_record(&d->dev, &s->str) != VK_SUCCESS) {
        return api_fail("vkBeginCommandBuffer failed");
    }
    VkBufferCopy region = {};
    region.srcOffset = 0;
    region.dstOffset = (VkDeviceSize) offset;
    region.size = (VkDeviceSize) size;
    vkCmdCopyBuffer(s->str.cmd, s->str.staging.buf, b->buf.buf, 1, &region);
    sgv::vk_barrier(&d->dev, &s->str);
    s->str.staging.in_flight = true;
    const scrapgpu_status st = api_submit(s);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    if (blocking) {
        const VkResult r = sgv::vk_stream_sync(&d->dev, &s->str);
        if (r != VK_SUCCESS) {
            return api_vk_error("the transfer fence wait failed", r);
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// device -> host: a copy into the staging buffer, a wait, and a memcpy to the host pointer
scrapgpu_status api_buffer_read(vk_api_stream * s, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking) {
    (void) blocking; // the host may read the buffer at return, so the copy is always waited for
    vk_api_buffer * b = api_buffer(buffer);
    if (s == nullptr || b == nullptr) {
        return api_invalid("stream or buffer is not a Vulkan object");
    }
    if (b->dev != s->dev) {
        return api_invalid("buffer and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (data == nullptr) {
        return api_invalid("read destination is null");
    }
    if (offset > b->buf.size || size > b->buf.size - offset) {
        return api_invalid("read exceeds the buffer");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    std::string why;
    if (!api_ensure(d, why)) {
        scrapgpu_set_last_error(why.c_str());
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    void * stage = sgv::vk_staging_map(&d->dev, &s->str, size);
    if (stage == nullptr) {
        return api_fail("the staging buffer could not be allocated");
    }
    if (sgv::vk_record(&d->dev, &s->str) != VK_SUCCESS) {
        return api_fail("vkBeginCommandBuffer failed");
    }
    VkBufferCopy region = {};
    region.srcOffset = (VkDeviceSize) offset;
    region.dstOffset = 0;
    region.size = (VkDeviceSize) size;
    vkCmdCopyBuffer(s->str.cmd, b->buf.buf, s->str.staging.buf, 1, &region);
    sgv::vk_barrier(&d->dev, &s->str);
    s->str.staging.in_flight = true;
    const scrapgpu_status st = api_submit(s);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    const VkResult r = sgv::vk_stream_sync(&d->dev, &s->str);
    if (r != VK_SUCCESS) {
        return api_vk_error("the transfer fence wait failed", r);
    }
    std::memcpy(data, stage, size);
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_copy(vk_api_stream * s, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking) {
    vk_api_buffer * db = api_buffer(dst);
    vk_api_buffer * sb = api_buffer(src);
    if (s == nullptr || db == nullptr || sb == nullptr) {
        return api_invalid("stream or buffers are not Vulkan objects");
    }
    if (db->dev != s->dev || sb->dev != s->dev) {
        return api_invalid("buffers and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (dst_offset > db->buf.size || size > db->buf.size - dst_offset ||
        src_offset > sb->buf.size || size > sb->buf.size - src_offset) {
        return api_invalid("copy exceeds a buffer");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    std::string why;
    if (!api_ensure(d, why)) {
        scrapgpu_set_last_error(why.c_str());
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    if (sgv::vk_record(&d->dev, &s->str) != VK_SUCCESS) {
        return api_fail("vkBeginCommandBuffer failed");
    }
    VkBufferCopy region = {};
    region.srcOffset = (VkDeviceSize) src_offset;
    region.dstOffset = (VkDeviceSize) dst_offset;
    region.size = (VkDeviceSize) size;
    vkCmdCopyBuffer(s->str.cmd, sb->buf.buf, db->buf.buf, 1, &region);
    sgv::vk_barrier(&d->dev, &s->str);
    const scrapgpu_status st = api_submit(s);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    if (blocking) {
        const VkResult r = sgv::vk_stream_sync(&d->dev, &s->str);
        if (r != VK_SUCCESS) {
            return api_vk_error("the copy fence wait failed", r);
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_buffer_fill(vk_api_stream * s, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking) {
    vk_api_buffer * b = api_buffer(buffer);
    if (s == nullptr || b == nullptr) {
        return api_invalid("stream or buffer is not a Vulkan object");
    }
    if (b->dev != s->dev) {
        return api_invalid("buffer and stream belong to different devices");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (offset > b->buf.size || size > b->buf.size - offset) {
        return api_invalid("fill exceeds the buffer");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    std::string why;
    if (!api_ensure(d, why)) {
        scrapgpu_set_last_error(why.c_str());
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    const uint32_t pattern = (uint32_t) byte * 0x01010101u;
    // vkCmdFillBuffer needs a 4-aligned dstOffset and a multiple-of-4 size: the bytes at both ends
    // go through a small staging patch instead (mapped before recording, it may wait on the fence)
    const VkDeviceSize start4 = ((VkDeviceSize) offset + 3) & ~(VkDeviceSize) 3;
    const size_t lead = (size_t) std::min<VkDeviceSize>((VkDeviceSize) size, start4 - (VkDeviceSize) offset);
    const size_t mid  = (size - lead) & ~(size_t) 3;
    const size_t tail = size - lead - mid;
    void * stage = nullptr;
    if (lead + tail != 0) {
        stage = sgv::vk_staging_map(&d->dev, &s->str, lead + tail);
        if (stage == nullptr) {
            return api_fail("the staging buffer could not be allocated");
        }
        std::memset(stage, byte, lead + tail);
    }
    if (sgv::vk_record(&d->dev, &s->str) != VK_SUCCESS) {
        return api_fail("vkBeginCommandBuffer failed");
    }
    if (lead != 0) {
        VkBufferCopy region = { 0, (VkDeviceSize) offset, (VkDeviceSize) lead };
        vkCmdCopyBuffer(s->str.cmd, s->str.staging.buf, b->buf.buf, 1, &region);
        sgv::vk_barrier(&s->dev->dev, &s->str);
        s->str.staging.in_flight = true;
    }
    if (mid != 0) {
        vkCmdFillBuffer(s->str.cmd, b->buf.buf, (VkDeviceSize) offset + lead, (VkDeviceSize) mid, pattern);
        sgv::vk_barrier(&s->dev->dev, &s->str);
    }
    if (tail != 0) {
        VkBufferCopy region = { (VkDeviceSize) lead, (VkDeviceSize) offset + lead + mid, (VkDeviceSize) tail };
        vkCmdCopyBuffer(s->str.cmd, s->str.staging.buf, b->buf.buf, 1, &region);
        sgv::vk_barrier(&s->dev->dev, &s->str);
        s->str.staging.in_flight = true;
    }
    sgv::vk_barrier(&d->dev, &s->str);
    const scrapgpu_status st = api_submit(s);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    if (blocking) {
        const VkResult r = sgv::vk_stream_sync(&d->dev, &s->str);
        if (r != VK_SUCCESS) {
            return api_vk_error("the fill fence wait failed", r);
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// streams
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_stream_create(scrapgpu_device_t * device, scrapgpu_stream_t ** out_stream) {
    vk_api_device * d = api_device(device);
    if (d == nullptr || out_stream == nullptr) {
        return api_invalid("device or out_stream is null");
    }
    *out_stream = nullptr;
    std::unique_ptr<vk_api_stream> s(new (std::nothrow) vk_api_stream());
    if (s == nullptr) {
        return api_fail("out of host memory");
    }
    s->hdr.backend = &g_vk_vtable;
    s->dev = d;
    *out_stream = reinterpret_cast<scrapgpu_stream_t *>(s.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_stream_free(scrapgpu_stream_t * stream) {
    vk_api_stream * s = api_stream(stream);
    if (s == nullptr) {
        return api_invalid("not a Vulkan stream");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    sgv::vk_stream_free(&d->dev, &s->str);
    delete s;
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_stream_sync(scrapgpu_stream_t * stream) {
    vk_api_stream * s = api_stream(stream);
    if (s == nullptr) {
        return api_invalid("not a Vulkan stream");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    const VkResult r = sgv::vk_stream_sync(&d->dev, &s->str);
    if (r != VK_SUCCESS) {
        return api_vk_error("the fence wait failed", r);
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// tensor transfers (NATURAL layout only; R32 does not exist on this backend)
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_tensor_transfer(vk_api_stream * s, const scrapgpu_tensor_desc * raw, uint8_t * host, size_t size, bool to_dev) {
    if (s == nullptr || raw == nullptr) {
        return api_invalid("stream or tensor desc is null");
    }
    if (raw->struct_size < offsetof(scrapgpu_tensor_desc, offset) + sizeof(raw->offset)) {
        return api_invalid("tensor desc is too small");
    }
    const scrapgpu_tensor_desc td = api_load_desc(*raw);
    vk_api_buffer * b = api_buffer(td.buffer);
    if (b == nullptr) {
        return api_invalid("tensor desc has no Vulkan buffer");
    }
    if (b->dev != s->dev) {
        return api_invalid("tensor buffer and stream belong to different devices");
    }
    const sgt::TensorDesc d = sgt::desc_of(td);
    if (td.offset > b->buf.size || size > b->buf.size - td.offset) {
        return api_invalid("transfer exceeds the buffer");
    }
    if (size > sgt::nbytes(d)) {
        return api_invalid("transfer size exceeds the tensor");
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (d.layout != SCRAPGPU_LAYOUT_NATURAL) {
        return api_unsupported("the Vulkan backend only supports the natural layout");
    }
    if (!sgt::is_contiguous(d)) {
        return api_unsupported("tensor is not contiguous; use buffer_read / buffer_write for raw strided transfers");
    }
    return to_dev ? api_buffer_write(s, reinterpret_cast<scrapgpu_buffer_t *>(b), td.offset, host, size, 1)
                  : api_buffer_read(s, reinterpret_cast<scrapgpu_buffer_t *>(b), td.offset, host, size, 1);
}

scrapgpu_status api_tensor_upload(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, const void * host, size_t size) {
    vk_api_stream * s = api_stream(stream);
    return api_tensor_transfer(s, desc, (uint8_t *) const_cast<void *>(host), size, true);
}

scrapgpu_status api_tensor_download(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, void * host, size_t size) {
    vk_api_stream * s = api_stream(stream);
    return api_tensor_transfer(s, desc, (uint8_t *) host, size, false);
}

// ---------------------------------------------------------------------------------------------
// ops and plans
// ---------------------------------------------------------------------------------------------

scrapgpu_status api_op_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason) {
    vk_api_device * d = api_device(device);
    if (d == nullptr) {
        if (reason != nullptr) {
            *reason = "not a Vulkan device";
        }
        return api_invalid("not a Vulkan device");
    }
    return api_supports(d, desc, reason);
}

scrapgpu_status api_op_execute(scrapgpu_stream_t * stream, const scrapgpu_op_desc * desc) {
    vk_api_stream * s = api_stream(stream);
    if (s == nullptr) {
        return api_invalid("not a Vulkan stream");
    }
    if (desc == nullptr) {
        return api_invalid("op desc is null");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    std::string why;
    if (!api_ensure(d, why)) {
        scrapgpu_set_last_error(why.c_str());
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    const scrapgpu_status st = api_exec_desc(s, desc);
    if (st != SCRAPGPU_STATUS_OK) {
        return st;
    }
    return api_submit(s);
}

scrapgpu_status api_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan) {
    vk_api_device * d = api_device(device);
    if (d == nullptr || desc == nullptr || out_plan == nullptr ||
        desc->struct_size < offsetof(scrapgpu_plan_desc, node_count) + sizeof(desc->node_count) ||
        (desc->node_count > 0 && desc->nodes == nullptr)) {
        return api_invalid("device, plan desc or node list is invalid");
    }
    *out_plan = nullptr;

    std::unique_ptr<vk_api_plan> plan(new (std::nothrow) vk_api_plan());
    if (plan == nullptr) {
        return api_fail("out of host memory");
    }
    plan->hdr.backend = &g_vk_vtable;
    plan->dev = d;
    plan->nodes.reserve(desc->node_count);
    for (size_t i = 0; i < desc->node_count; ++i) {
        const scrapgpu_plan_node & src = desc->nodes[i];
        if (src.struct_size < offsetof(scrapgpu_plan_node, op) + sizeof(src.op)) {
            return api_invalid("plan node is too small");
        }
        const char * why = nullptr;
        const scrapgpu_status st = api_supports(d, &src.op, &why);
        if (st != SCRAPGPU_STATUS_OK) {
            char buf[256];
            snprintf(buf, sizeof(buf), "plan node %zu: %s", i, why != nullptr ? why : "not supported");
            scrapgpu_set_last_error(buf);
            return st;
        }
        vk_api_plan_node n;
        n.desc = src.op;
        n.inputs.assign(src.op.inputs, src.op.inputs + src.op.input_count);
        n.outputs.assign(src.op.outputs, src.op.outputs + src.op.output_count);
        if (src.op.params != nullptr && src.op.params_size > 0) {
            const uint8_t * p = static_cast<const uint8_t *>(src.op.params);
            n.params.assign(p, p + src.op.params_size);
        }
        n.desc.inputs  = n.inputs.data();
        n.desc.outputs = n.outputs.data();
        n.desc.params  = n.params.empty() ? nullptr : n.params.data();
        plan->nodes.push_back(std::move(n));
    }
    *out_plan = reinterpret_cast<scrapgpu_plan_t *>(plan.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status api_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan_) {
    vk_api_stream * s = api_stream(stream);
    auto * plan = reinterpret_cast<vk_api_plan *>(plan_);
    if (s == nullptr || plan == nullptr || plan->hdr.backend != &g_vk_vtable) {
        return api_invalid("stream or plan is not a Vulkan object");
    }
    if (plan->dev != s->dev) {
        return api_invalid("plan and stream belong to different devices");
    }
    vk_api_device * d = s->dev;
    std::lock_guard<std::mutex> lock(d->dev.mutex);
    std::string why;
    if (!api_ensure(d, why)) {
        scrapgpu_set_last_error(why.c_str());
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    for (auto & node : plan->nodes) {
        const scrapgpu_status st = api_exec_desc(s, &node.desc);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    return api_submit(s);
}

scrapgpu_status api_plan_free(scrapgpu_plan_t * plan_) {
    auto * plan = reinterpret_cast<vk_api_plan *>(plan_);
    if (plan == nullptr || plan->hdr.backend != &g_vk_vtable) {
        return api_invalid("not a Vulkan plan");
    }
    delete plan;
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------------------------

const scrapgpu_backend_vtable g_vk_vtable = {
    /* .struct_size        = */ sizeof(scrapgpu_backend_vtable),
    /* .name               = */ "vulkan",
    /* .userdata           = */ nullptr,
    /* .get_device_count   = */ api_get_device_count,
    /* .get_device         = */ api_get_device,
    /* .device_get_info    = */ api_device_get_info,
    /* .buffer_alloc       = */ api_buffer_alloc,
    /* .buffer_free        = */ api_buffer_free,
    /* .buffer_write       = */ [](scrapgpu_stream_t * s, scrapgpu_buffer_t * b, uint64_t o, const void * p, size_t n, int bl) {
        return api_buffer_write(api_stream(s), b, o, p, n, bl);
    },
    /* .buffer_read        = */ [](scrapgpu_stream_t * s, scrapgpu_buffer_t * b, uint64_t o, void * p, size_t n, int bl) {
        return api_buffer_read(api_stream(s), b, o, p, n, bl);
    },
    /* .buffer_copy        = */ [](scrapgpu_stream_t * s, scrapgpu_buffer_t * db, uint64_t do_, scrapgpu_buffer_t * sb, uint64_t so, size_t n, int bl) {
        return api_buffer_copy(api_stream(s), db, do_, sb, so, n, bl);
    },
    /* .buffer_fill        = */ [](scrapgpu_stream_t * s, scrapgpu_buffer_t * b, uint64_t o, size_t n, uint8_t byte, int bl) {
        return api_buffer_fill(api_stream(s), b, o, n, byte, bl);
    },
    /* .stream_sync        = */ api_stream_sync,
    /* .op_supported       = */ api_op_supported,
    /* .plan_build         = */ api_plan_build,
    /* .plan_execute       = */ api_plan_execute,
    /* .plan_free          = */ api_plan_free,
    /* .stream_create      = */ api_stream_create,
    /* .stream_free        = */ api_stream_free,
    /* .tensor_upload      = */ api_tensor_upload,
    /* .tensor_download    = */ api_tensor_download,
    /* .op_execute         = */ api_op_execute,
    /* .stream_device      = */ [](scrapgpu_stream_t * s) -> scrapgpu_device_t * {
        vk_api_stream * st = api_stream(s);
        return st == nullptr ? nullptr : reinterpret_cast<scrapgpu_device_t *>(st->dev);
    },
};

} // namespace

scrapgpu_status scrapgpu_vulkan_register(void) {
    static std::mutex reg_mutex;
    static bool registered = false;
    static scrapgpu_status reg_status = SCRAPGPU_STATUS_OK;

    std::lock_guard<std::mutex> lock(reg_mutex);
    if (registered) {
        return reg_status;
    }
    registered = true;
    for (sgv::vk_phys & p : sgv::vk_probe_devices()) {
        auto d = std::make_unique<vk_api_device>();
        d->hdr.backend = &g_vk_vtable;
        d->dev.phys = p;
        g_devices.push_back(std::move(d));
    }
    reg_status = scrapgpu_register_backend(&g_vk_vtable);
    return reg_status;
}
