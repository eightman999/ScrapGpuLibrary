#include "scrapgpu/attention.h"

#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/op.h"
#include "scrapgpu/status.h"
#include "scrapgpu/tensor.h"

#include "tensor/tensor.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <new>
#include <vector>

namespace {

scrapgpu_status fail(scrapgpu_status status, const char * message) {
    scrapgpu_set_last_error(message);
    return status;
}

scrapgpu_status invalid(const char * message) {
    return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, message);
}

scrapgpu_status unsupported(const char * message) {
    return fail(SCRAPGPU_STATUS_UNSUPPORTED, message);
}

scrapgpu_status oom(const char * message) {
    return fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, message);
}

const scrapgpu_backend_vtable * backend_of(const void * object) {
    return object == nullptr ? nullptr : static_cast<const scrapgpu_object_header *>(object)->backend;
}

scrapgpu_attention_params load_params(const scrapgpu_attention_params & in, bool * ok) {
    *ok = in.struct_size >= sizeof(scrapgpu_attention_params);
    scrapgpu_attention_params out = {};
    if (*ok) {
        std::memcpy(&out, &in, std::min(in.struct_size, sizeof(out)));
        out.struct_size = sizeof(out);
    }
    return out;
}

scrapgpu_tensor_desc load_desc(const scrapgpu_tensor_desc & in, bool * ok) {
    *ok = in.struct_size >= sizeof(size_t);
    scrapgpu_tensor_desc out = {};
    if (*ok) {
        std::memcpy(&out, &in, std::min(in.struct_size, sizeof(out)));
        out.struct_size = sizeof(out);
    }
    return out;
}

bool is_positive_shape(const scrapgpu_tensor_desc & d) {
    return d.ne[0] > 0 && d.ne[1] > 0 && d.ne[2] > 0 && d.ne[3] > 0;
}

bool intervals_overlap(uint64_t a_off, uint64_t a_size, uint64_t b_off, uint64_t b_size) {
    return a_off < b_off ? b_off - a_off < a_size : a_off - b_off < b_size;
}

bool overlaps(const scrapgpu_tensor_desc & a, const scrapgpu_tensor_desc & b) {
    if (a.buffer == nullptr || b.buffer == nullptr || a.buffer != b.buffer) {
        return false;
    }
    const uint64_t a_size = sgt::nbytes(sgt::desc_of(a));
    const uint64_t b_size = sgt::nbytes(sgt::desc_of(b));
    return intervals_overlap(a.offset, a_size, b.offset, b_size);
}

bool saturating_offset(const scrapgpu_tensor_desc & d, int64_t i1, int64_t i2, int64_t i3, uint64_t * out) {
    uint64_t off = d.offset;
    off = sgt::add_sat_u64(off, sgt::mul_sat_u64(static_cast<uint64_t>(i1), d.nb[1]));
    off = sgt::add_sat_u64(off, sgt::mul_sat_u64(static_cast<uint64_t>(i2), d.nb[2]));
    off = sgt::add_sat_u64(off, sgt::mul_sat_u64(static_cast<uint64_t>(i3), d.nb[3]));
    if (off == std::numeric_limits<uint64_t>::max()) {
        return false;
    }
    *out = off;
    return true;
}

scrapgpu_tensor_desc make_2d_desc(scrapgpu_buffer_t * buffer, uint64_t offset, int64_t ne0, int64_t ne1) {
    const int64_t ne[4] = {ne0, ne1, 1, 1};
    const sgt::TensorDesc td = sgt::make_contiguous(SCRAPGPU_DTYPE_F32, ne, SCRAPGPU_LAYOUT_NATURAL);
    scrapgpu_tensor_desc out{};
    out.struct_size = sizeof(out);
    out.dtype = td.dtype;
    out.buffer = buffer;
    out.offset = offset;
    out.layout = td.layout;
    for (int i = 0; i < sgt::kMaxDims; ++i) {
        out.ne[i] = td.ne[i];
        out.nb[i] = td.nb[i];
    }
    return out;
}

bool is_masked_value(float v) {
    return std::isinf(v) && v < 0.0f;
}

struct AttentionSetup {
    scrapgpu_attention_params params{};
    scrapgpu_tensor_desc query{};
    scrapgpu_tensor_desc key{};
    scrapgpu_tensor_desc value{};
    scrapgpu_tensor_desc mask{};
    scrapgpu_tensor_desc output{};
    bool has_mask = false;
    int64_t d = 0;
    int64_t q = 0;
    int64_t k = 0;
    int64_t dv = 0;
    int64_t hq = 0;
    int64_t hkv = 0;
    int64_t batches = 0;
    int64_t heads_per_kv = 0;
    float scale = 0.0f;
    bool need_mask_workspace = false;
    int64_t tile_q = 0;
    uint64_t v_t_bytes = 0;
    uint64_t score_tile_bytes = 0;
    uint64_t mask_tile_bytes = 0;
    uint64_t workspace_bytes = 0;
    const scrapgpu_backend_vtable * backend = nullptr;
};

scrapgpu_status validate_f32_contiguous(const scrapgpu_tensor_desc & d, const char * name) {
    if (d.dtype != SCRAPGPU_DTYPE_F32) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s must be F32", name);
        return unsupported(buf);
    }
    if (d.layout != SCRAPGPU_LAYOUT_NATURAL) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s must use the natural layout", name);
        return invalid(buf);
    }
    if (!is_positive_shape(d)) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s must have positive dimensions", name);
        return invalid(buf);
    }
    if (!sgt::is_contiguous_strict(sgt::desc_of(d))) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s must be contiguous", name);
        return invalid(buf);
    }
    if (d.buffer == nullptr) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s has no buffer", name);
        return invalid(buf);
    }
    const uint64_t bytes = sgt::nbytes(sgt::desc_of(d));
    if (bytes == 0 || bytes == UINT64_MAX || bytes > SIZE_MAX || d.offset > UINT64_MAX - bytes) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s byte size is zero or overflowed", name);
        return invalid(buf);
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status validate_setup(scrapgpu_device_t * device, scrapgpu_stream_t * stream,
                               const scrapgpu_attention_params * raw_params, AttentionSetup * out) {
    if (device == nullptr || stream == nullptr || raw_params == nullptr || out == nullptr) {
        return invalid("device, stream, params or output is null");
    }

    bool ok = false;
    out->params = load_params(*raw_params, &ok);
    if (!ok) {
        return invalid("attention params struct is too small");
    }
    if (out->params.query == nullptr || out->params.key == nullptr || out->params.value == nullptr || out->params.output == nullptr) {
        return invalid("query, key, value and output descriptors are required");
    }

    out->query = load_desc(*out->params.query, &ok);
    if (!ok) return invalid("query tensor desc is too small");
    out->key = load_desc(*out->params.key, &ok);
    if (!ok) return invalid("key tensor desc is too small");
    out->value = load_desc(*out->params.value, &ok);
    if (!ok) return invalid("value tensor desc is too small");
    out->output = load_desc(*out->params.output, &ok);
    if (!ok) return invalid("output tensor desc is too small");
    out->has_mask = out->params.mask != nullptr;
    if (out->has_mask) {
        out->mask = load_desc(*out->params.mask, &ok);
        if (!ok) return invalid("mask tensor desc is too small");
    }

    const scrapgpu_status sq = validate_f32_contiguous(out->query, "query");
    if (sq != SCRAPGPU_STATUS_OK) return sq;
    const scrapgpu_status sk = validate_f32_contiguous(out->key, "key");
    if (sk != SCRAPGPU_STATUS_OK) return sk;
    const scrapgpu_status sv = validate_f32_contiguous(out->value, "value");
    if (sv != SCRAPGPU_STATUS_OK) return sv;
    const scrapgpu_status so = validate_f32_contiguous(out->output, "output");
    if (so != SCRAPGPU_STATUS_OK) return so;

    out->d = out->query.ne[0];
    out->q = out->query.ne[1];
    out->hq = out->query.ne[2];
    out->batches = out->query.ne[3];
    out->k = out->key.ne[1];
    out->hkv = out->key.ne[2];
    out->dv = out->value.ne[0];

    if (out->key.ne[0] != out->d) {
        return invalid("query and key head dimensions differ");
    }
    if (out->value.ne[1] != out->k) {
        return invalid("key and value sequence lengths differ");
    }
    if (out->value.ne[2] != out->hkv) {
        return invalid("key and value head counts differ");
    }
    if (out->key.ne[3] != out->batches || out->value.ne[3] != out->batches) {
        return invalid("query, key and value batch counts differ");
    }
    if (out->output.ne[0] != out->dv || out->output.ne[1] != out->q ||
        out->output.ne[2] != out->hq || out->output.ne[3] != out->batches) {
        return invalid("output must have the shape [Dv, Q, Hq, B]");
    }
    if (out->hkv <= 0 || out->hq % out->hkv != 0) {
        return invalid("Hq must be divisible by Hkv");
    }
    out->heads_per_kv = out->hq / out->hkv;

    if (overlaps(out->output, out->query) || overlaps(out->output, out->key) || overlaps(out->output, out->value) ||
        (out->has_mask && overlaps(out->output, out->mask))) {
        return invalid("output must not overlap query, key, value or mask");
    }

    out->backend = backend_of(device);
    const scrapgpu_backend_vtable * stream_backend = backend_of(stream);
    if (out->backend == nullptr || stream_backend == nullptr || out->backend != stream_backend) {
        return invalid("device and stream belong to different backends");
    }
    if (out->backend->stream_device == nullptr || out->backend->stream_device(stream) != device) {
        return invalid("stream does not belong to the supplied device");
    }

    for (const scrapgpu_tensor_desc * t : {&out->query, &out->key, &out->value, &out->output}) {
        if (backend_of(t->buffer) != out->backend) {
            return invalid("all tensors must belong to the same backend as the device");
        }
    }
    if (out->has_mask && backend_of(out->mask.buffer) != out->backend) {
        return invalid("mask must belong to the same backend as the device");
    }

    if (out->has_mask) {
        const scrapgpu_status sm = validate_f32_contiguous(out->mask, "mask");
        if (sm != SCRAPGPU_STATUS_OK) return sm;
        if (out->mask.ne[0] != out->k || out->mask.ne[1] != out->q ||
            (out->mask.ne[2] != 1 && out->mask.ne[2] != out->hq) ||
            (out->mask.ne[3] != 1 && out->mask.ne[3] != out->batches)) {
            return invalid("mask must have the shape [K, Q, 1 or Hq, 1 or B]");
        }
    }

    if (!std::isfinite(out->params.scale)) return invalid("attention scale must be finite");
    out->scale = out->params.scale;
    out->need_mask_workspace = out->params.causal != 0 || out->has_mask;
    out->v_t_bytes = sgt::mul_sat_u64(static_cast<uint64_t>(out->k), sgt::mul_sat_u64(static_cast<uint64_t>(out->dv), sizeof(float)));
    out->score_tile_bytes = sgt::mul_sat_u64(static_cast<uint64_t>(out->k), sizeof(float));
    out->mask_tile_bytes = out->need_mask_workspace ? out->score_tile_bytes : 0;
    if (out->v_t_bytes == std::numeric_limits<uint64_t>::max() ||
        out->score_tile_bytes == std::numeric_limits<uint64_t>::max() ||
        out->mask_tile_bytes == std::numeric_limits<uint64_t>::max()) {
        return unsupported("attention workspace size overflowed");
    }

    const uint64_t per_query = sgt::add_sat_u64(out->score_tile_bytes, out->mask_tile_bytes);
    if (per_query == 0 || per_query == std::numeric_limits<uint64_t>::max()) {
        return unsupported("attention per-query workspace overflowed");
    }
    const uint64_t min_budget = sgt::add_sat_u64(out->v_t_bytes, per_query);
    uint64_t budget = out->params.max_workspace_bytes;
    scrapgpu_device_info info{};
    scrapgpu_device_info_init(&info, sizeof(info));
    const scrapgpu_status info_status = scrapgpu_device_get_info(device, &info);
    if (info_status != SCRAPGPU_STATUS_OK) return info_status;
    if (info.max_allocation_used_bytes != 0) budget = std::min(budget, info.max_allocation_used_bytes);
    if (min_budget == std::numeric_limits<uint64_t>::max() || budget < min_budget) {
        return oom("attention workspace budget is too small for one query tile");
    }
    const uint64_t avail = budget - out->v_t_bytes;
    const uint64_t max_tile = std::min<uint64_t>(static_cast<uint64_t>(out->q), avail / per_query);
    if (max_tile == 0 || max_tile > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return oom("attention workspace budget produced an invalid tile size");
    }
    out->tile_q = static_cast<int64_t>(max_tile);
    out->workspace_bytes = sgt::add_sat_u64(out->v_t_bytes, sgt::mul_sat_u64(per_query, max_tile));
    if (out->workspace_bytes == std::numeric_limits<uint64_t>::max() ||
        out->workspace_bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return unsupported("attention workspace does not fit a backend allocation");
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status describe_unsupported(scrapgpu_status status, const char * what, const char * reason) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s: %s", what, reason != nullptr && reason[0] != '\0' ? reason : "not supported");
    return fail(status, buf);
}

scrapgpu_status check_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char * what) {
    const char * reason = nullptr;
    const scrapgpu_status st = scrapgpu_op_supported(device, desc, &reason);
    return st == SCRAPGPU_STATUS_OK ? st : describe_unsupported(st, what, reason);
}

scrapgpu_status support_cpy_transpose(scrapgpu_device_t * device, const scrapgpu_tensor_desc & src, const scrapgpu_tensor_desc & dst) {
    scrapgpu_cpy_params params{sizeof(params), 1};
    std::array<scrapgpu_tensor_desc, 1> inputs = {src};
    // CPY follows the input's logical strides; allow_transpose is permission,
    // not a request to transpose a contiguous input.
    std::swap(inputs[0].ne[0], inputs[0].ne[1]);
    std::swap(inputs[0].nb[0], inputs[0].nb[1]);
    std::array<scrapgpu_tensor_desc, 1> outputs = {dst};
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = SCRAPGPU_OP_CPY;
    desc.inputs = inputs.data();
    desc.input_count = inputs.size();
    desc.outputs = outputs.data();
    desc.output_count = outputs.size();
    desc.params = &params;
    desc.params_size = sizeof(params);
    return check_supported(device, &desc, "attention V transpose");
}

scrapgpu_status support_mul_mat(scrapgpu_device_t * device, const scrapgpu_tensor_desc & a,
                                const scrapgpu_tensor_desc & b, const scrapgpu_tensor_desc & d,
                                const char * what) {
    scrapgpu_mul_mat_params params{sizeof(params), 0};
    std::array<scrapgpu_tensor_desc, 2> inputs = {a, b};
    std::array<scrapgpu_tensor_desc, 1> outputs = {d};
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = SCRAPGPU_OP_MUL_MAT;
    desc.inputs = inputs.data();
    desc.input_count = inputs.size();
    desc.outputs = outputs.data();
    desc.output_count = outputs.size();
    desc.params = &params;
    desc.params_size = sizeof(params);
    return check_supported(device, &desc, what);
}

scrapgpu_status support_softmax(scrapgpu_device_t * device, const scrapgpu_tensor_desc & scores,
                                const scrapgpu_tensor_desc * mask, float scale) {
    scrapgpu_soft_max_params params{sizeof(params), scale, mask};
    std::array<scrapgpu_tensor_desc, 2> inputs = {scores, mask != nullptr ? *mask : scrapgpu_tensor_desc{}};
    std::array<scrapgpu_tensor_desc, 1> outputs = {scores};
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = SCRAPGPU_OP_SOFT_MAX;
    desc.inputs = inputs.data();
    desc.input_count = mask != nullptr ? 2u : 1u;
    desc.outputs = outputs.data();
    desc.output_count = outputs.size();
    desc.params = &params;
    desc.params_size = sizeof(params);
    return check_supported(device, &desc, "attention softmax");
}

scrapgpu_status run_cpy_transpose(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc & src, const scrapgpu_tensor_desc & dst) {
    scrapgpu_cpy_params params{sizeof(params), 1};
    std::array<scrapgpu_tensor_desc, 1> inputs = {src};
    std::swap(inputs[0].ne[0], inputs[0].ne[1]);
    std::swap(inputs[0].nb[0], inputs[0].nb[1]);
    std::array<scrapgpu_tensor_desc, 1> outputs = {dst};
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = SCRAPGPU_OP_CPY;
    desc.inputs = inputs.data();
    desc.input_count = inputs.size();
    desc.outputs = outputs.data();
    desc.output_count = outputs.size();
    desc.params = &params;
    desc.params_size = sizeof(params);
    return scrapgpu_op_execute(stream, &desc);
}

scrapgpu_status run_mul_mat(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc & a,
                            const scrapgpu_tensor_desc & b, const scrapgpu_tensor_desc & d) {
    scrapgpu_mul_mat_params params{sizeof(params), 0};
    std::array<scrapgpu_tensor_desc, 2> inputs = {a, b};
    std::array<scrapgpu_tensor_desc, 1> outputs = {d};
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = SCRAPGPU_OP_MUL_MAT;
    desc.inputs = inputs.data();
    desc.input_count = inputs.size();
    desc.outputs = outputs.data();
    desc.output_count = outputs.size();
    desc.params = &params;
    desc.params_size = sizeof(params);
    return scrapgpu_op_execute(stream, &desc);
}

scrapgpu_status run_softmax(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc & scores,
                            const scrapgpu_tensor_desc * mask, float scale) {
    scrapgpu_soft_max_params params{sizeof(params), scale, mask};
    std::array<scrapgpu_tensor_desc, 2> inputs = {scores, mask != nullptr ? *mask : scrapgpu_tensor_desc{}};
    std::array<scrapgpu_tensor_desc, 1> outputs = {scores};
    scrapgpu_op_desc desc{};
    desc.struct_size = sizeof(desc);
    desc.op = SCRAPGPU_OP_SOFT_MAX;
    desc.inputs = inputs.data();
    desc.input_count = mask != nullptr ? 2u : 1u;
    desc.outputs = outputs.data();
    desc.output_count = outputs.size();
    desc.params = &params;
    desc.params_size = sizeof(params);
    return scrapgpu_op_execute(stream, &desc);
}

struct BufferOwner {
    scrapgpu_buffer_t * buffer = nullptr;
    ~BufferOwner() {
        scrapgpu_buffer_free(buffer);
    }
};

scrapgpu_status make_selected_tile(const scrapgpu_tensor_desc & src, int64_t q_offset, int64_t head, int64_t batch,
                                   int64_t ne0, int64_t ne1, scrapgpu_tensor_desc * out) {
    uint64_t offset = 0;
    if (!saturating_offset(src, q_offset, head, batch, &offset)) {
        return unsupported("tensor slice offset overflowed");
    }
    *out = make_2d_desc(src.buffer, offset, ne0, ne1);
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status make_mask_tile(const AttentionSetup & setup, int64_t q_offset, int64_t head, int64_t batch,
                               int64_t q_len, scrapgpu_tensor_desc * out) {
    const int64_t mh = setup.mask.ne[2] == 1 ? 0 : head;
    const int64_t mb = setup.mask.ne[3] == 1 ? 0 : batch;
    return make_selected_tile(setup.mask, q_offset, mh, mb, setup.k, q_len, out);
}

scrapgpu_status build_mask_tile(const AttentionSetup & setup, scrapgpu_stream_t * stream,
                                int64_t q_offset, int64_t q_len, int64_t head, int64_t batch,
                                std::vector<float> * mask_host, std::vector<uint8_t> * fully_masked) {
    const size_t elems = static_cast<size_t>(setup.k) * static_cast<size_t>(q_len);
    mask_host->assign(elems, 0.0f);
    fully_masked->assign(static_cast<size_t>(q_len), 0u);

    if (setup.has_mask) {
        scrapgpu_tensor_desc mask_tile{};
        const scrapgpu_status st = make_mask_tile(setup, q_offset, head, batch, q_len, &mask_tile);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
        const scrapgpu_status dl = scrapgpu_tensor_download(stream, &mask_tile, mask_host->data(), elems * sizeof(float));
        if (dl != SCRAPGPU_STATUS_OK) {
            return dl;
        }
    }

    for (int64_t q = 0; q < q_len; ++q) {
        const uint64_t visible = setup.params.query_position_offset > std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(q_offset + q)
                ? std::numeric_limits<uint64_t>::max()
                : setup.params.query_position_offset + static_cast<uint64_t>(q_offset + q);
        bool any_open = false;
        float * col = mask_host->data() + static_cast<size_t>(q) * static_cast<size_t>(setup.k);
        for (int64_t i = 0; i < setup.k; ++i) {
            if (setup.params.causal != 0 && static_cast<uint64_t>(i) > visible) {
                col[i] = -std::numeric_limits<float>::infinity();
            }
            if (!is_masked_value(col[i])) {
                any_open = true;
            }
        }
        (*fully_masked)[static_cast<size_t>(q)] = any_open ? 0u : 1u;
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status zero_fully_masked_columns(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc & scores,
                                          int64_t k, int64_t q_len, const std::vector<uint8_t> & fully_masked,
                                          std::vector<float> * zero_column) {
    if (zero_column->size() != static_cast<size_t>(k)) {
        zero_column->assign(static_cast<size_t>(k), 0.0f);
    }
    for (int64_t q = 0; q < q_len; ++q) {
        if (fully_masked[static_cast<size_t>(q)] == 0) {
            continue;
        }
        scrapgpu_tensor_desc col = make_2d_desc(scores.buffer,
                                                scores.offset + static_cast<uint64_t>(q) * scores.nb[1],
                                                k, 1);
        const scrapgpu_status st = scrapgpu_tensor_upload(stream, &col, zero_column->data(), zero_column->size() * sizeof(float));
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status check_supported_tiles(scrapgpu_device_t * device, const AttentionSetup & setup) {
    const scrapgpu_tensor_desc k_desc = make_2d_desc(nullptr, 0, setup.d, setup.k);
    const scrapgpu_tensor_desc v_desc = make_2d_desc(nullptr, 0, setup.dv, setup.k);
    const scrapgpu_tensor_desc vt_desc = make_2d_desc(nullptr, 0, setup.k, setup.dv);
    const int64_t tiles[2] = {setup.tile_q, setup.q % setup.tile_q == 0 ? 0 : setup.q % setup.tile_q};

    const scrapgpu_status sc = support_cpy_transpose(device, v_desc, vt_desc);
    if (sc != SCRAPGPU_STATUS_OK) {
        return sc;
    }

    for (int64_t q_len : tiles) {
        if (q_len <= 0) {
            continue;
        }
        const scrapgpu_tensor_desc q_desc = make_2d_desc(nullptr, 0, setup.d, q_len);
        const scrapgpu_tensor_desc scores = make_2d_desc(nullptr, 0, setup.k, q_len);
        const scrapgpu_tensor_desc out_desc = make_2d_desc(nullptr, 0, setup.dv, q_len);
        scrapgpu_tensor_desc mask_desc{};
        scrapgpu_tensor_desc * mask_ptr = nullptr;
        if (setup.need_mask_workspace) {
            mask_desc = make_2d_desc(nullptr, 0, setup.k, q_len);
            mask_ptr = &mask_desc;
        }
        const scrapgpu_status s0 = support_mul_mat(device, k_desc, q_desc, scores, "attention score matmul");
        if (s0 != SCRAPGPU_STATUS_OK) {
            return s0;
        }
        const scrapgpu_status s1 = support_softmax(device, scores, mask_ptr, setup.scale);
        if (s1 != SCRAPGPU_STATUS_OK) {
            return s1;
        }
        const scrapgpu_status s2 = support_mul_mat(device, vt_desc, scores, out_desc, "attention value matmul");
        if (s2 != SCRAPGPU_STATUS_OK) {
            return s2;
        }
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status execute_attention(scrapgpu_device_t * device, scrapgpu_stream_t * stream,
                                  const scrapgpu_attention_params * raw_params) {
    AttentionSetup setup;
    {
        const scrapgpu_status st = validate_setup(device, stream, raw_params, &setup);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }
    {
        const scrapgpu_status st = check_supported_tiles(device, setup);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }

    BufferOwner workspace;
    if (setup.workspace_bytes != 0) {
        const scrapgpu_status st = scrapgpu_buffer_alloc(device, static_cast<size_t>(setup.workspace_bytes), &workspace.buffer);
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
    }

    scrapgpu_tensor_desc vt_workspace = make_2d_desc(workspace.buffer, 0, setup.k, setup.dv);
    scrapgpu_tensor_desc scores_workspace = make_2d_desc(workspace.buffer, setup.v_t_bytes, setup.k, setup.tile_q);
    scrapgpu_tensor_desc mask_workspace{};
    if (setup.need_mask_workspace) {
        mask_workspace = make_2d_desc(workspace.buffer, setup.v_t_bytes + setup.score_tile_bytes * static_cast<uint64_t>(setup.tile_q),
                                      setup.k, setup.tile_q);
    }

    std::vector<float> mask_host;
    std::vector<uint8_t> fully_masked;
    std::vector<float> zero_column;

    for (int64_t batch = 0; batch < setup.batches; ++batch) {
        for (int64_t kv_head = 0; kv_head < setup.hkv; ++kv_head) {
            scrapgpu_tensor_desc v_src{};
            {
                const scrapgpu_status st = make_selected_tile(setup.value, 0, kv_head, batch, setup.dv, setup.k, &v_src);
                if (st != SCRAPGPU_STATUS_OK) {
                    return st;
                }
            }
            {
                const scrapgpu_status st = run_cpy_transpose(stream, v_src, vt_workspace);
                if (st != SCRAPGPU_STATUS_OK) {
                    return st;
                }
            }

            const int64_t q_head_begin = kv_head * setup.heads_per_kv;
            const int64_t q_head_end = q_head_begin + setup.heads_per_kv;
            for (int64_t q_head = q_head_begin; q_head < q_head_end; ++q_head) {
                scrapgpu_tensor_desc k_src{};
                {
                    const scrapgpu_status st = make_selected_tile(setup.key, 0, kv_head, batch, setup.d, setup.k, &k_src);
                    if (st != SCRAPGPU_STATUS_OK) {
                        return st;
                    }
                }
                for (int64_t q0 = 0; q0 < setup.q; q0 += setup.tile_q) {
                    const int64_t q_len = std::min(setup.tile_q, setup.q - q0);
                    scrapgpu_tensor_desc q_src{};
                    scrapgpu_tensor_desc out_dst{};
                    {
                        const scrapgpu_status stq = make_selected_tile(setup.query, q0, q_head, batch, setup.d, q_len, &q_src);
                        if (stq != SCRAPGPU_STATUS_OK) {
                            return stq;
                        }
                        const scrapgpu_status sto = make_selected_tile(setup.output, q0, q_head, batch, setup.dv, q_len, &out_dst);
                        if (sto != SCRAPGPU_STATUS_OK) {
                            return sto;
                        }
                    }

                    const scrapgpu_tensor_desc * mask_ptr = nullptr;
                    scrapgpu_tensor_desc mask_tile{};
                    if (setup.need_mask_workspace) {
                        const scrapgpu_status st = build_mask_tile(setup, stream, q0, q_len, q_head, batch, &mask_host, &fully_masked);
                        if (st != SCRAPGPU_STATUS_OK) {
                            return st;
                        }
                        mask_tile = make_2d_desc(mask_workspace.buffer, mask_workspace.offset, setup.k, q_len);
                        const scrapgpu_status up = scrapgpu_tensor_upload(stream, &mask_tile, mask_host.data(), mask_host.size() * sizeof(float));
                        if (up != SCRAPGPU_STATUS_OK) {
                            return up;
                        }
                        mask_ptr = &mask_tile;
                    }

                    const scrapgpu_tensor_desc scores = make_2d_desc(scores_workspace.buffer, scores_workspace.offset, setup.k, q_len);
                    const scrapgpu_status st_scores = run_mul_mat(stream, k_src, q_src, scores);
                    if (st_scores != SCRAPGPU_STATUS_OK) {
                        return st_scores;
                    }

                    const scrapgpu_status st_softmax = run_softmax(stream, scores, mask_ptr, setup.scale);
                    if (st_softmax != SCRAPGPU_STATUS_OK) {
                        return st_softmax;
                    }

                    if (setup.need_mask_workspace) {
                        const scrapgpu_status st_zero = zero_fully_masked_columns(stream, scores, setup.k, q_len, fully_masked, &zero_column);
                        if (st_zero != SCRAPGPU_STATUS_OK) {
                            return st_zero;
                        }
                    }

                    const scrapgpu_status st_values = run_mul_mat(stream, vt_workspace, scores, out_dst);
                    if (st_values != SCRAPGPU_STATUS_OK) {
                        return st_values;
                    }
                }
            }
        }
    }

    return SCRAPGPU_STATUS_OK;
}

} // namespace

extern "C" {

scrapgpu_status scrapgpu_scaled_dot_product_attention(scrapgpu_device_t * device,
                                                      scrapgpu_stream_t * stream,
                                                      const scrapgpu_attention_params * params) {
    try {
        return execute_attention(device, stream, params);
    } catch (const std::bad_alloc &) {
        return oom("out of host memory while preparing attention");
    } catch (const std::exception & e) {
        return fail(SCRAPGPU_STATUS_BACKEND_ERROR, e.what());
    } catch (...) {
        return fail(SCRAPGPU_STATUS_BACKEND_ERROR, "unknown exception while running attention");
    }
}

}
