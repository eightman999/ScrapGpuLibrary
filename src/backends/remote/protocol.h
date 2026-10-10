#ifndef SCRAPGPU_REMOTE_PROTOCOL_H
#define SCRAPGPU_REMOTE_PROTOCOL_H

// The wire protocol between the "remote" proxy backend (client.cpp) and scrapgpu-device-server
// (server.cpp). One request / one response per backend call on a Unix domain socket; bulk data
// (buffer read / write, tensor upload / download) goes through a POSIX shared memory region whose
// name the server hands out in the HELLO reply and that grows on demand (SGR_OP_SHM_ENSURE). The
// socket stays inside one machine and both ends are built from the same headers, so integers are
// serialized little-endian by the helpers below and structs on the wire carry no pointers.
//
// Message: 16-byte header {u32 magic, u32 code, u64 payload_size} + payload.
//   request  code = sgr_op;   response code = scrapgpu_status of the call.
// Every response payload starts with the error string the server side set
// (u32 len + bytes; scrapgpu_last_error() after clearing it before the call), then the
// op-specific fields. Handle values are server-side ids (0 = none).

#include "scrapgpu/backend.h"
#include "scrapgpu/op.h"
#include "scrapgpu/tensor.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace sgr {

constexpr uint32_t kMagic   = 0x53475052u; // 'SGRP'
constexpr uint32_t kVersion = 1u;
constexpr size_t   kHdrSize = 16;
constexpr size_t   kShmInitial = 1u << 20;  // 1 MiB; grown on demand
constexpr size_t   kMaxTensors = 32;        // sanity bound on serialized input/output counts
// the biggest single message on the wire (audit H-2): bulk data goes through the shared memory
// region, so nothing needs a bigger inline payload; both ends check it before allocating
constexpr size_t   kMaxPayload = 64u << 20; // 64 MiB
// sanity bound on the shared memory a peer may ask for (no buffer can be bigger than this)
constexpr uint64_t kMaxShm     = 4ull << 30; // 4 GiB

// 0xFFFFFFFF marks a NULL string / no value on the wire
constexpr uint32_t kNullStr = 0xFFFFFFFFu;

enum sgr_op : uint32_t {
    SGR_OP_HELLO            = 1,  // req {u32 version}; resp {u32 version, u64 shm_size, str shm_name}
    SGR_OP_SHM_ENSURE       = 2,  // req {u64 needed}; resp {u64 shm_size}
    SGR_OP_GET_DEVICE_COUNT = 3,  // resp {u64 count}
    SGR_OP_GET_DEVICE       = 4,  // req {u64 index}; resp {u64 device_id}
    SGR_OP_DEVICE_GET_INFO  = 5,  // req {u64 device}; resp {device_info wire}
    SGR_OP_BUFFER_ALLOC     = 10, // req {u64 device, u64 size}; resp {u64 buffer}
    SGR_OP_BUFFER_FREE      = 11, // req {u64 buffer}
    SGR_OP_BUFFER_WRITE     = 12, // req {u64 stream, u64 buffer, u64 offset, u64 size, u32 blocking, u32 via_shm, [data]}
    SGR_OP_BUFFER_READ      = 13, // req {u64 stream, u64 buffer, u64 offset, u64 size, u32 blocking};
                                  // resp {u64 shm_size_now, u32 via_shm, [data]}
    SGR_OP_BUFFER_COPY      = 14, // req {u64 stream, u64 dst, u64 dst_off, u64 src, u64 src_off, u64 size, u32 blocking}
    SGR_OP_BUFFER_FILL      = 15, // req {u64 stream, u64 buffer, u64 offset, u64 size, u32 byte, u32 blocking}
    SGR_OP_STREAM_CREATE    = 20, // req {u64 device}; resp {u64 stream}
    SGR_OP_STREAM_FREE      = 21, // req {u64 stream}
    SGR_OP_STREAM_SYNC      = 22, // req {u64 stream}
    SGR_OP_STREAM_DEVICE    = 23, // req {u64 stream}; resp {u64 device}
    SGR_OP_OP_SUPPORTED     = 30, // req {u64 device, op_desc wire}; resp {str reason}
    SGR_OP_OP_EXECUTE       = 31, // req {u64 stream, op_desc wire}
    SGR_OP_PLAN_BUILD       = 32, // req {u64 device, plan wire}; resp {u64 plan}
    SGR_OP_PLAN_EXECUTE     = 33, // req {u64 stream, u64 plan}
    SGR_OP_PLAN_FREE        = 34, // req {u64 plan}
    SGR_OP_TENSOR_UPLOAD    = 40, // req {u64 stream, tensor_desc wire, u64 size, u32 via_shm, [data]}
    SGR_OP_TENSOR_DOWNLOAD  = 41, // req {u64 stream, tensor_desc wire, u64 size};
                                  // resp {u64 shm_size_now, u32 via_shm, [data]}
};

// ---- little-endian put / get -------------------------------------------------

inline void put_u32(std::vector<uint8_t> & b, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b.push_back((uint8_t) (v >> (8 * i)));
    }
}

inline void put_u64(std::vector<uint8_t> & b, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        b.push_back((uint8_t) (v >> (8 * i)));
    }
}

inline void put_i64(std::vector<uint8_t> & b, int64_t v) {
    put_u64(b, (uint64_t) v);
}

inline void put_str(std::vector<uint8_t> & b, const char * s) {
    if (s == nullptr) {
        put_u32(b, kNullStr);
        return;
    }
    const size_t n = std::strlen(s);
    put_u32(b, (uint32_t) n);
    b.insert(b.end(), s, s + n);
}

inline void put_bytes(std::vector<uint8_t> & b, const void * p, size_t n) {
    const uint8_t * q = static_cast<const uint8_t *>(p);
    b.insert(b.end(), q, q + n);
}

struct cur {
    const uint8_t * p;
    size_t          n;
    bool            ok = true;
};

inline bool get_u32(cur & c, uint32_t & v) {
    if (c.n < 4) {
        c.ok = false;
        return false;
    }
    v = 0;
    for (int i = 0; i < 4; ++i) {
        v |= (uint32_t) c.p[i] << (8 * i);
    }
    c.p += 4;
    c.n -= 4;
    return true;
}

inline bool get_u64(cur & c, uint64_t & v) {
    if (c.n < 8) {
        c.ok = false;
        return false;
    }
    v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (uint64_t) c.p[i] << (8 * i);
    }
    c.p += 8;
    c.n -= 8;
    return true;
}

inline bool get_i64(cur & c, int64_t & v) {
    uint64_t u = 0;
    if (!get_u64(c, u)) {
        return false;
    }
    v = (int64_t) u;
    return true;
}

// decodes a string into out (empty for NULL); `present` reports whether the wire value was non-NULL
inline bool get_str(cur & c, std::string & out, bool * present = nullptr) {
    uint32_t len = 0;
    if (!get_u32(c, len)) {
        return false;
    }
    if (len == kNullStr) {
        out.clear();
        if (present != nullptr) {
            *present = false;
        }
        return true;
    }
    if ((uint64_t) len > c.n) {
        c.ok = false;
        return false;
    }
    out.assign(reinterpret_cast<const char *>(c.p), len);
    c.p += len;
    c.n -= len;
    if (present != nullptr) {
        *present = true;
    }
    return true;
}

inline bool get_bytes(cur & c, void * dst, size_t n) {
    if (c.n < n) {
        c.ok = false;
        return false;
    }
    std::memcpy(dst, c.p, n);
    c.p += n;
    c.n -= n;
    return true;
}

// ---- tensor / op descs ---------------------------------------------------------
//
// tensor desc wire: {u32 dtype, u32 layout, i64 ne[4], u64 nb[4], u64 buffer_id, u64 offset}
// (a smaller struct_size on the caller side keeps the defaults of the tail first; buffer_id is
// the server-side id, 0 for no buffer)
// op desc wire: {u32 op, u32 n_in, u32 n_out, u32 params_size, in[n_in], out[n_out], params bytes}
// plan wire:    {u64 node_count, node * {u32 output_use_count, op_desc wire}}
//
// The desc pointers inside the params structs (ROPE freq_factors / n_offs, SOFT_MAX mask) travel
// as their raw bytes; only NULL-vs-set survives the wire. The receiving side rewires them to the
// decoded inputs[] entries (same rule as the plan build of the backends: non-NULL means "the
// matching inputs[] slot").

inline void put_tensor_desc(std::vector<uint8_t> & b, const scrapgpu_tensor_desc & in, uint64_t buffer_id) {
    scrapgpu_tensor_desc t = {};
    if (in.struct_size >= sizeof(size_t)) {
        std::memcpy(&t, &in, in.struct_size < sizeof(t) ? in.struct_size : sizeof(t));
    }
    put_u32(b, (uint32_t) t.dtype);
    put_u32(b, (uint32_t) t.layout);
    for (int i = 0; i < 4; ++i) {
        put_i64(b, t.ne[i]);
    }
    for (int i = 0; i < 4; ++i) {
        put_u64(b, t.nb[i]);
    }
    put_u64(b, buffer_id);
    put_u64(b, t.offset);
}

// decodes a wire tensor desc; *buffer_id is the raw wire value for the caller to resolve
inline bool get_tensor_desc(cur & c, scrapgpu_tensor_desc & out, uint64_t & buffer_id) {
    uint32_t v = 0;
    out = {};
    out.struct_size = sizeof(out);
    if (!get_u32(c, v)) {
        return false;
    }
    out.dtype = (scrapgpu_dtype) v;
    if (!get_u32(c, v)) {
        return false;
    }
    out.layout = (scrapgpu_layout) v;
    for (int i = 0; i < 4; ++i) {
        if (!get_i64(c, out.ne[i])) {
            return false;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (!get_u64(c, out.nb[i])) {
            return false;
        }
    }
    if (!get_u64(c, buffer_id)) {
        return false;
    }
    return get_u64(c, out.offset);
}

// buffer_of: translates the buffer pointer of a desc into the wire id (0 when NULL; the callback
// returns false + sets fail when the pointer does not belong to this connection)
template <typename F>
bool put_op_desc(std::vector<uint8_t> & b, const scrapgpu_op_desc & in, F buffer_of) {
    if (in.struct_size < offsetof(scrapgpu_op_desc, params_size) + sizeof(in.params_size) ||
        in.input_count > kMaxTensors || in.output_count > kMaxTensors ||
        in.params_size > (1u << 20)) {
        return false;
    }
    const uint32_t n_in  = (uint32_t) in.input_count;
    const uint32_t n_out = (uint32_t) in.output_count;
    const uint32_t psize = (uint32_t) in.params_size;
    if ((n_in > 0 && in.inputs == nullptr) || (n_out > 0 && in.outputs == nullptr) ||
        (psize > 0 && in.params == nullptr)) {
        return false;
    }
    put_u32(b, (uint32_t) in.op);
    put_u32(b, n_in);
    put_u32(b, n_out);
    put_u32(b, psize);
    for (uint32_t i = 0; i < n_in; ++i) {
        put_tensor_desc(b, in.inputs[i], buffer_of(in.inputs[i].buffer));
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        put_tensor_desc(b, in.outputs[i], buffer_of(in.outputs[i].buffer));
    }
    if (psize > 0 && in.params != nullptr) {
        put_bytes(b, in.params, psize);
    }
    return true;
}

// the decoded desc: inputs / outputs / params live in the vectors (fixed pointers into them);
// buffer_of resolves a wire buffer id to a scrapgpu_buffer_t *
template <typename F>
bool get_op_desc(cur & c, scrapgpu_op_desc & out, std::vector<scrapgpu_tensor_desc> & in_v,
                 std::vector<scrapgpu_tensor_desc> & out_v, std::vector<uint8_t> & params, F buffer_of) {
    uint32_t op = 0, n_in = 0, n_out = 0, psize = 0;
    if (!get_u32(c, op) || !get_u32(c, n_in) || !get_u32(c, n_out) || !get_u32(c, psize)) {
        return false;
    }
    if (n_in > kMaxTensors || n_out > kMaxTensors || psize > (1u << 20)) {
        c.ok = false;
        return false;
    }
    in_v.resize(n_in);
    out_v.resize(n_out);
    for (uint32_t i = 0; i < n_in; ++i) {
        uint64_t id = 0;
        if (!get_tensor_desc(c, in_v[i], id)) {
            return false;
        }
        in_v[i].buffer = buffer_of(id);
    }
    for (uint32_t i = 0; i < n_out; ++i) {
        uint64_t id = 0;
        if (!get_tensor_desc(c, out_v[i], id)) {
            return false;
        }
        out_v[i].buffer = buffer_of(id);
    }
    params.resize(psize);
    if (psize > 0 && !get_bytes(c, params.data(), psize)) {
        return false;
    }
    out = {};
    out.struct_size  = sizeof(out);
    out.op           = (scrapgpu_op) op;
    out.inputs       = in_v.empty() ? nullptr : in_v.data();
    out.input_count  = n_in;
    out.outputs      = out_v.empty() ? nullptr : out_v.data();
    out.output_count = n_out;
    out.params       = params.empty() ? nullptr : params.data();
    out.params_size  = psize;
    return true;
}

// rewrites the tensor-desc pointers inside the params of a decoded op: they survived the wire only
// as NULL-vs-set and must point at the decoded inputs[] entries (as the backend plan build does)
inline void fixup_params(scrapgpu_op_desc & d, std::vector<scrapgpu_tensor_desc> & in_v,
                         std::vector<uint8_t> & params) {
    if (d.op == SCRAPGPU_OP_ROPE && params.size() >= sizeof(scrapgpu_rope_params)) {
        scrapgpu_rope_params p;
        std::memcpy(&p, params.data(), sizeof(p));
        p.freq_factors = p.freq_factors != nullptr && in_v.size() > 2 ? &in_v[2] : nullptr;
        p.n_offs       = p.n_offs       != nullptr && in_v.size() > 3 ? &in_v[3] : nullptr;
        std::memcpy(params.data(), &p, sizeof(p));
    }
    if (d.op == SCRAPGPU_OP_SOFT_MAX && params.size() >= sizeof(scrapgpu_soft_max_params)) {
        scrapgpu_soft_max_params p;
        std::memcpy(&p, params.data(), sizeof(p));
        p.mask = p.mask != nullptr && in_v.size() > 1 ? &in_v[1] : nullptr;
        std::memcpy(params.data(), &p, sizeof(p));
    }
}

// ---- device info -----------------------------------------------------------------
// fixed fields in declaration order, then the ten strings.

inline void put_device_info(std::vector<uint8_t> & b, const scrapgpu_device_info & i) {
    put_u32(b, (uint32_t) i.generation);
    put_u64(b, i.total_memory_bytes);
    put_u64(b, i.max_allocation_reported_bytes);
    put_u64(b, i.max_allocation_used_bytes);
    put_u64(b, i.max_work_group_size);
    put_u64(b, i.local_memory_bytes);
    put_u32(b, i.compute_units);
    put_u32(b, i.warp_or_wave_width);
    put_u64(b, i.quirks);
    put_u64(b, i.probed_quirks);
    put_str(b, i.name);
    put_str(b, i.vendor);
    put_str(b, i.platform_name);
    put_str(b, i.driver_version);
    put_str(b, i.backend_name);
    put_str(b, i.reported_api_version);
    put_str(b, i.used_api_version);
    put_str(b, i.generation_reason);
    put_str(b, i.tuning_profile_name);
    put_str(b, i.tuning_overrides);
}

inline bool get_device_info(cur & c, scrapgpu_device_info & i, std::string storage[10]) {
    uint32_t v = 0;
    if (!get_u32(c, v)) {
        return false;
    }
    i.generation = (scrapgpu_device_generation) v;
    if (!get_u64(c, i.total_memory_bytes) || !get_u64(c, i.max_allocation_reported_bytes) ||
        !get_u64(c, i.max_allocation_used_bytes) || !get_u64(c, i.max_work_group_size) ||
        !get_u64(c, i.local_memory_bytes)) {
        return false;
    }
    if (!get_u32(c, i.compute_units) || !get_u32(c, i.warp_or_wave_width)) {
        return false;
    }
    if (!get_u64(c, i.quirks) || !get_u64(c, i.probed_quirks)) {
        return false;
    }
    bool present = false;
    const char ** fields[10] = { &i.name, &i.vendor, &i.platform_name, &i.driver_version,
                                 &i.backend_name, &i.reported_api_version, &i.used_api_version,
                                 &i.generation_reason, &i.tuning_profile_name, &i.tuning_overrides };
    for (int k = 0; k < 10; ++k) {
        if (!get_str(c, storage[k], &present)) {
            return false;
        }
        *fields[k] = present ? storage[k].c_str() : nullptr;
    }
    return true;
}

} // namespace sgr

#endif
