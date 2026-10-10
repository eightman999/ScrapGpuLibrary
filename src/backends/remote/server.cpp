// scrapgpu-device-server: registers the local backend(s) chosen with --backend (a comma-separated
// list, default "opencl11"; environment such as GGML_CL11_DEVICES / GGML_CL11_PLATFORMS passes
// through unchanged) and serves every backend vtable call to one client over a Unix domain socket
// (--socket PATH). Bulk data (buffer read / write, tensor upload / download) goes through a POSIX
// shared memory region the server creates on HELLO and grows on demand (protocol.h). The server
// exits when the client disconnects. Dispatch is through the public C API, so the core's own
// checks (same-backend plan/stream, desc validation) apply unchanged in this process.

#include "protocol.h"

#include "scrapgpu/api.h"
#include "scrapgpu/backend.h"
#include "scrapgpu/buffer.h"
#include "scrapgpu/device.h"
#include "scrapgpu/op.h"
#include "scrapgpu/stream.h"
#include "scrapgpu/tensor.h"

#ifdef SCRAPGPU_SERVER_WITH_OPENCL11
#include "scrapgpu/opencl11.h"
#endif
#ifdef SCRAPGPU_SERVER_WITH_VULKAN
#include "scrapgpu/vulkan.h"
#endif

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>

namespace {

using namespace sgr; // the opcodes

int g_client = -1;

int g_shm_fd = -1;
uint8_t * g_shm = nullptr;
size_t g_shm_size = 0;
std::string g_shm_name;
std::string g_socket_path;

std::vector<scrapgpu_device_t *> g_devices;
std::unordered_map<uint64_t, scrapgpu_buffer_t *> g_buffers;
std::unordered_map<uint64_t, scrapgpu_stream_t *> g_streams;
std::unordered_map<uint64_t, scrapgpu_plan_t *>   g_plans;
std::unordered_map<uint64_t, uint64_t>            g_stream_dev;
uint64_t g_next_id = 1;

// ---------------------------------------------------------------------------------------------
// socket IO
// ---------------------------------------------------------------------------------------------

bool write_all(int fd, const void * p, size_t n) {
    const uint8_t * q = static_cast<const uint8_t *>(p);
    while (n > 0) {
#ifdef MSG_NOSIGNAL
        const ssize_t w = send(fd, q, n, MSG_NOSIGNAL);
#else
        const ssize_t w = write(fd, q, n); // SIGPIPE is ignored in this process
#endif
        if (w < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        q += w;
        n -= (size_t) w;
    }
    return true;
}

// false on a clean disconnect (EOF) or a real error: both end the server
bool read_all(int fd, void * p, size_t n) {
    uint8_t * q = static_cast<uint8_t *>(p);
    while (n > 0) {
        const ssize_t r = read(fd, q, n);
        if (r < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (r == 0) {
            return false;
        }
        q += r;
        n -= (size_t) r;
    }
    return true;
}

// the response: the call's status, then the last-error string and the op-specific fields
bool respond(scrapgpu_status status, const std::vector<uint8_t> & body) {
    std::vector<uint8_t> msg;
    msg.reserve(sgr::kHdrSize + 4 + body.size() + 256);
    const char * err = status == SCRAPGPU_STATUS_OK ? "" : scrapgpu_last_error();
    const uint64_t size = 4 + std::strlen(err) + body.size();
    sgr::put_u32(msg, sgr::kMagic);
    sgr::put_u32(msg, (uint32_t) status);
    sgr::put_u64(msg, size);
    sgr::put_str(msg, status == SCRAPGPU_STATUS_OK ? nullptr : err);
    msg.insert(msg.end(), body.begin(), body.end());
    return write_all(g_client, msg.data(), msg.size());
}

bool respond_ok(const std::vector<uint8_t> & body) {
    return respond(SCRAPGPU_STATUS_OK, body);
}

bool fail(scrapgpu_status status, const char * message) {
    scrapgpu_set_last_error(message);
    return respond(status, {});
}

// ---------------------------------------------------------------------------------------------
// shared memory (created on HELLO, grown on SHM_ENSURE and for bulk reads)
// ---------------------------------------------------------------------------------------------

void srv_shm_create() {
    for (int i = 0; g_shm_fd < 0 && i < 8; ++i) {
        char name[64];
        snprintf(name, sizeof(name), "/scrapgpu-srv-%d-%d", (int) getpid(), i);
        g_shm_name = name;
        g_shm_fd   = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    }
    if (g_shm_fd < 0) {
        return; // shm stays disabled: bulk goes through the socket
    }
    if (ftruncate(g_shm_fd, (off_t) sgr::kShmInitial) != 0) {
        close(g_shm_fd);
        g_shm_fd = -1;
        return;
    }
    void * p = mmap(nullptr, sgr::kShmInitial, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (p == MAP_FAILED) {
        close(g_shm_fd);
        g_shm_fd = -1;
        return;
    }
    g_shm      = static_cast<uint8_t *>(p);
    g_shm_size = sgr::kShmInitial;
}

bool srv_shm_resize(size_t size) {
    if (g_shm != nullptr) {
        munmap(g_shm, g_shm_size);
        g_shm = nullptr;
    }
    g_shm_size = 0;
    if (ftruncate(g_shm_fd, (off_t) size) != 0) {
        return false;
    }
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, g_shm_fd, 0);
    if (p == MAP_FAILED) {
        return false;
    }
    g_shm      = static_cast<uint8_t *>(p);
    g_shm_size = size;
    return true;
}

// grows the region to at least `need` (the client sees the new size in the response)
bool srv_shm_ensure(size_t need) {
    if (need <= g_shm_size) {
        return g_shm_size > 0;
    }
    if (g_shm_fd < 0) {
        return false;
    }
    const size_t now = std::max(need, (g_shm_size == 0 ? sgr::kShmInitial : g_shm_size) * 2);
    return srv_shm_resize(now);
}

// ---------------------------------------------------------------------------------------------
// handle translation
// ---------------------------------------------------------------------------------------------

scrapgpu_buffer_t * buffer_of(uint64_t id, bool & bad) {
    if (id == 0) {
        return nullptr;
    }
    const auto it = g_buffers.find(id);
    if (it == g_buffers.end()) {
        bad = true;
        return nullptr;
    }
    return it->second;
}

bool stream_of(uint64_t id, scrapgpu_stream_t ** out) {
    const auto it = g_streams.find(id);
    if (it == g_streams.end()) {
        return false;
    }
    *out = it->second;
    return true;
}

// a decoded op desc; the vectors hold the descs / params the wire carried
struct srv_op {
    scrapgpu_op_desc                   desc = {};
    std::vector<scrapgpu_tensor_desc>  in;
    std::vector<scrapgpu_tensor_desc>  out;
    std::vector<uint8_t>               params;
};

bool get_op(sgr::cur & c, srv_op & o) {
    bool bad = false;
    if (!sgr::get_op_desc(c, o.desc, o.in, o.out, o.params, [&](uint64_t id) { return buffer_of(id, bad); }) || bad) {
        return false;
    }
    sgr::fixup_params(o.desc, o.in, o.params);
    return true;
}

// ---------------------------------------------------------------------------------------------
// request handlers (the last-error string of the public API is forwarded by respond()); the return
// value is the socket write result: false means the client went away and the loop ends
// ---------------------------------------------------------------------------------------------

bool handle(uint32_t op, sgr::cur c) {
    scrapgpu_clear_last_error();
    switch (op) {
        case SGR_OP_HELLO: {
            uint32_t version = 0;
            if (!sgr::get_u32(c, version) || version != sgr::kVersion) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: protocol version mismatch");
            }
            if (g_shm_fd < 0) {
                srv_shm_create();
            }
            std::vector<uint8_t> b;
            sgr::put_u32(b, sgr::kVersion);
            sgr::put_u64(b, g_devices.size());
            sgr::put_u64(b, g_shm_size);
            sgr::put_str(b, g_shm_size > 0 ? g_shm_name.c_str() : nullptr);
            return respond_ok(b);
        }
        case SGR_OP_SHM_ENSURE: {
            uint64_t need = 0;
            if (!sgr::get_u64(c, need)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed SHM_ENSURE");
            }
            if (need > sgr::kMaxShm) {
                return fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "remote: the shared memory request exceeds the protocol limit");
            }
            if (need > 0 && !srv_shm_ensure((size_t) need)) {
                return fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "remote: the shared memory could not be grown");
            }
            std::vector<uint8_t> b;
            sgr::put_u64(b, g_shm_size);
            return respond_ok(b);
        }
        case SGR_OP_GET_DEVICE_COUNT: {
            std::vector<uint8_t> b;
            sgr::put_u64(b, g_devices.size());
            return respond_ok(b);
        }
        case SGR_OP_GET_DEVICE: {
            uint64_t index = 0;
            if (!sgr::get_u64(c, index) || index >= g_devices.size()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: device index out of range");
            }
            std::vector<uint8_t> b;
            sgr::put_u64(b, index); // device ids are the server-side indices
            return respond_ok(b);
        }
        case SGR_OP_DEVICE_GET_INFO: {
            uint64_t id = 0;
            if (!sgr::get_u64(c, id) || id >= g_devices.size()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown device id");
            }
            scrapgpu_device_info info;
            scrapgpu_device_info_init(&info, sizeof(info));
            const scrapgpu_status st = scrapgpu_device_get_info(g_devices[(size_t) id], &info);
            if (st != SCRAPGPU_STATUS_OK) {
                return respond(st, {});
            }
            std::vector<uint8_t> b;
            sgr::put_device_info(b, info);
            return respond_ok(b);
        }
        case SGR_OP_BUFFER_ALLOC: {
            uint64_t dev = 0, size = 0;
            if (!sgr::get_u64(c, dev) || !sgr::get_u64(c, size) || dev >= g_devices.size()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed BUFFER_ALLOC");
            }
            scrapgpu_buffer_t * buf = nullptr;
            const scrapgpu_status st = scrapgpu_buffer_alloc(g_devices[(size_t) dev], (size_t) size, &buf);
            if (st != SCRAPGPU_STATUS_OK) {
                return respond(st, {});
            }
            const uint64_t id = g_next_id++;
            g_buffers[id] = buf;
            std::vector<uint8_t> b;
            sgr::put_u64(b, id);
            return respond_ok(b);
        }
        case SGR_OP_BUFFER_FREE: {
            uint64_t id = 0;
            if (!sgr::get_u64(c, id)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed BUFFER_FREE");
            }
            const auto it = g_buffers.find(id);
            if (it == g_buffers.end()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown buffer id");
            }
            // the backend refuses the free while a live plan references the buffer (audit M-2):
            // the id must stay valid then — erase it only when the buffer was really freed
            const scrapgpu_status st = scrapgpu_buffer_free(it->second);
            if (st == SCRAPGPU_STATUS_OK) {
                g_buffers.erase(it);
            }
            return respond(st, {});
        }
        case SGR_OP_BUFFER_WRITE: {
            uint64_t sid = 0, bid = 0, off = 0, size = 0;
            uint32_t blocking = 0, via = 0;
            if (!sgr::get_u64(c, sid) || !sgr::get_u64(c, bid) || !sgr::get_u64(c, off) ||
                !sgr::get_u64(c, size) || !sgr::get_u32(c, blocking) || !sgr::get_u32(c, via)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed BUFFER_WRITE");
            }
            (void) blocking;
            scrapgpu_stream_t * s = nullptr;
            bool bad = false;
            scrapgpu_buffer_t * b = buffer_of(bid, bad);
            if (!stream_of(sid, &s) || bad || bid == 0) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown stream or buffer id");
            }
            const void * data = nullptr;
            if (size > 0) {
                if (via != 0) {
                    if (g_shm == nullptr || g_shm_size < size) {
                        return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: write data not in the shared memory");
                    }
                    data = g_shm;
                } else {
                    if (c.n < size) {
                        return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: short write payload");
                    }
                    data = c.p;
                }
            }
            // always blocking: the shared memory / request payload is recycled once the response
            // goes out, so the transfer must have consumed it by then
            return respond(scrapgpu_buffer_write(s, b, off, data, (size_t) size, 1), {});
        }
        case SGR_OP_BUFFER_READ: {
            uint64_t sid = 0, bid = 0, off = 0, size = 0;
            uint32_t blocking = 0;
            if (!sgr::get_u64(c, sid) || !sgr::get_u64(c, bid) || !sgr::get_u64(c, off) ||
                !sgr::get_u64(c, size) || !sgr::get_u32(c, blocking)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed BUFFER_READ");
            }
            (void) blocking;
            scrapgpu_stream_t * s = nullptr;
            bool bad = false;
            scrapgpu_buffer_t * b = buffer_of(bid, bad);
            if (!stream_of(sid, &s) || bad || bid == 0) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown stream or buffer id");
            }
            const bool shm = size > 0 && size <= sgr::kMaxShm && srv_shm_ensure((size_t) size);
            void * dst = shm ? g_shm : nullptr;
            std::vector<uint8_t> heap;
            if (!shm && size > 0) {
                // bulk data is expected to travel through the shared memory; the inline fallback
                // stays bounded by the message limit (audit H-2)
                if (size > sgr::kMaxPayload) {
                    return fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "remote: read exceeds the inline limit and the shared memory is unavailable");
                }
                heap.resize((size_t) size);
                dst = heap.data();
            }
            // always blocking: the data ships with the response, so it must have landed by then
            const scrapgpu_status st = scrapgpu_buffer_read(s, b, off, dst, (size_t) size, 1);
            if (st != SCRAPGPU_STATUS_OK) {
                return respond(st, {});
            }
            std::vector<uint8_t> resp;
            sgr::put_u64(resp, g_shm_size);
            sgr::put_u32(resp, shm ? 1u : 0u);
            if (!shm && size > 0) {
                sgr::put_bytes(resp, heap.data(), (size_t) size);
            }
            return respond_ok(resp);
        }
        case SGR_OP_BUFFER_COPY: {
            uint64_t sid = 0, d = 0, doff = 0, src = 0, soff = 0, size = 0;
            uint32_t blocking = 0;
            if (!sgr::get_u64(c, sid) || !sgr::get_u64(c, d) || !sgr::get_u64(c, doff) ||
                !sgr::get_u64(c, src) || !sgr::get_u64(c, soff) || !sgr::get_u64(c, size) ||
                !sgr::get_u32(c, blocking)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed BUFFER_COPY");
            }
            scrapgpu_stream_t * s = nullptr;
            bool bad = false;
            scrapgpu_buffer_t * db = buffer_of(d, bad);
            scrapgpu_buffer_t * sb = buffer_of(src, bad);
            if (!stream_of(sid, &s) || bad || d == 0 || src == 0) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown stream or buffer id");
            }
            return respond(scrapgpu_buffer_copy(s, db, doff, sb, soff, (size_t) size, (int) blocking), {});
        }
        case SGR_OP_BUFFER_FILL: {
            uint64_t sid = 0, bid = 0, off = 0, size = 0;
            uint32_t byte = 0, blocking = 0;
            if (!sgr::get_u64(c, sid) || !sgr::get_u64(c, bid) || !sgr::get_u64(c, off) ||
                !sgr::get_u64(c, size) || !sgr::get_u32(c, byte) || !sgr::get_u32(c, blocking)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed BUFFER_FILL");
            }
            scrapgpu_stream_t * s = nullptr;
            bool bad = false;
            scrapgpu_buffer_t * b = buffer_of(bid, bad);
            if (!stream_of(sid, &s) || bad || bid == 0) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown stream or buffer id");
            }
            return respond(scrapgpu_buffer_fill(s, b, off, (size_t) size, (uint8_t) byte, (int) blocking), {});
        }
        case SGR_OP_STREAM_CREATE: {
            uint64_t dev = 0;
            if (!sgr::get_u64(c, dev) || dev >= g_devices.size()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed STREAM_CREATE");
            }
            scrapgpu_stream_t * s = nullptr;
            const scrapgpu_status st = scrapgpu_stream_create(g_devices[(size_t) dev], &s);
            if (st != SCRAPGPU_STATUS_OK) {
                return respond(st, {});
            }
            const uint64_t id = g_next_id++;
            g_streams[id]    = s;
            g_stream_dev[id] = dev;
            std::vector<uint8_t> b;
            sgr::put_u64(b, id);
            return respond_ok(b);
        }
        case SGR_OP_STREAM_FREE: {
            uint64_t id = 0;
            if (!sgr::get_u64(c, id)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed STREAM_FREE");
            }
            const auto it = g_streams.find(id);
            if (it == g_streams.end()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown stream id");
            }
            const scrapgpu_status st = scrapgpu_stream_free(it->second);
            g_streams.erase(it);
            g_stream_dev.erase(id);
            return respond(st, {});
        }
        case SGR_OP_STREAM_SYNC: {
            uint64_t id = 0;
            scrapgpu_stream_t * s = nullptr;
            if (!sgr::get_u64(c, id) || !stream_of(id, &s)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown stream id");
            }
            return respond(scrapgpu_stream_sync(s), {});
        }
        case SGR_OP_STREAM_DEVICE: {
            uint64_t id = 0;
            if (!sgr::get_u64(c, id)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed STREAM_DEVICE");
            }
            const auto it = g_stream_dev.find(id);
            if (it == g_stream_dev.end()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown stream id");
            }
            std::vector<uint8_t> b;
            sgr::put_u64(b, it->second);
            return respond_ok(b);
        }
        case SGR_OP_OP_SUPPORTED: {
            uint64_t dev = 0;
            srv_op o;
            if (!sgr::get_u64(c, dev) || dev >= g_devices.size() || !get_op(c, o)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed OP_SUPPORTED");
            }
            const char * reason = nullptr;
            const scrapgpu_status st = scrapgpu_op_supported(g_devices[(size_t) dev], &o.desc, &reason);
            std::vector<uint8_t> b;
            sgr::put_str(b, reason);
            return respond(st, b);
        }
        case SGR_OP_OP_EXECUTE: {
            uint64_t sid = 0;
            srv_op o;
            scrapgpu_stream_t * s = nullptr;
            if (!sgr::get_u64(c, sid) || !stream_of(sid, &s) || !get_op(c, o)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed OP_EXECUTE");
            }
            return respond(scrapgpu_op_execute(s, &o.desc), {});
        }
        case SGR_OP_PLAN_BUILD: {
            uint64_t dev = 0, n = 0;
            if (!sgr::get_u64(c, dev) || !sgr::get_u64(c, n) || dev >= g_devices.size() || n > 4096) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed PLAN_BUILD");
            }
            std::vector<scrapgpu_plan_node> nodes((size_t) n);
            std::vector<srv_op> storage((size_t) n);
            for (size_t i = 0; i < (size_t) n; ++i) {
                uint32_t use = 0;
                if (!sgr::get_u32(c, use) || !get_op(c, storage[i])) {
                    return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed plan node");
                }
                nodes[i].struct_size      = sizeof(scrapgpu_plan_node);
                nodes[i].op               = storage[i].desc;
                nodes[i].output_use_count = use;
            }
            scrapgpu_plan_desc pd = {};
            pd.struct_size = sizeof(pd);
            pd.nodes       = nodes.empty() ? nullptr : nodes.data();
            pd.node_count  = (size_t) n;
            scrapgpu_plan_t * plan = nullptr;
            const scrapgpu_status st = scrapgpu_plan_build(g_devices[(size_t) dev], &pd, &plan);
            if (st != SCRAPGPU_STATUS_OK) {
                return respond(st, {});
            }
            const uint64_t id = g_next_id++;
            g_plans[id] = plan;
            std::vector<uint8_t> b;
            sgr::put_u64(b, id);
            return respond_ok(b);
        }
        case SGR_OP_PLAN_EXECUTE: {
            uint64_t sid = 0, pid = 0;
            scrapgpu_stream_t * s = nullptr;
            if (!sgr::get_u64(c, sid) || !sgr::get_u64(c, pid) || !stream_of(sid, &s)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed PLAN_EXECUTE");
            }
            const auto it = g_plans.find(pid);
            if (it == g_plans.end()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown plan id");
            }
            return respond(scrapgpu_plan_execute(s, it->second), {});
        }
        case SGR_OP_PLAN_FREE: {
            uint64_t pid = 0;
            if (!sgr::get_u64(c, pid)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed PLAN_FREE");
            }
            const auto it = g_plans.find(pid);
            if (it == g_plans.end()) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown plan id");
            }
            const scrapgpu_status st = scrapgpu_plan_free(it->second);
            g_plans.erase(it);
            return respond(st, {});
        }
        case SGR_OP_TENSOR_UPLOAD:
        case SGR_OP_TENSOR_DOWNLOAD: {
            uint64_t sid = 0, size = 0, bid = 0;
            scrapgpu_tensor_desc td;
            scrapgpu_stream_t * s = nullptr;
            if (!sgr::get_u64(c, sid) || !sgr::get_tensor_desc(c, td, bid) || !sgr::get_u64(c, size) ||
                !stream_of(sid, &s)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed tensor transfer");
            }
            bool bad = false;
            td.buffer = buffer_of(bid, bad);
            if (bad || (bid != 0 && td.buffer == nullptr)) {
                return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: unknown buffer id in tensor desc");
            }
            if (op == SGR_OP_TENSOR_UPLOAD) {
                uint32_t via = 0;
                if (!sgr::get_u32(c, via)) {
                    return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: malformed TENSOR_UPLOAD");
                }
                const void * data = nullptr;
                if (size > 0) {
                    if (via != 0) {
                        if (g_shm == nullptr || g_shm_size < size) {
                            return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: upload data not in the shared memory");
                        }
                        data = g_shm;
                    } else {
                        if (c.n < size) {
                            return fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: short upload payload");
                        }
                        data = c.p;
                    }
                }
                // tensor_upload is synchronous wrt the host pointer by contract (the R32 packing of
                // the backend included), so the shared memory / payload can be recycled on return
                return respond(scrapgpu_tensor_upload(s, &td, data, (size_t) size), {});
            }
            const bool shm = size > 0 && size <= sgr::kMaxShm && srv_shm_ensure((size_t) size);
            void * dst = shm ? g_shm : nullptr;
            std::vector<uint8_t> heap;
            if (!shm && size > 0) {
                if (size > sgr::kMaxPayload) {
                    return fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "remote: download exceeds the inline limit and the shared memory is unavailable");
                }
                heap.resize((size_t) size);
                dst = heap.data();
            }
            const scrapgpu_status st = scrapgpu_tensor_download(s, &td, dst, (size_t) size);
            if (st != SCRAPGPU_STATUS_OK) {
                return respond(st, {});
            }
            std::vector<uint8_t> resp;
            sgr::put_u64(resp, g_shm_size);
            sgr::put_u32(resp, shm ? 1u : 0u);
            if (!shm && size > 0) {
                sgr::put_bytes(resp, heap.data(), (size_t) size);
            }
            return respond_ok(resp);
        }
        default:
            return fail(SCRAPGPU_STATUS_UNSUPPORTED, "remote: unknown request");
    }
}

// ---------------------------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------------------------

void usage(const char * argv0) {
    std::fprintf(stderr, "usage: %s --socket PATH [--backend opencl11[,vulkan]]\n", argv0);
}

int srv_register(const std::string & name) {
    if (name == "opencl11") {
#ifdef SCRAPGPU_SERVER_WITH_OPENCL11
        return scrapgpu_opencl11_register() == SCRAPGPU_STATUS_OK ? 0 : 1;
#else
        std::fprintf(stderr, "scrapgpu-device-server: built without the opencl11 backend\n");
        return 1;
#endif
    }
    if (name == "vulkan") {
#ifdef SCRAPGPU_SERVER_WITH_VULKAN
        return scrapgpu_vulkan_register() == SCRAPGPU_STATUS_OK ? 0 : 1;
#else
        std::fprintf(stderr, "scrapgpu-device-server: built without the vulkan backend\n");
        return 1;
#endif
    }
    std::fprintf(stderr, "scrapgpu-device-server: unknown backend '%s'\n", name.c_str());
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    std::string backends = "opencl11";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--socket" && i + 1 < argc) {
            g_socket_path = argv[++i];
        } else if (a == "--backend" && i + 1 < argc) {
            backends = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (g_socket_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (g_socket_path.size() >= sizeof(((sockaddr_un *) nullptr)->sun_path)) {
        std::fprintf(stderr, "scrapgpu-device-server: socket path is too long\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);

    size_t pos = 0;
    int nbackends = 0;
    while (pos <= backends.size()) {
        const size_t comma = backends.find(',', pos);
        const std::string name = backends.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!name.empty() && srv_register(name) != 0) {
            return 1;
        }
        if (!name.empty()) {
            ++nbackends;
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    if (nbackends == 0) {
        std::fprintf(stderr, "scrapgpu-device-server: no backend named\n");
        return 1;
    }

    const size_t ndev = scrapgpu_get_device_count();
    for (size_t i = 0; i < ndev; ++i) {
        scrapgpu_device_t * d = nullptr;
        if (scrapgpu_get_device(i, &d) != SCRAPGPU_STATUS_OK || d == nullptr) {
            break;
        }
        g_devices.push_back(d);
    }
    std::fprintf(stderr, "scrapgpu-device-server: %s, %zu device(s), socket %s\n",
                 backends.c_str(), g_devices.size(), g_socket_path.c_str());

    unlink(g_socket_path.c_str()); // a leftover of a previous run
    const int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::fprintf(stderr, "scrapgpu-device-server: socket: %s\n", strerror(errno));
        return 1;
    }
    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    std::strcpy(addr.sun_path, g_socket_path.c_str());
    if (bind(listen_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || chmod(g_socket_path.c_str(), 0600) != 0 ||
        listen(listen_fd, 1) != 0) {
        std::fprintf(stderr, "scrapgpu-device-server: bind/chmod/listen %s: %s\n", g_socket_path.c_str(), strerror(errno));
        return 1;
    }
    // do not linger forever when the spawning client died before connecting
    pollfd pfd = { listen_fd, POLLIN, 0 };
    if (poll(&pfd, 1, 300000) <= 0) {
        std::fprintf(stderr, "scrapgpu-device-server: no client connected, exiting\n");
        return 1;
    }
    g_client = accept(listen_fd, nullptr, nullptr);
    close(listen_fd);
    if (g_client < 0) {
        std::fprintf(stderr, "scrapgpu-device-server: accept: %s\n", strerror(errno));
        return 1;
    }
    // the GPU memory of this process is served to one client of the same user only
    {
        uid_t peer_uid = (uid_t) -1;
#if defined(__APPLE__) || defined(__FreeBSD__)
        gid_t peer_gid = 0;
        const bool known = getpeereid(g_client, &peer_uid, &peer_gid) == 0;
#else
        ucred cred = {};
        socklen_t len = sizeof(cred);
        const bool known = getsockopt(g_client, SOL_SOCKET, SO_PEERCRED, &cred, &len) == 0;
        peer_uid = cred.uid;
#endif
        if (!known || peer_uid != getuid()) {
            std::fprintf(stderr, "scrapgpu-device-server: refusing a client of another user\n");
            close(g_client);
            return 1;
        }
    }
#ifdef SO_NOSIGPIPE
    {
        int one = 1;
        setsockopt(g_client, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
    }
#endif

    for (;;) {
        uint8_t hdr[sgr::kHdrSize];
        if (!read_all(g_client, hdr, sizeof(hdr))) {
            break; // the client disconnected
        }
        sgr::cur hc{hdr, sgr::kHdrSize};
        uint32_t magic = 0, op = 0;
        uint64_t size = 0;
        if (!sgr::get_u32(hc, magic) || magic != sgr::kMagic || !sgr::get_u32(hc, op) || !sgr::get_u64(hc, size)) {
            break;
        }
        if (size > (uint64_t) sgr::kMaxPayload) {
            // an oversized message would let a client make the server allocate without bound
            // (audit H-2): answer with an error and close the connection cleanly — the payload
            // cannot be drained at a bounded cost
            fail(SCRAPGPU_STATUS_INVALID_ARGUMENT, "remote: request payload exceeds the protocol limit");
            break;
        }
        std::vector<uint8_t> payload;
        try {
            payload.resize((size_t) size);
        } catch (const std::bad_alloc &) {
            fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "remote: out of memory for the request payload");
            break;
        }
        if (size > 0 && !read_all(g_client, payload.data(), (size_t) size)) {
            break;
        }
        sgr::cur body{payload.empty() ? nullptr : payload.data(), payload.size()};
        bool alive = true;
        try {
            alive = handle(op, body);
        } catch (const std::bad_alloc &) {
            // a handler ran out of memory (a response or a decoded plan): report it and keep serving
            alive = fail(SCRAPGPU_STATUS_OUT_OF_MEMORY, "remote: out of memory");
        }
        if (!alive) {
            break;
        }
    }

    close(g_client);
    if (g_shm != nullptr) {
        munmap(g_shm, g_shm_size);
    }
    if (g_shm_fd >= 0) {
        close(g_shm_fd);
    }
    if (!g_shm_name.empty()) {
        shm_unlink(g_shm_name.c_str());
    }
    if (!g_socket_path.empty()) {
        unlink(g_socket_path.c_str());
    }
    return 0;
}
