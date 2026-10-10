// The "remote" proxy backend: the devices of a scrapgpu-device-server process appear next to the
// local ones (scrapgpu_get_device, names prefixed with "remote:") and every vtable call is a
// request / response pair on a Unix domain socket. Bulk data (buffer read / write, tensor upload /
// download) goes through a shared memory region instead of the socket when it fits (grown on
// demand; wire format in protocol.h). Handles are server-side ids carried by the proxy objects
// below. POSIX only: socketpair-free spawn is fork + exec of the server found next to the calling
// executable (SCRAPGPU_DEVICE_SERVER overrides the path).

#include "protocol.h"

#include "scrapgpu/backend.h"
#include "scrapgpu/remote.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unistd.h>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace {

using namespace sgr; // the opcodes

constexpr uint64_t kProxyMagic = 0x5347524d545631ull; // 'SGRMTV1' tag on every proxy object

struct rmt_client;

struct rmt_device {
    scrapgpu_object_header hdr;
    uint64_t               magic = kProxyMagic;
    rmt_client *           owner = nullptr;
    uint64_t               id    = 0;
    // device_get_info scratch: the info strings point into these members (stable until the next call)
    std::string            info_str[10];
    std::string            prefixed_name;
};

struct rmt_buffer {
    scrapgpu_object_header hdr;
    uint64_t               magic = kProxyMagic;
    rmt_client *           owner = nullptr;
    uint64_t               id    = 0;
};

struct rmt_stream {
    scrapgpu_object_header hdr;
    uint64_t               magic = kProxyMagic;
    rmt_client *           owner = nullptr;
    uint64_t               id    = 0;
    rmt_device *           dev   = nullptr;
};

struct rmt_plan {
    scrapgpu_object_header hdr;
    uint64_t               magic = kProxyMagic;
    rmt_client *           owner = nullptr;
    uint64_t               id    = 0;
};

struct rmt_client {
    scrapgpu_backend_vtable vt;              // per-connection vtable: the objects point at it
    int                     fd        = -1;  // the socket
    int                     shm_fd    = -1;
    uint8_t *               shm       = nullptr;
    size_t                  shm_size  = 0;
    pid_t                   child     = -1;  // spawned server (waitpid'd on close); -1 for connect
    bool                    dead      = false;
    std::string             socket_path;     // unlinked on close (spawned sockets only)
    std::mutex              mtx;
    size_t                  device_count = 0;
    std::vector<std::unique_ptr<rmt_device>> devices;
    std::string             reason_scratch;  // the op_supported reason of the last call
};

// ---------------------------------------------------------------------------------------------
// status helpers
// ---------------------------------------------------------------------------------------------

scrapgpu_status rmt_invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

scrapgpu_status rmt_lost(rmt_client * c, const char * what) {
    c->dead = true;
    scrapgpu_set_last_error(what);
    return SCRAPGPU_STATUS_DEVICE_LOST;
}

// ---------------------------------------------------------------------------------------------
// socket IO
// ---------------------------------------------------------------------------------------------

bool write_all(int fd, const void * p, size_t n) {
    const uint8_t * q = static_cast<const uint8_t *>(p);
    while (n > 0) {
#ifdef MSG_NOSIGNAL
        const ssize_t w = send(fd, q, n, MSG_NOSIGNAL);
#else
        const ssize_t w = write(fd, q, n); // the socket has SO_NOSIGPIPE
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
            return false; // the server closed the connection
        }
        q += r;
        n -= (size_t) r;
    }
    return true;
}

void hdr_put(uint8_t hdr[sgr::kHdrSize], uint32_t code, uint64_t size) {
    std::vector<uint8_t> b;
    b.reserve(sgr::kHdrSize);
    sgr::put_u32(b, sgr::kMagic);
    sgr::put_u32(b, code);
    sgr::put_u64(b, size);
    std::memcpy(hdr, b.data(), sgr::kHdrSize);
}

bool hdr_get(const uint8_t hdr[sgr::kHdrSize], uint32_t & code, uint64_t & size) {
    sgr::cur c{hdr, sgr::kHdrSize};
    uint32_t magic = 0;
    if (!sgr::get_u32(c, magic) || magic != sgr::kMagic || !sgr::get_u32(c, code) || !sgr::get_u64(c, size)) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------------
// the request / response round trip (rmt_client::mtx is held for the whole exchange)
// ---------------------------------------------------------------------------------------------

struct rmt_resp {
    scrapgpu_status       status = SCRAPGPU_STATUS_BACKEND_ERROR;
    sgr::cur              body = { nullptr, 0 }; // payload after the error string
    std::vector<uint8_t>  storage;
};

rmt_resp rmt_invoke(rmt_client * c, uint32_t op, const std::vector<uint8_t> & req) {
    rmt_resp r;
    if (c->dead) {
        r.status = SCRAPGPU_STATUS_DEVICE_LOST;
        scrapgpu_set_last_error("the remote server connection is closed");
        return r;
    }
    uint8_t hdr[sgr::kHdrSize];
    hdr_put(hdr, op, req.size());
    if (!write_all(c->fd, hdr, sizeof(hdr)) || (!req.empty() && !write_all(c->fd, req.data(), req.size()))) {
        r.status = rmt_lost(c, "remote: sending the request failed (server died?)");
        return r;
    }
    uint64_t size = 0;
    uint32_t code = 0;
    if (!read_all(c->fd, hdr, sizeof(hdr)) || !hdr_get(hdr, code, size)) {
        r.status = rmt_lost(c, "remote: the server closed the connection");
        return r;
    }
    // a legitimate response carries at most the inline bulk limit plus the framing; anything
    // bigger means a corrupt or hostile peer (audit H-2)
    if (size > (uint64_t) sgr::kMaxPayload + 64 * 1024) {
        r.status = rmt_lost(c, "remote: malformed response header");
        return r;
    }
    try {
        r.storage.resize((size_t) size);
    } catch (const std::bad_alloc &) {
        // the payload cannot be read without the buffer, so the stream cannot be resynchronised
        r.status = rmt_lost(c, "remote: out of memory for the response payload");
        return r;
    }
    if (size > 0 && !read_all(c->fd, r.storage.data(), (size_t) size)) {
        r.status = rmt_lost(c, "remote: the server closed the connection mid-response");
        return r;
    }
    r.status = (scrapgpu_status) code;
    r.body   = { r.storage.data(), r.storage.size() };
    std::string err;
    bool present = false;
    if (!sgr::get_str(r.body, err, &present)) {
        r.status = rmt_lost(c, "remote: malformed response");
        return r;
    }
    if (r.status != SCRAPGPU_STATUS_OK) {
        scrapgpu_set_last_error(present && !err.empty() ? err.c_str() : "remote call failed");
    }
    return r;
}

// ---------------------------------------------------------------------------------------------
// shared memory (grown on demand, negotiated through the socket)
// ---------------------------------------------------------------------------------------------

bool rmt_shm_map(rmt_client * c, size_t size) {
    if (c->shm != nullptr) {
        munmap(c->shm, c->shm_size);
        c->shm = nullptr;
    }
    c->shm_size = 0;
    if (size == 0 || c->shm_fd < 0) {
        return size == 0;
    }
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, c->shm_fd, 0);
    if (p == MAP_FAILED) {
        return false;
    }
    c->shm      = static_cast<uint8_t *>(p);
    c->shm_size = size;
    return true;
}

// grows the region to at least `need` bytes (a SHM_ENSURE round trip, only on growth)
bool rmt_shm_ensure(rmt_client * c, size_t need) {
    if (need <= c->shm_size) {
        return c->shm_size > 0 || need == 0;
    }
    if (c->shm_fd < 0) {
        return false;
    }
    std::vector<uint8_t> req;
    sgr::put_u64(req, need);
    rmt_resp r = rmt_invoke(c, SGR_OP_SHM_ENSURE, req);
    if (r.status != SCRAPGPU_STATUS_OK) {
        return false;
    }
    uint64_t now = 0;
    if (!sgr::get_u64(r.body, now)) {
        return false;
    }
    return now >= need && rmt_shm_map(c, (size_t) now);
}

// a bulk send: the data goes in front of the request through shm when it fits, else inline in the
// payload. `head` already carries the call's fields; this appends {u32 via_shm, [data]}.
rmt_resp rmt_push(rmt_client * c, uint32_t op, std::vector<uint8_t> & req, const void * data, size_t size) {
    const bool shm = size > 0 && c->shm_fd >= 0 && rmt_shm_ensure(c, size);
    if (!shm && size > 0 && size + req.size() > sgr::kMaxPayload) {
        // inline payloads are bounded by the protocol limit; bigger transfers need the shm (the
        // buffer_write callers split their data into per-message chunks instead)
        scrapgpu_set_last_error("remote: bulk transfer needs the shared memory (inline limit)");
        rmt_resp r;
        r.status = SCRAPGPU_STATUS_OUT_OF_MEMORY;
        return r;
    }
    if (shm) {
        std::memcpy(c->shm, data, size);
    }
    sgr::put_u32(req, shm ? 1u : 0u);
    if (!shm && size > 0) {
        sgr::put_bytes(req, data, size);
    }
    return rmt_invoke(c, op, req);
}

// a bulk receive: the response carries {u64 shm_size_now, u32 via_shm, [data]}; the server grows
// the region on its own and reports the new size
scrapgpu_status rmt_pull(rmt_client * c, rmt_resp & r, void * data, size_t size) {
    uint64_t shm_now = 0;
    uint32_t via = 0;
    if (r.status != SCRAPGPU_STATUS_OK) {
        return r.status;
    }
    if (!sgr::get_u64(r.body, shm_now) || !sgr::get_u32(r.body, via)) {
        scrapgpu_set_last_error("remote: malformed bulk response");
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    if (size == 0) {
        return SCRAPGPU_STATUS_OK;
    }
    if (via != 0) {
        if (shm_now > sgr::kMaxShm) {
            scrapgpu_set_last_error("remote: the reported shared memory size exceeds the protocol limit");
            return SCRAPGPU_STATUS_BACKEND_ERROR;
        }
        if (shm_now > c->shm_size && !rmt_shm_map(c, (size_t) shm_now)) {
            scrapgpu_set_last_error("remote: cannot map the shared memory the server grew");
            return SCRAPGPU_STATUS_BACKEND_ERROR;
        }
        if (c->shm == nullptr || c->shm_size < size) {
            scrapgpu_set_last_error("remote: the shared memory is smaller than the transfer");
            return SCRAPGPU_STATUS_BACKEND_ERROR;
        }
        std::memcpy(data, c->shm, size);
        return SCRAPGPU_STATUS_OK;
    }
    if (!sgr::get_bytes(r.body, data, size)) {
        scrapgpu_set_last_error("remote: short bulk response");
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// proxy object checks and id translation
// ---------------------------------------------------------------------------------------------

rmt_device * as_device(scrapgpu_device_t * p) {
    auto * d = reinterpret_cast<rmt_device *>(p);
    return d != nullptr && d->magic == kProxyMagic ? d : nullptr;
}

rmt_buffer * as_buffer(scrapgpu_buffer_t * p) {
    auto * b = reinterpret_cast<rmt_buffer *>(p);
    return b != nullptr && b->magic == kProxyMagic ? b : nullptr;
}

rmt_stream * as_stream(scrapgpu_stream_t * p) {
    auto * s = reinterpret_cast<rmt_stream *>(p);
    return s != nullptr && s->magic == kProxyMagic ? s : nullptr;
}

rmt_plan * as_plan(scrapgpu_plan_t * p) {
    auto * q = reinterpret_cast<rmt_plan *>(p);
    return q != nullptr && q->magic == kProxyMagic ? q : nullptr;
}

// the wire id of a proxy buffer; *ok is cleared on a buffer that does not belong to this remote
uint64_t buffer_id_of(rmt_client * c, scrapgpu_buffer_t * p, bool & ok) {
    if (p == nullptr) {
        return 0;
    }
    rmt_buffer * b = as_buffer(p);
    if (b == nullptr || b->owner != c) {
        ok = false;
        return 0;
    }
    return b->id;
}

// serializes an op desc, translating every tensor's buffer to a wire id. `bad` reports a desc
// field error or a foreign buffer.
bool rmt_put_op(std::vector<uint8_t> & req, rmt_client * c, const scrapgpu_op_desc * desc, bool & bad) {
    return sgr::put_op_desc(req, *desc, [&](scrapgpu_buffer_t * p) { return buffer_id_of(c, p, bad); }) && !bad;
}

// ---------------------------------------------------------------------------------------------
// vtable: devices
// ---------------------------------------------------------------------------------------------

size_t rmt_get_device_count(void * userdata) {
    auto * c = static_cast<rmt_client *>(userdata);
    return c->dead ? 0 : c->device_count;
}

scrapgpu_status rmt_get_device(void * userdata, size_t index, scrapgpu_device_t ** out_device) {
    auto * c = static_cast<rmt_client *>(userdata);
    if (out_device == nullptr) {
        return rmt_invalid("out_device is null");
    }
    *out_device = nullptr;
    if (c->dead || index >= c->device_count) {
        return rmt_invalid("remote device index out of range");
    }
    if (index >= c->devices.size()) {
        c->devices.resize(c->device_count);
    }
    if (c->devices[index] == nullptr) {
        auto d = std::unique_ptr<rmt_device>(new (std::nothrow) rmt_device());
        if (d == nullptr) {
            scrapgpu_set_last_error("out of host memory");
            return SCRAPGPU_STATUS_OUT_OF_MEMORY;
        }
        // device ids are the server-side device indices
        std::vector<uint8_t> req;
        sgr::put_u64(req, index);
        std::lock_guard<std::mutex> lock(c->mtx);
        rmt_resp r = rmt_invoke(c, SGR_OP_GET_DEVICE, req);
        if (r.status != SCRAPGPU_STATUS_OK) {
            return r.status;
        }
        uint64_t id = 0;
        if (!sgr::get_u64(r.body, id)) {
            return rmt_invalid("remote: malformed get_device response");
        }
        d->hdr.backend = &c->vt;
        d->owner       = c;
        d->id          = id;
        c->devices[index] = std::move(d);
    }
    *out_device = reinterpret_cast<scrapgpu_device_t *>(c->devices[index].get());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status rmt_device_get_info(scrapgpu_device_t * device, scrapgpu_device_info * info) {
    rmt_device * d = as_device(device);
    if (d == nullptr) {
        return rmt_invalid("not a remote device");
    }
    if (info == nullptr || info->struct_size < sizeof(size_t)) {
        return rmt_invalid("device info struct is too small");
    }
    rmt_client * c = d->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, d->id);
    std::lock_guard<std::mutex> lock(c->mtx);
    rmt_resp r = rmt_invoke(c, SGR_OP_DEVICE_GET_INFO, req);
    if (r.status != SCRAPGPU_STATUS_OK) {
        return r.status;
    }
    scrapgpu_device_info full = {};
    if (!sgr::get_device_info(r.body, full, d->info_str)) {
        return rmt_invalid("remote: malformed device info response");
    }
    // the remote device joins the local list under a "remote:" name
    d->prefixed_name = std::string("remote:") + (full.name != nullptr ? full.name : "?");
    full.name        = d->prefixed_name.c_str();
    std::memcpy(info, &full, info->struct_size < sizeof(full) ? info->struct_size : sizeof(full));
    info->struct_size = info->struct_size < sizeof(full) ? info->struct_size : sizeof(full);
    return SCRAPGPU_STATUS_OK;
}

// ---------------------------------------------------------------------------------------------
// vtable: buffers
// ---------------------------------------------------------------------------------------------

scrapgpu_status rmt_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer) {
    rmt_device * d = as_device(device);
    if (d == nullptr || out_buffer == nullptr) {
        return rmt_invalid("not a remote device or out_buffer is null");
    }
    *out_buffer = nullptr;
    rmt_client * c = d->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, d->id);
    sgr::put_u64(req, size);
    std::lock_guard<std::mutex> lock(c->mtx);
    rmt_resp r = rmt_invoke(c, SGR_OP_BUFFER_ALLOC, req);
    if (r.status != SCRAPGPU_STATUS_OK) {
        return r.status;
    }
    uint64_t id = 0;
    if (!sgr::get_u64(r.body, id) || id == 0) {
        return rmt_invalid("remote: malformed buffer_alloc response");
    }
    auto b = std::unique_ptr<rmt_buffer>(new (std::nothrow) rmt_buffer());
    if (b == nullptr) {
        scrapgpu_set_last_error("out of host memory");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    b->hdr.backend = &c->vt;
    b->owner       = c;
    b->id          = id;
    *out_buffer    = reinterpret_cast<scrapgpu_buffer_t *>(b.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status rmt_buffer_free(scrapgpu_buffer_t * buffer) {
    rmt_buffer * b = as_buffer(buffer);
    if (b == nullptr) {
        return rmt_invalid("not a remote buffer");
    }
    rmt_client * c = b->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, b->id);
    std::lock_guard<std::mutex> lock(c->mtx);
    const scrapgpu_status st = rmt_invoke(c, SGR_OP_BUFFER_FREE, req).status;
    if (st != SCRAPGPU_STATUS_INVALID_ARGUMENT) {
        // the proxy is dead regardless of the transport result — except when the server refused
        // the free (a live plan references the buffer, audit M-2): then the id is still valid
        delete b;
    }
    return st;
}

scrapgpu_status rmt_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking) {
    rmt_stream * s = as_stream(stream);
    rmt_buffer * b = as_buffer(buffer);
    if (s == nullptr || b == nullptr || s->owner != b->owner) {
        return rmt_invalid("stream or buffer is not a remote object of this connection");
    }
    rmt_client * c = s->owner;
    const uint8_t * q = static_cast<const uint8_t *>(data);
    std::lock_guard<std::mutex> lock(c->mtx);
    // an inline request is bounded by the protocol limit: a bigger write is chunked so it still
    // works when the shared memory is unavailable (each chunk is an ordered BUFFER_WRITE)
    for (size_t done = 0;;) {
        const size_t chunk = std::min<size_t>(size - done, sgr::kMaxPayload / 2);
        std::vector<uint8_t> req;
        sgr::put_u64(req, s->id);
        sgr::put_u64(req, b->id);
        sgr::put_u64(req, offset + done);
        sgr::put_u64(req, chunk);
        sgr::put_u32(req, blocking ? 1u : 0u);
        const scrapgpu_status st = rmt_push(c, SGR_OP_BUFFER_WRITE, req, q + done, chunk).status;
        if (st != SCRAPGPU_STATUS_OK) {
            return st;
        }
        done += chunk;
        if (done >= size) {
            return SCRAPGPU_STATUS_OK;
        }
    }
}

scrapgpu_status rmt_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking) {
    rmt_stream * s = as_stream(stream);
    rmt_buffer * b = as_buffer(buffer);
    if (s == nullptr || b == nullptr || s->owner != b->owner) {
        return rmt_invalid("stream or buffer is not a remote object of this connection");
    }
    rmt_client * c = s->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    sgr::put_u64(req, b->id);
    sgr::put_u64(req, offset);
    sgr::put_u64(req, size);
    sgr::put_u32(req, blocking ? 1u : 0u);
    std::lock_guard<std::mutex> lock(c->mtx);
    rmt_resp r = rmt_invoke(c, SGR_OP_BUFFER_READ, req);
    return rmt_pull(c, r, data, size);
}

scrapgpu_status rmt_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking) {
    rmt_stream * s = as_stream(stream);
    rmt_buffer * d = as_buffer(dst);
    rmt_buffer * x = as_buffer(src);
    if (s == nullptr || d == nullptr || x == nullptr || s->owner != d->owner || s->owner != x->owner) {
        return rmt_invalid("stream or buffers are not remote objects of this connection");
    }
    rmt_client * c = s->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    sgr::put_u64(req, d->id);
    sgr::put_u64(req, dst_offset);
    sgr::put_u64(req, x->id);
    sgr::put_u64(req, src_offset);
    sgr::put_u64(req, size);
    sgr::put_u32(req, blocking ? 1u : 0u);
    std::lock_guard<std::mutex> lock(c->mtx);
    return rmt_invoke(c, SGR_OP_BUFFER_COPY, req).status;
}

scrapgpu_status rmt_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking) {
    rmt_stream * s = as_stream(stream);
    rmt_buffer * b = as_buffer(buffer);
    if (s == nullptr || b == nullptr || s->owner != b->owner) {
        return rmt_invalid("stream or buffer is not a remote object of this connection");
    }
    rmt_client * c = s->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    sgr::put_u64(req, b->id);
    sgr::put_u64(req, offset);
    sgr::put_u64(req, size);
    sgr::put_u32(req, byte);
    sgr::put_u32(req, blocking ? 1u : 0u);
    std::lock_guard<std::mutex> lock(c->mtx);
    return rmt_invoke(c, SGR_OP_BUFFER_FILL, req).status;
}

// ---------------------------------------------------------------------------------------------
// vtable: streams
// ---------------------------------------------------------------------------------------------

scrapgpu_status rmt_stream_create(scrapgpu_device_t * device, scrapgpu_stream_t ** out_stream) {
    rmt_device * d = as_device(device);
    if (d == nullptr || out_stream == nullptr) {
        return rmt_invalid("not a remote device or out_stream is null");
    }
    *out_stream = nullptr;
    rmt_client * c = d->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, d->id);
    std::lock_guard<std::mutex> lock(c->mtx);
    rmt_resp r = rmt_invoke(c, SGR_OP_STREAM_CREATE, req);
    if (r.status != SCRAPGPU_STATUS_OK) {
        return r.status;
    }
    uint64_t id = 0;
    if (!sgr::get_u64(r.body, id) || id == 0) {
        return rmt_invalid("remote: malformed stream_create response");
    }
    auto s = std::unique_ptr<rmt_stream>(new (std::nothrow) rmt_stream());
    if (s == nullptr) {
        scrapgpu_set_last_error("out of host memory");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    s->hdr.backend = &c->vt;
    s->owner       = c;
    s->id          = id;
    s->dev         = d;
    *out_stream    = reinterpret_cast<scrapgpu_stream_t *>(s.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status rmt_stream_free(scrapgpu_stream_t * stream) {
    rmt_stream * s = as_stream(stream);
    if (s == nullptr) {
        return rmt_invalid("not a remote stream");
    }
    rmt_client * c = s->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    std::lock_guard<std::mutex> lock(c->mtx);
    const scrapgpu_status st = rmt_invoke(c, SGR_OP_STREAM_FREE, req).status;
    delete s;
    return st;
}

scrapgpu_status rmt_stream_sync(scrapgpu_stream_t * stream) {
    rmt_stream * s = as_stream(stream);
    if (s == nullptr) {
        return rmt_invalid("not a remote stream");
    }
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    std::lock_guard<std::mutex> lock(s->owner->mtx);
    return rmt_invoke(s->owner, SGR_OP_STREAM_SYNC, req).status;
}

// the device a stream runs on: the proxy pointer IS the identity the core compares, so two streams
// of the same server device take the device-copy fast path of scrapgpu_tensor_copy (forwarded as a
// BUFFER_COPY), a remote/local pair goes through the host staging path
scrapgpu_device_t * rmt_stream_device(scrapgpu_stream_t * stream) {
    rmt_stream * s = as_stream(stream);
    return s == nullptr ? nullptr : reinterpret_cast<scrapgpu_device_t *>(s->dev);
}

// ---------------------------------------------------------------------------------------------
// vtable: tensor transfers and ops
// ---------------------------------------------------------------------------------------------

scrapgpu_status rmt_tensor_upload(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, const void * host, size_t size) {
    rmt_stream * s = as_stream(stream);
    if (s == nullptr || desc == nullptr) {
        return rmt_invalid("not a remote stream or tensor desc is null");
    }
    rmt_client * c = s->owner;
    bool bad = desc->buffer != nullptr && (as_buffer(desc->buffer) == nullptr || as_buffer(desc->buffer)->owner != c);
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    sgr::put_tensor_desc(req, *desc, bad ? 0 : buffer_id_of(c, desc->buffer, bad));
    sgr::put_u64(req, size);
    if (bad) {
        return rmt_invalid("tensor buffer does not belong to this remote");
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    return rmt_push(c, SGR_OP_TENSOR_UPLOAD, req, host, size).status;
}

scrapgpu_status rmt_tensor_download(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, void * host, size_t size) {
    rmt_stream * s = as_stream(stream);
    if (s == nullptr || desc == nullptr) {
        return rmt_invalid("not a remote stream or tensor desc is null");
    }
    rmt_client * c = s->owner;
    bool bad = desc->buffer != nullptr && (as_buffer(desc->buffer) == nullptr || as_buffer(desc->buffer)->owner != c);
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    sgr::put_tensor_desc(req, *desc, bad ? 0 : buffer_id_of(c, desc->buffer, bad));
    sgr::put_u64(req, size);
    if (bad) {
        return rmt_invalid("tensor buffer does not belong to this remote");
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    rmt_resp r = rmt_invoke(c, SGR_OP_TENSOR_DOWNLOAD, req);
    return rmt_pull(c, r, host, size);
}

scrapgpu_status rmt_op_supported(scrapgpu_device_t * device, const scrapgpu_op_desc * desc, const char ** reason) {
    rmt_device * d = as_device(device);
    if (reason != nullptr) {
        *reason = nullptr;
    }
    if (d == nullptr || desc == nullptr) {
        if (reason != nullptr) {
            *reason = "not a remote device";
        }
        return rmt_invalid("not a remote device or op desc is null");
    }
    rmt_client * c = d->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, d->id);
    bool bad = false;
    if (!rmt_put_op(req, c, desc, bad)) {
        const char * why = bad ? "tensor buffer does not belong to this remote" : "op desc could not be serialized";
        if (reason != nullptr) {
            *reason = why;
        }
        return rmt_invalid(why);
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    rmt_resp r = rmt_invoke(c, SGR_OP_OP_SUPPORTED, req);
    std::string why;
    bool present = false;
    if (!sgr::get_str(r.body, why, &present)) {
        // a transport error already set the status and the last error
        return r.status != SCRAPGPU_STATUS_OK ? r.status : rmt_invalid("remote: malformed op_supported response");
    }
    if (present) {
        c->reason_scratch = why;
        if (reason != nullptr) {
            *reason = c->reason_scratch.c_str();
        }
    }
    return r.status;
}

scrapgpu_status rmt_op_execute(scrapgpu_stream_t * stream, const scrapgpu_op_desc * desc) {
    rmt_stream * s = as_stream(stream);
    if (s == nullptr || desc == nullptr) {
        return rmt_invalid("not a remote stream or op desc is null");
    }
    rmt_client * c = s->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    bool bad = false;
    if (!rmt_put_op(req, c, desc, bad)) {
        return rmt_invalid(bad ? "tensor buffer does not belong to this remote" : "op desc could not be serialized");
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    return rmt_invoke(c, SGR_OP_OP_EXECUTE, req).status;
}

// ---------------------------------------------------------------------------------------------
// vtable: plans
// ---------------------------------------------------------------------------------------------

scrapgpu_status rmt_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan) {
    rmt_device * d = as_device(device);
    if (d == nullptr || desc == nullptr || out_plan == nullptr ||
        desc->struct_size < offsetof(scrapgpu_plan_desc, node_count) + sizeof(desc->node_count) ||
        desc->node_count > 4096 || (desc->node_count > 0 && desc->nodes == nullptr)) {
        return rmt_invalid("device, plan desc or node list is invalid");
    }
    *out_plan = nullptr;
    rmt_client * c = d->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, d->id);
    sgr::put_u64(req, desc->node_count);
    bool bad = false;
    for (size_t i = 0; i < desc->node_count && !bad; ++i) {
        const scrapgpu_plan_node & n = desc->nodes[i];
        if (n.struct_size < offsetof(scrapgpu_plan_node, op) + sizeof(n.op)) {
            bad = true;
            break;
        }
        sgr::put_u32(req, n.output_use_count);
        if (!rmt_put_op(req, c, &n.op, bad)) {
            bad = true;
        }
    }
    if (bad) {
        return rmt_invalid("a plan node could not be serialized (foreign buffer or bad desc)");
    }
    std::lock_guard<std::mutex> lock(c->mtx);
    rmt_resp r = rmt_invoke(c, SGR_OP_PLAN_BUILD, req);
    if (r.status != SCRAPGPU_STATUS_OK) {
        return r.status;
    }
    uint64_t id = 0;
    if (!sgr::get_u64(r.body, id) || id == 0) {
        return rmt_invalid("remote: malformed plan_build response");
    }
    auto p = std::unique_ptr<rmt_plan>(new (std::nothrow) rmt_plan());
    if (p == nullptr) {
        scrapgpu_set_last_error("out of host memory");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    p->hdr.backend = &c->vt;
    p->owner       = c;
    p->id          = id;
    *out_plan      = reinterpret_cast<scrapgpu_plan_t *>(p.release());
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status rmt_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan) {
    rmt_stream * s = as_stream(stream);
    rmt_plan *   p = as_plan(plan);
    if (s == nullptr || p == nullptr || s->owner != p->owner) {
        return rmt_invalid("stream or plan is not a remote object of this connection");
    }
    std::vector<uint8_t> req;
    sgr::put_u64(req, s->id);
    sgr::put_u64(req, p->id);
    std::lock_guard<std::mutex> lock(s->owner->mtx);
    return rmt_invoke(s->owner, SGR_OP_PLAN_EXECUTE, req).status;
}

scrapgpu_status rmt_plan_free(scrapgpu_plan_t * plan) {
    rmt_plan * p = as_plan(plan);
    if (p == nullptr) {
        return rmt_invalid("not a remote plan");
    }
    rmt_client * c = p->owner;
    std::vector<uint8_t> req;
    sgr::put_u64(req, p->id);
    std::lock_guard<std::mutex> lock(c->mtx);
    const scrapgpu_status st = rmt_invoke(c, SGR_OP_PLAN_FREE, req).status;
    delete p;
    return st;
}

// ---------------------------------------------------------------------------------------------
// connect / spawn
// ---------------------------------------------------------------------------------------------

int rmt_connect_socket(const char * path) {
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    sockaddr_un addr = {};
    addr.sun_family = AF_UNIX;
    if (std::strlen(path) >= sizeof(addr.sun_path)) {
        close(fd);
        errno = ENAMETOOLONG;
        return -1;
    }
    std::strcpy(addr.sun_path, path);
    if (connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
#ifdef SO_NOSIGPIPE
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
    return fd;
}

// the handshake: protocol version, the shm region (name + size) and the device count
scrapgpu_status rmt_handshake(rmt_client * c) {
    std::vector<uint8_t> req;
    sgr::put_u32(req, sgr::kVersion);
    rmt_resp r = rmt_invoke(c, SGR_OP_HELLO, req);
    if (r.status != SCRAPGPU_STATUS_OK) {
        return r.status;
    }
    uint32_t version = 0;
    uint64_t count = 0, shm_size = 0;
    std::string shm_name;
    bool present = false;
    if (!sgr::get_u32(r.body, version) || version != sgr::kVersion ||
        !sgr::get_u64(r.body, count) || !sgr::get_u64(r.body, shm_size) ||
        !sgr::get_str(r.body, shm_name, &present)) {
        return rmt_lost(c, "remote: malformed handshake response");
    }
    c->device_count = (size_t) count;
    if (present && shm_size > 0) {
        c->shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0600);
        if (c->shm_fd >= 0) {
            rmt_shm_map(c, (size_t) shm_size); // a mapping failure just disables shm (inline fallback)
            if (c->shm == nullptr) {
                close(c->shm_fd);
                c->shm_fd = -1;
            }
        }
    }
    return SCRAPGPU_STATUS_OK;
}

// the per-client vtable; the registered copy in the core carries userdata = the client
void rmt_init_vtable(scrapgpu_backend_vtable & vt, rmt_client * c) {
    vt.struct_size        = sizeof(vt);
    vt.name               = "remote";
    vt.userdata           = c;
    vt.get_device_count   = rmt_get_device_count;
    vt.get_device         = rmt_get_device;
    vt.device_get_info    = rmt_device_get_info;
    vt.buffer_alloc       = rmt_buffer_alloc;
    vt.buffer_free        = rmt_buffer_free;
    vt.buffer_write       = rmt_buffer_write;
    vt.buffer_read        = rmt_buffer_read;
    vt.buffer_copy        = rmt_buffer_copy;
    vt.buffer_fill        = rmt_buffer_fill;
    vt.stream_sync        = rmt_stream_sync;
    vt.op_supported       = rmt_op_supported;
    vt.plan_build         = rmt_plan_build;
    vt.plan_execute       = rmt_plan_execute;
    vt.plan_free          = rmt_plan_free;
    vt.stream_create      = rmt_stream_create;
    vt.stream_free        = rmt_stream_free;
    vt.tensor_upload      = rmt_tensor_upload;
    vt.tensor_download    = rmt_tensor_download;
    vt.op_execute         = rmt_op_execute;
    vt.stream_device      = rmt_stream_device;
}

// the directory of the running executable (the server is built next to the tests and examples)
std::string rmt_exe_dir() {
    char path[4096];
#ifdef __APPLE__
    uint32_t n = sizeof(path);
    if (_NSGetExecutablePath(path, &n) != 0) {
        return {};
    }
    char real[4096];
    if (realpath(path, real) == nullptr) {
        std::strncpy(real, path, sizeof(real) - 1);
        real[sizeof(real) - 1] = '\0';
    }
#else
    const ssize_t r = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (r <= 0) {
        return {};
    }
    path[r] = '\0';
    char * real = path;
#endif
    std::string dir(real);
    const size_t slash = dir.find_last_of('/');
    return slash == std::string::npos ? std::string() : dir.substr(0, slash);
}

std::string rmt_server_path() {
    const char * env = std::getenv("SCRAPGPU_DEVICE_SERVER");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    const std::string dir = rmt_exe_dir();
    if (!dir.empty()) {
        const std::string sibling = dir + "/scrapgpu-device-server";
        if (access(sibling.c_str(), X_OK) == 0) {
            return sibling;
        }
    }
    return "scrapgpu-device-server"; // PATH lookup by execvp
}

void rmt_disconnect(rmt_client * c);

// registers the client backend; on failure the client is shut down (socket closed, spawned server
// reaped) and deleted. Takes ownership of `c` either way.
scrapgpu_status rmt_register_client(std::unique_ptr<rmt_client> cp, scrapgpu_remote_t ** out) {
    rmt_client * c = cp.get();
    rmt_init_vtable(c->vt, c);
    const scrapgpu_status st = scrapgpu_register_backend(&c->vt);
    if (st != SCRAPGPU_STATUS_OK) {
        rmt_disconnect(c);
        if (c->child > 0) {
            kill(c->child, SIGKILL);
            waitpid(c->child, nullptr, 0);
        }
        if (!c->socket_path.empty()) {
            unlink(c->socket_path.c_str());
        }
        return st;
    }
    cp.release();
    *out = reinterpret_cast<scrapgpu_remote_t *>(c);
    return SCRAPGPU_STATUS_OK;
}

void rmt_disconnect(rmt_client * c) {
    if (c->fd >= 0) {
        close(c->fd);
        c->fd = -1;
    }
    c->dead = true;
    if (c->shm != nullptr) {
        munmap(c->shm, c->shm_size);
        c->shm = nullptr;
        c->shm_size = 0;
    }
    if (c->shm_fd >= 0) {
        close(c->shm_fd);
        c->shm_fd = -1;
    }
    c->device_count = 0;
}

} // namespace

extern "C" {

scrapgpu_status scrapgpu_remote_connect(const char * socket_path, scrapgpu_remote_t ** out) {
    if (socket_path == nullptr || socket_path[0] == '\0' || out == nullptr) {
        scrapgpu_set_last_error("socket path or out is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    *out = nullptr;
    std::unique_ptr<rmt_client> c(new (std::nothrow) rmt_client());
    if (c == nullptr) {
        scrapgpu_set_last_error("out of host memory");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    c->fd = rmt_connect_socket(socket_path);
    if (c->fd < 0) {
        char buf[256];
        snprintf(buf, sizeof(buf), "remote: cannot connect to %s: %s", socket_path, strerror(errno));
        scrapgpu_set_last_error(buf);
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    c->socket_path = socket_path;
    const scrapgpu_status st = rmt_handshake(c.get());
    if (st != SCRAPGPU_STATUS_OK) {
        rmt_disconnect(c.get());
        return st;
    }
    return rmt_register_client(std::move(c), out);
}

scrapgpu_status scrapgpu_remote_spawn(const char * const * env_spec, scrapgpu_remote_t ** out) {
    if (out == nullptr) {
        scrapgpu_set_last_error("out is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    *out = nullptr;
    for (const char * const * e = env_spec; e != nullptr && *e != nullptr; ++e) {
        if (std::strchr(*e, '=') == nullptr) {
            scrapgpu_set_last_error("remote spawn env entries must be KEY=VALUE");
            return SCRAPGPU_STATUS_INVALID_ARGUMENT;
        }
    }
    const std::string server = rmt_server_path();
    const char * backend = nullptr;
    for (const char * const * e = env_spec; e != nullptr && *e != nullptr; ++e) {
        if (std::strncmp(*e, "SCRAPGPU_REMOTE_BACKEND=", 24) == 0) {
            backend = *e + 24;
        }
    }
    if (backend == nullptr) {
        backend = std::getenv("SCRAPGPU_REMOTE_BACKEND");
    }
    if (backend == nullptr || backend[0] == '\0') {
        backend = "opencl11";
    }

    static std::atomic<int> spawn_counter{0};
    // the socket lives in a private directory (mode 0700, unpredictable name): no other user can create, replace or
    // connect to it
    char sock_dir[] = "/tmp/scrapgpu-XXXXXX";
    if (mkdtemp(sock_dir) == nullptr) {
        char buf[160];
        snprintf(buf, sizeof(buf), "remote: mkdtemp failed: %s", strerror(errno));
        scrapgpu_set_last_error(buf);
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    char sock_path[160];
    snprintf(sock_path, sizeof(sock_path), "%s/server-%d.sock", sock_dir, spawn_counter.fetch_add(1));

    const pid_t pid = fork();
    if (pid < 0) {
        char buf[160];
        snprintf(buf, sizeof(buf), "remote: fork failed: %s", strerror(errno));
        scrapgpu_set_last_error(buf);
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    if (pid == 0) {
        // child: apply the env overrides and exec the server (stderr stays with the parent's)
        for (const char * const * e = env_spec; e != nullptr && *e != nullptr; ++e) {
            const char * eq = std::strchr(*e, '=');
            const std::string key(*e, eq);
            setenv(key.c_str(), eq + 1, 1);
        }
        execl(server.c_str(), "scrapgpu-device-server", "--backend", backend, "--socket", sock_path, (char *) nullptr);
        // PATH lookup as the fallback of a bare name
        execlp("scrapgpu-device-server", "scrapgpu-device-server", "--backend", backend, "--socket", sock_path, (char *) nullptr);
        _exit(127);
    }

    // parent: the server binds the socket itself; connect as soon as it shows up
    std::unique_ptr<rmt_client> c(new (std::nothrow) rmt_client());
    if (c == nullptr) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        scrapgpu_set_last_error("out of host memory");
        return SCRAPGPU_STATUS_OUT_OF_MEMORY;
    }
    c->child       = pid;
    c->socket_path = sock_path;
    for (int i = 0; i < 3000; ++i) { // up to ~30 s: OpenCL probing of old cards is slow
        int wstatus = 0;
        if (waitpid(pid, &wstatus, WNOHANG) == pid) {
            scrapgpu_set_last_error("remote: the device server exited before the socket came up (see stderr)");
            unlink(sock_path);
        rmdir(sock_dir);
            return SCRAPGPU_STATUS_BACKEND_ERROR;
        }
        c->fd = rmt_connect_socket(sock_path);
        if (c->fd >= 0) {
            break;
        }
        usleep(10000);
    }
    if (c->fd < 0) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
        unlink(sock_path);
        rmdir(sock_dir);
        scrapgpu_set_last_error("remote: the device server socket never came up");
        return SCRAPGPU_STATUS_BACKEND_ERROR;
    }
    const scrapgpu_status st = rmt_handshake(c.get());
    if (st != SCRAPGPU_STATUS_OK) {
        rmt_disconnect(c.get());
        waitpid(pid, nullptr, 0); // the server exits when the client disconnects
        unlink(sock_path);
        rmdir(sock_dir);
        return st;
    }
    return rmt_register_client(std::move(c), out);
}

void scrapgpu_remote_close(scrapgpu_remote_t * remote) {
    auto * c = reinterpret_cast<rmt_client *>(remote);
    if (c == nullptr) {
        return;
    }
    rmt_disconnect(c);
    // the server exits when the client disconnects; reap a spawned child (kill it if it lingers)
    if (c->child > 0) {
        for (int i = 0; i < 200; ++i) {
            if (waitpid(c->child, nullptr, WNOHANG) == c->child) {
                break;
            }
            usleep(5000);
        }
        if (waitpid(c->child, nullptr, WNOHANG) == 0) {
            kill(c->child, SIGKILL);
            waitpid(c->child, nullptr, 0);
        }
    }
    // a spawned socket path is ours to remove; a connect()ed one belongs to the server
    if (c->child > 0 && !c->socket_path.empty()) {
        unlink(c->socket_path.c_str());
        const size_t slash = c->socket_path.rfind('/');
        if (slash != std::string::npos) {
            rmdir(c->socket_path.substr(0, slash).c_str()); // the private directory of a spawned server
        }
    }
    c->devices.clear();
    delete c;
}

}
