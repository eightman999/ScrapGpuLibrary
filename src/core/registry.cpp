#include "scrapgpu/backend.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <vector>

namespace {
std::mutex g_registry_mutex;
std::vector<scrapgpu_backend_vtable> g_backends;

scrapgpu_status unsupported(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_UNSUPPORTED;
}

scrapgpu_status invalid(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_INVALID_ARGUMENT;
}

// the backend an object belongs to (the first member of every backend object is a
// scrapgpu_object_header), nullptr when it does not carry the header
const scrapgpu_backend_vtable * backend_of(const void * object) {
    return object == nullptr ? nullptr : static_cast<const scrapgpu_object_header *>(object)->backend;
}

template <typename F, typename... A>
scrapgpu_status dispatch(const void * object, F field, const char * what, A &&... args) {
    const scrapgpu_backend_vtable * backend = backend_of(object);
    if (backend == nullptr) {
        return invalid("object does not belong to a registered backend");
    }
    if (backend->*field == nullptr) {
        return unsupported(what);
    }
    return (backend->*field)(std::forward<A>(args)...);
}
}

extern "C" {

scrapgpu_status scrapgpu_device_info_init(scrapgpu_device_info * info, size_t struct_size) {
    if (info == nullptr || struct_size < sizeof(size_t)) {
        scrapgpu_set_last_error("device info struct is too small");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    const size_t n = std::min(struct_size, sizeof(scrapgpu_device_info));
    auto * bytes = reinterpret_cast<unsigned char *>(info);
    std::fill(bytes, bytes + n, 0);
    info->struct_size = struct_size;
    if (struct_size >= offsetof(scrapgpu_device_info, generation) + sizeof(info->generation)) {
        info->generation = SCRAPGPU_DEVICE_GENERATION_UNKNOWN;
    }
    return SCRAPGPU_STATUS_OK;
}

scrapgpu_status scrapgpu_register_backend(const scrapgpu_backend_vtable * backend) {
    if (backend == nullptr ||
        backend->struct_size < offsetof(scrapgpu_backend_vtable, plan_free) + sizeof(backend->plan_free) ||
        backend->name == nullptr) {
        scrapgpu_set_last_error("backend vtable is incomplete");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    // the vtable is extended at the end (new entries on a larger ABI version): copy what the caller
    // passed and leave the rest nullptr
    scrapgpu_backend_vtable copy = {};
    std::memcpy(&copy, backend, std::min(backend->struct_size, sizeof(scrapgpu_backend_vtable)));
    copy.struct_size = sizeof(scrapgpu_backend_vtable);
    g_backends.push_back(copy);
    return SCRAPGPU_STATUS_OK;
}

void scrapgpu_unregister_all_backends(void) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    g_backends.clear();
}

size_t scrapgpu_get_device_count(void) {
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    size_t total = 0;
    for (const auto & backend : g_backends) {
        if (backend.get_device_count != nullptr) {
            total += backend.get_device_count(backend.userdata);
        }
    }
    return total;
}

scrapgpu_status scrapgpu_get_device(size_t index, scrapgpu_device_t ** out_device) {
    if (out_device == nullptr) {
        scrapgpu_set_last_error("out_device is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    for (const auto & backend : g_backends) {
        if (backend.get_device_count == nullptr || backend.get_device == nullptr) {
            continue;
        }
        const size_t count = backend.get_device_count(backend.userdata);
        if (index < count) {
            return backend.get_device(backend.userdata, index, out_device);
        }
        index -= count;
    }
    *out_device = nullptr;
    return unsupported("no backend device at the requested index");
}

scrapgpu_status scrapgpu_device_get_info(scrapgpu_device_t * device, scrapgpu_device_info * info) {
    if (device == nullptr || info == nullptr) {
        scrapgpu_set_last_error("device or info is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    return dispatch(device, &scrapgpu_backend_vtable::device_get_info, "no registered backend reports the device", device, info);
}

scrapgpu_status scrapgpu_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer) {
    if (device == nullptr || out_buffer == nullptr) {
        scrapgpu_set_last_error("device or out_buffer is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;
    return dispatch(device, &scrapgpu_backend_vtable::buffer_alloc, "the device backend cannot allocate buffers", device, size, out_buffer);
}

scrapgpu_status scrapgpu_buffer_free(scrapgpu_buffer_t * buffer) {
    if (buffer == nullptr) {
        return SCRAPGPU_STATUS_OK;
    }
    return dispatch(buffer, &scrapgpu_backend_vtable::buffer_free, "the buffer backend cannot free buffers", buffer);
}

scrapgpu_status scrapgpu_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking) {
    if (stream == nullptr || buffer == nullptr || (data == nullptr && size != 0)) {
        return invalid("stream, buffer or data is null");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::buffer_write, "the stream backend cannot write buffers", stream, buffer, offset, data, size, blocking);
}

scrapgpu_status scrapgpu_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking) {
    if (stream == nullptr || buffer == nullptr || (data == nullptr && size != 0)) {
        return invalid("stream, buffer or data is null");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::buffer_read, "the stream backend cannot read buffers", stream, buffer, offset, data, size, blocking);
}

scrapgpu_status scrapgpu_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking) {
    if (stream == nullptr || dst == nullptr || src == nullptr) {
        return invalid("stream, dst or src is null");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::buffer_copy, "the stream backend cannot copy buffers", stream, dst, dst_offset, src, src_offset, size, blocking);
}

scrapgpu_status scrapgpu_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking) {
    if (stream == nullptr || buffer == nullptr) {
        return invalid("stream or buffer is null");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::buffer_fill, "the stream backend cannot fill buffers", stream, buffer, offset, size, byte, blocking);
}

scrapgpu_status scrapgpu_stream_create(scrapgpu_device_t * device, scrapgpu_stream_t ** out_stream) {
    if (device == nullptr || out_stream == nullptr) {
        return invalid("device or out_stream is null");
    }
    *out_stream = nullptr;
    return dispatch(device, &scrapgpu_backend_vtable::stream_create, "the device backend cannot create streams", device, out_stream);
}

scrapgpu_status scrapgpu_stream_free(scrapgpu_stream_t * stream) {
    if (stream == nullptr) {
        return SCRAPGPU_STATUS_OK;
    }
    return dispatch(stream, &scrapgpu_backend_vtable::stream_free, "the stream backend cannot free streams", stream);
}

scrapgpu_status scrapgpu_stream_sync(scrapgpu_stream_t * stream) {
    if (stream == nullptr) {
        return invalid("stream is null");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::stream_sync, "the stream backend cannot synchronize", stream);
}

scrapgpu_status scrapgpu_tensor_upload(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, const void * host, size_t size) {
    if (stream == nullptr || desc == nullptr || (host == nullptr && size != 0)) {
        return invalid("stream, tensor desc or host pointer is null");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::tensor_upload, "the stream backend cannot upload tensors", stream, desc, host, size);
}

scrapgpu_status scrapgpu_tensor_download(scrapgpu_stream_t * stream, const scrapgpu_tensor_desc * desc, void * host, size_t size) {
    if (stream == nullptr || desc == nullptr || (host == nullptr && size != 0)) {
        return invalid("stream, tensor desc or host pointer is null");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::tensor_download, "the stream backend cannot download tensors", stream, desc, host, size);
}

scrapgpu_status scrapgpu_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan) {
    if (device == nullptr || desc == nullptr || out_plan == nullptr) {
        scrapgpu_set_last_error("device, plan desc, or out_plan is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    *out_plan = nullptr;
    return dispatch(device, &scrapgpu_backend_vtable::plan_build, "the device backend cannot build plans", device, desc, out_plan);
}

scrapgpu_status scrapgpu_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan) {
    if (stream == nullptr || plan == nullptr) {
        scrapgpu_set_last_error("stream or plan is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    const scrapgpu_backend_vtable * backend = backend_of(stream);
    if (backend == nullptr || backend != backend_of(plan)) {
        return invalid("stream and plan belong to different backends");
    }
    return dispatch(stream, &scrapgpu_backend_vtable::plan_execute, "the stream backend cannot execute plans", stream, plan);
}

scrapgpu_status scrapgpu_plan_free(scrapgpu_plan_t * plan) {
    if (plan == nullptr) {
        return SCRAPGPU_STATUS_OK;
    }
    return dispatch(plan, &scrapgpu_backend_vtable::plan_free, "the plan backend cannot free plans", plan);
}

}
