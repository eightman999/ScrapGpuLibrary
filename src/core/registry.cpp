#include "scrapgpu/backend.h"

#include <algorithm>
#include <cstddef>
#include <mutex>
#include <vector>

namespace {
std::mutex g_registry_mutex;
std::vector<scrapgpu_backend_vtable> g_backends;

scrapgpu_status unsupported(const char * message) {
    scrapgpu_set_last_error(message);
    return SCRAPGPU_STATUS_UNSUPPORTED;
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
    g_backends.push_back(*backend);
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
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    for (const auto & backend : g_backends) {
        if (backend.device_get_info != nullptr) {
            const scrapgpu_status status = backend.device_get_info(device, info);
            if (status != SCRAPGPU_STATUS_UNSUPPORTED) {
                return status;
            }
        }
    }
    return unsupported("no registered backend owns the device");
}

scrapgpu_status scrapgpu_buffer_alloc(scrapgpu_device_t * device, size_t size, scrapgpu_buffer_t ** out_buffer) {
    (void) size;
    if (device == nullptr || out_buffer == nullptr) {
        scrapgpu_set_last_error("device or out_buffer is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    *out_buffer = nullptr;
    return unsupported("no buffer backend registered");
}

scrapgpu_status scrapgpu_buffer_free(scrapgpu_buffer_t * buffer) {
    if (buffer == nullptr) {
        return SCRAPGPU_STATUS_OK;
    }
    return unsupported("no buffer backend registered");
}

scrapgpu_status scrapgpu_buffer_write(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, const void * data, size_t size, int blocking) {
    (void) stream; (void) buffer; (void) offset; (void) data; (void) size; (void) blocking;
    return unsupported("no buffer backend registered");
}

scrapgpu_status scrapgpu_buffer_read(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, void * data, size_t size, int blocking) {
    (void) stream; (void) buffer; (void) offset; (void) data; (void) size; (void) blocking;
    return unsupported("no buffer backend registered");
}

scrapgpu_status scrapgpu_buffer_copy(scrapgpu_stream_t * stream, scrapgpu_buffer_t * dst, uint64_t dst_offset, scrapgpu_buffer_t * src, uint64_t src_offset, size_t size, int blocking) {
    (void) stream; (void) dst; (void) dst_offset; (void) src; (void) src_offset; (void) size; (void) blocking;
    return unsupported("no buffer backend registered");
}

scrapgpu_status scrapgpu_buffer_fill(scrapgpu_stream_t * stream, scrapgpu_buffer_t * buffer, uint64_t offset, size_t size, uint8_t byte, int blocking) {
    (void) stream; (void) buffer; (void) offset; (void) size; (void) byte; (void) blocking;
    return unsupported("no buffer backend registered");
}

scrapgpu_status scrapgpu_plan_build(scrapgpu_device_t * device, const scrapgpu_plan_desc * desc, scrapgpu_plan_t ** out_plan) {
    if (device == nullptr || desc == nullptr || out_plan == nullptr) {
        scrapgpu_set_last_error("device, plan desc, or out_plan is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    *out_plan = nullptr;
    return unsupported("no plan backend registered");
}

scrapgpu_status scrapgpu_plan_execute(scrapgpu_stream_t * stream, scrapgpu_plan_t * plan) {
    if (stream == nullptr || plan == nullptr) {
        scrapgpu_set_last_error("stream or plan is null");
        return SCRAPGPU_STATUS_INVALID_ARGUMENT;
    }
    return unsupported("no plan backend registered");
}

scrapgpu_status scrapgpu_plan_free(scrapgpu_plan_t * plan) {
    if (plan == nullptr) {
        return SCRAPGPU_STATUS_OK;
    }
    return unsupported("no plan backend registered");
}

}
