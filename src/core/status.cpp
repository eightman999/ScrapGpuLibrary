#include "scrapgpu/status.h"

#include <string>

namespace {
thread_local std::string g_last_error;
}

extern "C" {

uint32_t scrapgpu_get_abi_version(void) {
    return (SCRAPGPU_ABI_VERSION_MAJOR << 16u) | SCRAPGPU_ABI_VERSION_MINOR;
}

const char * scrapgpu_status_string(scrapgpu_status status) {
    switch (status) {
        case SCRAPGPU_STATUS_OK: return "OK";
        case SCRAPGPU_STATUS_UNSUPPORTED: return "UNSUPPORTED";
        case SCRAPGPU_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case SCRAPGPU_STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case SCRAPGPU_STATUS_BACKEND_ERROR: return "BACKEND_ERROR";
        case SCRAPGPU_STATUS_DEVICE_LOST: return "DEVICE_LOST";
    }
    return "UNKNOWN";
}

const char * scrapgpu_last_error(void) {
    return g_last_error.c_str();
}

void scrapgpu_clear_last_error(void) {
    g_last_error.clear();
}

void scrapgpu_set_last_error(const char * message) {
    g_last_error = message == nullptr ? "" : message;
}

}
