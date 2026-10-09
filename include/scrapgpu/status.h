#ifndef SCRAPGPU_STATUS_H
#define SCRAPGPU_STATUS_H

#include "scrapgpu/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum scrapgpu_status {
    SCRAPGPU_STATUS_OK = 0,
    SCRAPGPU_STATUS_UNSUPPORTED = 1,
    SCRAPGPU_STATUS_INVALID_ARGUMENT = 2,
    SCRAPGPU_STATUS_OUT_OF_MEMORY = 3,
    SCRAPGPU_STATUS_BACKEND_ERROR = 4,
    SCRAPGPU_STATUS_DEVICE_LOST = 5,
} scrapgpu_status;

SCRAPGPU_API const char * scrapgpu_status_string(scrapgpu_status status);
SCRAPGPU_API const char * scrapgpu_last_error(void);
SCRAPGPU_API void scrapgpu_clear_last_error(void);
SCRAPGPU_API void scrapgpu_set_last_error(const char * message);

#ifdef __cplusplus
}
#endif

#endif
