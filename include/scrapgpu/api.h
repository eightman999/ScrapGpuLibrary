#ifndef SCRAPGPU_API_H
#define SCRAPGPU_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(SCRAPGPU_BUILD_SHARED)
#  if defined(SCRAPGPU_BUILDING_LIBRARY)
#    define SCRAPGPU_API __declspec(dllexport)
#  else
#    define SCRAPGPU_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define SCRAPGPU_API __attribute__((visibility("default")))
#else
#  define SCRAPGPU_API
#endif

#define SCRAPGPU_ABI_VERSION_MAJOR 0u
#define SCRAPGPU_ABI_VERSION_MINOR 3u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct scrapgpu_device scrapgpu_device_t;
typedef struct scrapgpu_buffer scrapgpu_buffer_t;
typedef struct scrapgpu_stream scrapgpu_stream_t;
typedef struct scrapgpu_plan scrapgpu_plan_t;

SCRAPGPU_API uint32_t scrapgpu_get_abi_version(void);

#ifdef __cplusplus
}
#endif

#endif
