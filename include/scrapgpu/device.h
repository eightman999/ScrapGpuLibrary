#ifndef SCRAPGPU_DEVICE_H
#define SCRAPGPU_DEVICE_H

#include "scrapgpu/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum scrapgpu_device_generation {
    SCRAPGPU_DEVICE_GENERATION_NV_FERMI = 0,
    SCRAPGPU_DEVICE_GENERATION_NV_KEPLER = 1,
    SCRAPGPU_DEVICE_GENERATION_AMD_GCN = 2,
    SCRAPGPU_DEVICE_GENERATION_AMD_RDNA = 3,
    SCRAPGPU_DEVICE_GENERATION_APPLE = 4,
    SCRAPGPU_DEVICE_GENERATION_UNKNOWN = 5,
} scrapgpu_device_generation;

enum scrapgpu_device_quirks {
    SCRAPGPU_DEVICE_QUIRK_LAZY_ALLOC = 1u << 0,
    SCRAPGPU_DEVICE_QUIRK_ALLOC_ABOVE_MAX_OK = 1u << 1,
    SCRAPGPU_DEVICE_QUIRK_CONST_SUBBUFFER_OFFSET_BROKEN = 1u << 2,
    SCRAPGPU_DEVICE_QUIRK_SUBBUFFER_RELEASE_ORDER = 1u << 3,
    SCRAPGPU_DEVICE_QUIRK_FMA_EMULATED = 1u << 4,
    SCRAPGPU_DEVICE_QUIRK_LAUNCH_COST_GROWS_WITH_PROGRAM = 1u << 5,
    SCRAPGPU_DEVICE_QUIRK_DENORMALS_FLUSHED = 1u << 6,
};

typedef struct scrapgpu_device_info {
    size_t struct_size;
    const char * name;
    const char * vendor;
    const char * platform_name;
    const char * driver_version;
    const char * backend_name;
    const char * reported_api_version;
    const char * used_api_version;
    scrapgpu_device_generation generation;
    const char * generation_reason;
    uint64_t total_memory_bytes;
    uint64_t max_allocation_reported_bytes;
    uint64_t max_allocation_used_bytes;
    uint64_t max_work_group_size;
    uint64_t local_memory_bytes;
    uint32_t compute_units;
    uint32_t warp_or_wave_width;
    uint64_t quirks;
    uint64_t probed_quirks;
    const char * tuning_profile_name;
    const char * tuning_overrides;
} scrapgpu_device_info;

SCRAPGPU_API scrapgpu_status scrapgpu_device_info_init(scrapgpu_device_info * info, size_t struct_size);
SCRAPGPU_API size_t scrapgpu_get_device_count(void);
SCRAPGPU_API scrapgpu_status scrapgpu_get_device(size_t index, scrapgpu_device_t ** out_device);
SCRAPGPU_API scrapgpu_status scrapgpu_device_get_info(scrapgpu_device_t * device, scrapgpu_device_info * info);

#ifdef __cplusplus
}
#endif

#endif
