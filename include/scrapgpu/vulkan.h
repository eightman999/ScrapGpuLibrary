#ifndef SCRAPGPU_VULKAN_H
#define SCRAPGPU_VULKAN_H

#include "scrapgpu/api.h"
#include "scrapgpu/status.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers the Vulkan backend (src/backends/vulkan): Vulkan 1.2 devices with 16-bit storage,
// filtered by SCRAPGPU_VULKAN_DEVICES (comma-separated substrings of the device name). NVIDIA
// devices are skipped unless SCRAPGPU_VULKAN_ALLOW_NVIDIA=1 (the 390 driver stack crashes the
// ICD). Safe to call more than once; when no Vulkan loader or no Vulkan 1.2 instance exists,
// registration succeeds with 0 devices.
SCRAPGPU_API scrapgpu_status scrapgpu_vulkan_register(void);

#ifdef __cplusplus
}
#endif

#endif
