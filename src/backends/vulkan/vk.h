#ifndef SCRAPGPU_VULKAN_VK_H
#define SCRAPGPU_VULKAN_VK_H

#include "volk.h"

#include "tensor/tensor.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

// The internals of the Vulkan backend (src/backends/vulkan). Vulkan 1.2 + 16-bit storage:
// the shaders address quantized and F16 data as uint16[] and unpack the halves manually, so the
// only feature the device must have is storageBuffer16BitAccess. No link to libvulkan: volk loads
// it at run time (scrapgpu_vulkan_register succeeds with 0 devices when the loader is missing).

namespace sgv {

// shader (SPIR-V module) ids; must match the order of sg_vk_shader_table in the generated header
enum vk_shader_id {
    VK_S_ADD = 0,
    VK_S_MUL,
    VK_S_RMS_NORM,
    VK_S_MV_F32,
    VK_S_MV_F16,
    VK_S_MV_Q4_0,
    VK_S_MV_Q8_0,
    VK_S_MM_F32,
    VK_S_MM_F16,
    VK_S_MM_Q4_0,
    VK_S_MM_Q8_0,
    VK_S_COUNT
};

// a physical device that passed the enumeration filters
struct vk_phys {
    VkPhysicalDevice pdev = VK_NULL_HANDLE;
    std::string name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_version = 0;
    uint32_t driver_version = 0;
    int queue_family = -1;
    VkPhysicalDeviceProperties props = {};
    VkPhysicalDeviceMemoryProperties mem = {};
    uint32_t subgroup_size = 0;
    uint64_t max_allocation = 0;
    uint32_t compute_units = 0; // VK_AMD_shader_core_properties when present, else 0
};

// the logical device and everything built on it (lazy, guarded by mutex)
struct vk_dev {
    vk_phys phys;
    std::mutex mutex;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    uint32_t mt_device = 0; // memory type of the compute buffers (DEVICE_LOCAL preferred)
    uint32_t mt_host = 0;   // HOST_VISIBLE | HOST_COHERENT staging memory
    bool pipes_ready = false;
    VkDescriptorSetLayout ds_layout = VK_NULL_HANDLE;
    VkDescriptorPool ds_pool = VK_NULL_HANDLE;
    VkPipelineLayout pl_layout = VK_NULL_HANDLE;
    VkShaderModule modules[VK_S_COUNT] = {};
    VkPipeline pipes[VK_S_COUNT] = {};
};

struct vk_buf {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t size = 0;
};

// a host-visible buffer a stream stages the host<->device transfers through
struct vk_staging {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void * ptr = nullptr;
    size_t size = 0;
    bool in_flight = false;
};

struct vk_str {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    bool recording = false;
    std::vector<VkFence> fences;      // in-flight submissions, oldest first
    std::vector<VkDescriptorSet> sets; // allocated sets, freed on stream_sync
    vk_staging staging;
};

// the one instance shared by every device (volk loads the library; created once)
bool vk_instance_ready();
VkInstance vk_instance();

// physical devices that satisfy Vulkan 1.2 + 16-bit storage + a compute queue and pass the
// SCRAPGPU_VULKAN_DEVICES name filter and the NVIDIA skip rule
std::vector<vk_phys> vk_probe_devices();

// logical device, compute queue, command pool, memory type selection and the pipelines; false on
// failure (why is filled). Called once per device, guarded by its mutex.
bool vk_device_init(vk_dev * d, std::string & why);

bool vk_buf_create(vk_dev * d, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags, vk_buf * out);
void vk_buf_destroy(vk_dev * d, vk_buf * b);

// the mapped staging memory of a stream (grows on demand); waits for the submission that last used it
void * vk_staging_map(vk_dev * d, vk_str * s, size_t size);
void vk_staging_free(vk_dev * d, vk_str * s);

// command buffer recording: begin (lazy), submit with a fresh fence, wait every pending fence.
// On a successful submit the fence is pushed on s->fences; vk_stream_sync drains it, frees the
// descriptor sets and marks the staging free.
VkResult vk_record(vk_dev * d, vk_str * s);
VkResult vk_submit(vk_dev * d, vk_str * s);
VkResult vk_stream_sync(vk_dev * d, vk_str * s);
void vk_stream_free(vk_dev * d, vk_str * s);

// a full-range memory barrier between the commands recorded so far and the next one
void vk_barrier(vk_dev * d, vk_str * s);

// allocate a descriptor set (freed on the next stream_sync), bind the three buffers and record
// the dispatch + a barrier. Buffers may repeat; unused bindings get `dst`.
bool vk_dispatch(vk_dev * d, vk_str * s, int shader, const void * pc, uint32_t pc_size,
                 const vk_buf * a, const vk_buf * b, const vk_buf * dst,
                 uint32_t gx, uint32_t gy, uint32_t gz);

} // namespace sgv

#endif
