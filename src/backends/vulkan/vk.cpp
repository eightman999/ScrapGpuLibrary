// The internals of the Vulkan backend: one instance (Vulkan 1.2), physical devices filtered by
// SCRAPGPU_VULKAN_DEVICES and the NVIDIA rule, a logical device with one compute queue per
// scrapgpu device, device-local buffers with a host-visible staging buffer per stream, and a
// stream = a command buffer that is submitted with a fence per call (sync = wait).

#include "vk.h"

#include "vk-shaders.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace sgv {

namespace {

VkInstance g_instance = VK_NULL_HANDLE;
bool g_instance_tried = false;

const char * vk_result_str(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
        case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
        case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
        case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
        case VK_ERROR_UNKNOWN: return "VK_ERROR_UNKNOWN";
        default: return "VkResult";
    }
}

// the comma-separated substrings of SCRAPGPU_VULKAN_DEVICES (empty list = everything passes)
std::vector<std::string> device_filter() {
    std::vector<std::string> out;
    const char * env = getenv("SCRAPGPU_VULKAN_DEVICES");
    if (env == nullptr) {
        return out;
    }
    std::string s = env;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string item = s.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (!item.empty()) {
            out.push_back(item);
        }
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

bool env_flag(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && strcmp(v, "0") != 0 && v[0] != '\0';
}

bool query_phys(vk_phys & p) {
    vkGetPhysicalDeviceProperties(p.pdev, &p.props);
    p.name = p.props.deviceName;
    p.vendor_id = p.props.vendorID;
    p.device_id = p.props.deviceID;
    p.api_version = p.props.apiVersion;
    p.driver_version = p.props.driverVersion;
    if (p.api_version < VK_API_VERSION_1_2) {
        return false;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(p.pdev, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(p.pdev, &nq, qf.data());
    for (uint32_t i = 0; i < nq; ++i) {
        if (qf[i].queueCount > 0 && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            p.queue_family = (int) i;
            break;
        }
    }
    if (p.queue_family < 0) {
        return false;
    }

    VkPhysicalDeviceVulkan11Features f11 = {};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceFeatures2 f2 = {};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &f11;
    vkGetPhysicalDeviceFeatures2(p.pdev, &f2);
    if (f11.storageBuffer16BitAccess != VK_TRUE) {
        return false;
    }

    VkPhysicalDeviceShaderCorePropertiesAMD amd = {};
    amd.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
    VkPhysicalDeviceVulkan11Properties p11 = {};
    p11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    VkPhysicalDeviceProperties2 p2 = {};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &p11;

    // VK_AMD_shader_core_properties: a device extension; only chain it when the device exposes it
    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(p.pdev, nullptr, &next, nullptr);
    std::vector<VkExtensionProperties> exts(next);
    vkEnumerateDeviceExtensionProperties(p.pdev, nullptr, &next, exts.data());
    bool amd_ext = false;
    for (const auto & e : exts) {
        if (strcmp(e.extensionName, VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME) == 0) {
            amd_ext = true;
        }
    }
    if (amd_ext) {
        p11.pNext = &amd;
    }
    vkGetPhysicalDeviceProperties2(p.pdev, &p2);

    p.subgroup_size = p11.subgroupSize;
    p.max_allocation = p11.maxMemoryAllocationSize;
    p.compute_units = amd_ext ? amd.shaderEngineCount * amd.shaderArraysPerEngineCount * amd.computeUnitsPerShaderArray : 0;
    vkGetPhysicalDeviceMemoryProperties(p.pdev, &p.mem);
    return true;
}

} // namespace

bool vk_instance_ready() {
    if (g_instance_tried) {
        return g_instance != VK_NULL_HANDLE;
    }
    g_instance_tried = true;
    if (volkInitialize() != VK_SUCCESS) {
        return false;
    }
    uint32_t inst_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion != nullptr) {
        if (vkEnumerateInstanceVersion(&inst_version) != VK_SUCCESS) {
            inst_version = VK_API_VERSION_1_0;
        }
    }
    if (inst_version < VK_API_VERSION_1_2) {
        return false;
    }
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "scrapgpu";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &app;
    if (vkCreateInstance(&ci, nullptr, &g_instance) != VK_SUCCESS) {
        g_instance = VK_NULL_HANDLE;
        return false;
    }
    volkLoadInstance(g_instance);
    return true;
}

VkInstance vk_instance() {
    return g_instance;
}

std::vector<vk_phys> vk_probe_devices() {
    std::vector<vk_phys> out;
    if (!vk_instance_ready()) {
        return out;
    }
    const std::vector<std::string> filter = device_filter();
    const bool allow_nvidia = env_flag("SCRAPGPU_VULKAN_ALLOW_NVIDIA");

    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(g_instance, &n, nullptr) != VK_SUCCESS) {
        return out;
    }
    std::vector<VkPhysicalDevice> devs(n);
    if (vkEnumeratePhysicalDevices(g_instance, &n, devs.data()) != VK_SUCCESS) {
        return out;
    }
    for (VkPhysicalDevice pdev : devs) {
        vk_phys p;
        p.pdev = pdev;
        if (!query_phys(p)) {
            continue;
        }
        if (p.vendor_id == 0x10DE && !allow_nvidia) {
            continue; // the NVIDIA ICD of this machine crashes (HARDWARE.md)
        }
        if (!filter.empty()) {
            bool match = false;
            for (const std::string & f : filter) {
                if (p.name.find(f) != std::string::npos) {
                    match = true;
                }
            }
            if (!match) {
                continue;
            }
        }
        out.push_back(p);
    }
    return out;
}

bool vk_device_init(vk_dev * d, std::string & why) {
    if (d->dev != VK_NULL_HANDLE) {
        return true;
    }
    // any failure leaves the device torn down, so the next call retries from scratch
    struct fail {
        vk_dev * d;
        ~fail() {
            if (d == nullptr || d->dev == VK_NULL_HANDLE) {
                return;
            }
            for (VkPipeline p : d->pipes) {
                if (p != VK_NULL_HANDLE) {
                    vkDestroyPipeline(d->dev, p, nullptr);
                }
            }
            for (VkShaderModule m : d->modules) {
                if (m != VK_NULL_HANDLE) {
                    vkDestroyShaderModule(d->dev, m, nullptr);
                }
            }
            if (d->ds_pool != VK_NULL_HANDLE) {
                vkDestroyDescriptorPool(d->dev, d->ds_pool, nullptr);
            }
            if (d->pl_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(d->dev, d->pl_layout, nullptr);
            }
            if (d->ds_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(d->dev, d->ds_layout, nullptr);
            }
            if (d->pool != VK_NULL_HANDLE) {
                vkDestroyCommandPool(d->dev, d->pool, nullptr);
            }
            vkDestroyDevice(d->dev, nullptr);
            d->dev = VK_NULL_HANDLE;
            d->queue = VK_NULL_HANDLE;
            d->pool = VK_NULL_HANDLE;
            d->ds_layout = VK_NULL_HANDLE;
            d->ds_pool = VK_NULL_HANDLE;
            d->pl_layout = VK_NULL_HANDLE;
            d->pipes_ready = false;
            for (int i = 0; i < VK_S_COUNT; ++i) {
                d->modules[i] = VK_NULL_HANDLE;
                d->pipes[i] = VK_NULL_HANDLE;
            }
        }
    } cleanup{d};

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {};
    qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi.queueFamilyIndex = (uint32_t) d->phys.queue_family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &prio;

    VkPhysicalDeviceVulkan11Features f11 = {};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.storageBuffer16BitAccess = VK_TRUE;
    VkDeviceCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext = &f11;
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &qi;
    VkResult r = vkCreateDevice(d->phys.pdev, &ci, nullptr, &d->dev);
    if (r != VK_SUCCESS) {
        why = std::string("vkCreateDevice: ") + vk_result_str(r);
        return false;
    }
    // device functions stay on the pointers volkLoadInstance loaded from vkGetInstanceProcAddr:
    // those trampolines dispatch on the handle, which is correct for every device of the instance
    vkGetDeviceQueue(d->dev, (uint32_t) d->phys.queue_family, 0, &d->queue);

    VkCommandPoolCreateInfo pi = {};
    pi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pi.queueFamilyIndex = (uint32_t) d->phys.queue_family;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    r = vkCreateCommandPool(d->dev, &pi, nullptr, &d->pool);
    if (r != VK_SUCCESS) {
        why = std::string("vkCreateCommandPool: ") + vk_result_str(r);
        return false;
    }

    // memory types: DEVICE_LOCAL for compute (host-visible device memory allowed, plain
    // device-local preferred), HOST_VISIBLE | HOST_COHERENT for staging
    // (scan every type: the staging type can come after the first plain device-local one, e.g. RADV lists
    // DEVICE_LOCAL at 0 and HOST_VISIBLE | HOST_COHERENT at 2)
    int mt_device = -1, mt_host = -1;
    bool device_plain = false;
    for (uint32_t i = 0; i < d->phys.mem.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = d->phys.mem.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 && (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 && mt_host < 0) {
            mt_host = (int) i;
        }
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 && !device_plain) {
            if ((f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0) {
                mt_device = (int) i; // the preferred kind
                device_plain = true;
            } else if (mt_device < 0) {
                mt_device = (int) i;
            }
        }
    }
    if (mt_device < 0) {
        mt_device = mt_host; // unified memory (integrated GPUs)
    }
    if (mt_device < 0 || mt_host < 0) {
        why = "no usable memory type";
        return false;
    }
    d->mt_device = (uint32_t) mt_device;
    d->mt_host = (uint32_t) mt_host;

    // descriptor set layout: four storage buffers, one layout shared by every shader
    VkDescriptorSetLayoutBinding binds[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        binds[i].binding = i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli = {};
    dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dli.bindingCount = 4;
    dli.pBindings = binds;
    r = vkCreateDescriptorSetLayout(d->dev, &dli, nullptr, &d->ds_layout);
    if (r != VK_SUCCESS) {
        why = std::string("vkCreateDescriptorSetLayout: ") + vk_result_str(r);
        return false;
    }

    VkPushConstantRange pcr = {};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0;
    pcr.size = 128;
    VkPipelineLayoutCreateInfo pli = {};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &d->ds_layout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    r = vkCreatePipelineLayout(d->dev, &pli, nullptr, &d->pl_layout);
    if (r != VK_SUCCESS) {
        why = std::string("vkCreatePipelineLayout: ") + vk_result_str(r);
        return false;
    }

    VkDescriptorPoolSize ps = {};
    ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ps.descriptorCount = 4 * 512;
    VkDescriptorPoolCreateInfo dpi = {};
    dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 512;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    r = vkCreateDescriptorPool(d->dev, &dpi, nullptr, &d->ds_pool);
    if (r != VK_SUCCESS) {
        why = std::string("vkCreateDescriptorPool: ") + vk_result_str(r);
        return false;
    }

    if (sg_vk_shader_table[0].code == nullptr) {
        // the shaders were not built (no glslc at configure time): the device still works for
        // buffers and transfers; op_supported reports "shaders not built"
        cleanup.d = nullptr;
        return true;
    }
    for (int i = 0; i < VK_S_COUNT; ++i) {
        VkShaderModuleCreateInfo smi = {};
        smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smi.codeSize = sg_vk_shader_table[i].size;
        smi.pCode = sg_vk_shader_table[i].code;
        r = vkCreateShaderModule(d->dev, &smi, nullptr, &d->modules[i]);
        if (r != VK_SUCCESS) {
            why = std::string("vkCreateShaderModule(") + sg_vk_shader_table[i].name + "): " + vk_result_str(r);
            return false;
        }
    }
    for (int i = 0; i < VK_S_COUNT; ++i) {
        VkComputePipelineCreateInfo pci = {};
        pci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pci.stage.module = d->modules[i];
        pci.stage.pName = "main";
        pci.layout = d->pl_layout;
        r = vkCreateComputePipelines(d->dev, VK_NULL_HANDLE, 1, &pci, nullptr, &d->pipes[i]);
        if (r != VK_SUCCESS) {
            why = std::string("vkCreateComputePipelines(") + sg_vk_shader_table[i].name + "): " + vk_result_str(r);
            return false;
        }
    }
    d->pipes_ready = true;
    cleanup.d = nullptr;
    return true;
}

bool vk_buf_create(vk_dev * d, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags, vk_buf * out) {
    VkBufferCreateInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = (VkDeviceSize) size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(d->dev, &bi, nullptr, &out->buf) != VK_SUCCESS) {
        return false;
    }
    VkMemoryRequirements req = {};
    vkGetBufferMemoryRequirements(d->dev, out->buf, &req);
    uint32_t type_index = d->mt_device;
    if ((mem_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        type_index = d->mt_host;
    }
    if ((req.memoryTypeBits & (1u << type_index)) == 0) {
        // the buffer may not accept the preferred type: fall back to the first fitting one
        bool found = false;
        for (uint32_t i = 0; i < d->phys.mem.memoryTypeCount && !found; ++i) {
            const VkMemoryPropertyFlags f = d->phys.mem.memoryTypes[i].propertyFlags;
            if ((req.memoryTypeBits & (1u << i)) != 0 && (f & mem_flags) == mem_flags) {
                type_index = i;
                found = true;
            }
        }
        if (!found) {
            vkDestroyBuffer(d->dev, out->buf, nullptr);
            return false;
        }
    }
    VkMemoryAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = type_index;
    if (vkAllocateMemory(d->dev, &ai, nullptr, &out->mem) != VK_SUCCESS) {
        vkDestroyBuffer(d->dev, out->buf, nullptr);
        return false;
    }
    if (vkBindBufferMemory(d->dev, out->buf, out->mem, 0) != VK_SUCCESS) {
        vkFreeMemory(d->dev, out->mem, nullptr);
        vkDestroyBuffer(d->dev, out->buf, nullptr);
        return false;
    }
    out->size = size;
    return true;
}

void vk_buf_destroy(vk_dev * d, vk_buf * b) {
    if (b->buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(d->dev, b->buf, nullptr);
    }
    if (b->mem != VK_NULL_HANDLE) {
        vkFreeMemory(d->dev, b->mem, nullptr);
    }
    *b = vk_buf();
}

void * vk_staging_map(vk_dev * d, vk_str * s, size_t size) {
    if (s->staging.in_flight) {
        if (vk_stream_sync(d, s) != VK_SUCCESS) {
            return nullptr;
        }
    }
    if (s->staging.ptr != nullptr && s->staging.size >= size) {
        return s->staging.ptr;
    }
    if (s->staging.buf != VK_NULL_HANDLE) {
        vkUnmapMemory(d->dev, s->staging.mem);
        vk_buf b = { s->staging.buf, s->staging.mem, s->staging.size };
        vk_buf_destroy(d, &b);
    }
    vk_buf st = {};
    if (!vk_buf_create(d, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &st)) {
        s->staging = vk_staging();
        return nullptr;
    }
    s->staging.buf = st.buf;
    s->staging.mem = st.mem;
    s->staging.size = size;
    if (vkMapMemory(d->dev, s->staging.mem, 0, size, 0, &s->staging.ptr) != VK_SUCCESS) {
        vk_staging_free(d, s);
        return nullptr;
    }
    return s->staging.ptr;
}

void vk_staging_free(vk_dev * d, vk_str * s) {
    if (s->staging.ptr != nullptr) {
        vkUnmapMemory(d->dev, s->staging.mem);
        s->staging.ptr = nullptr;
    }
    if (s->staging.buf != VK_NULL_HANDLE) {
        vk_buf b = { s->staging.buf, s->staging.mem, s->staging.size };
        vk_buf_destroy(d, &b);
    }
    s->staging = vk_staging();
}

VkResult vk_record(vk_dev * d, vk_str * s) {
    if (s->recording) {
        return VK_SUCCESS;
    }
    if (s->cmd == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = d->pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        const VkResult r = vkAllocateCommandBuffers(d->dev, &ai, &s->cmd);
        if (r != VK_SUCCESS) {
            return r;
        }
    } else {
        vkResetCommandBuffer(s->cmd, 0);
    }
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult r = vkBeginCommandBuffer(s->cmd, &bi);
    s->recording = r == VK_SUCCESS;
    return r;
}

VkResult vk_submit(vk_dev * d, vk_str * s) {
    if (!s->recording) {
        return VK_SUCCESS;
    }
    VkResult r = vkEndCommandBuffer(s->cmd);
    if (r != VK_SUCCESS) {
        return r;
    }
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fi = {};
    fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    r = vkCreateFence(d->dev, &fi, nullptr, &fence);
    if (r != VK_SUCCESS) {
        return r;
    }
    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &s->cmd;
    r = vkQueueSubmit(d->queue, 1, &si, fence);
    if (r != VK_SUCCESS) {
        vkDestroyFence(d->dev, fence, nullptr);
        return r;
    }
    s->fences.push_back(fence);
    s->recording = false;
    return VK_SUCCESS;
}

VkResult vk_stream_sync(vk_dev * d, vk_str * s) {
    const VkResult sub = vk_submit(d, s);
    if (sub != VK_SUCCESS) {
        return sub;
    }
    VkResult r = VK_SUCCESS;
    if (!s->fences.empty()) {
        r = vkWaitForFences(d->dev, (uint32_t) s->fences.size(), s->fences.data(), VK_TRUE, UINT64_MAX);
        for (VkFence f : s->fences) {
            vkDestroyFence(d->dev, f, nullptr);
        }
        s->fences.clear();
    }
    for (VkDescriptorSet set : s->sets) {
        vkFreeDescriptorSets(d->dev, d->ds_pool, 1, &set);
    }
    s->sets.clear();
    s->staging.in_flight = false;
    return r;
}

void vk_stream_free(vk_dev * d, vk_str * s) {
    vk_stream_sync(d, s);
    vk_staging_free(d, s);
    if (s->cmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(d->dev, d->pool, 1, &s->cmd);
        s->cmd = VK_NULL_HANDLE;
    }
}

void vk_barrier(vk_dev * d, vk_str * s) {
    (void) d;
    VkMemoryBarrier mb = {};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(s->cmd,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         1, &mb, 0, nullptr, 0, nullptr);
}

bool vk_dispatch(vk_dev * d, vk_str * s, int shader, const void * pc, uint32_t pc_size,
                 const vk_buf * a, const vk_buf * b, const vk_buf * dst,
                 uint32_t gx, uint32_t gy, uint32_t gz) {
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai = {};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = d->ds_pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &d->ds_layout;
    if (vkAllocateDescriptorSets(d->dev, &ai, &set) != VK_SUCCESS) {
        // the pool is drained only on stream_sync: sync once and retry (the sync submits and
        // closes the pending command buffer, so recording has to be restarted)
        if (vk_stream_sync(d, s) != VK_SUCCESS || vk_record(d, s) != VK_SUCCESS) {
            return false;
        }
        if (vkAllocateDescriptorSets(d->dev, &ai, &set) != VK_SUCCESS) {
            return false;
        }
    }
    s->sets.push_back(set);

    VkDescriptorBufferInfo infos[4];
    const vk_buf * bufs[4] = { a, b, dst, dst };
    for (int i = 0; i < 4; ++i) {
        infos[i].buffer = bufs[i]->buf;
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
    }
    VkWriteDescriptorSet wr = {};
    wr.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    wr.dstSet = set;
    wr.dstBinding = 0;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    wr.descriptorCount = 4;
    wr.pBufferInfo = infos;
    vkUpdateDescriptorSets(d->dev, 1, &wr, 0, nullptr);

    vkCmdBindPipeline(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipes[shader]);
    vkCmdBindDescriptorSets(s->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, d->pl_layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(s->cmd, d->pl_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_size, pc);
    vkCmdDispatch(s->cmd, gx, gy, gz);
    vk_barrier(d, s);
    return true;
}

} // namespace sgv
