#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif

#include <vulkan/vulkan.h>
#include <soar/core/types.hpp>
#include <soar/core/error.hpp>
#include <string>
#include <memory>

namespace soar::vk {

/**
 * @brief Dynamic loader for Vulkan API functions without compile-time/link-time library dependency.
 * 
 * Dynamically loads vulkan-1.dll (Windows), libvulkan.so.1 (Linux), or libvulkan.dylib (macOS),
 * queries vkGetInstanceProcAddr, and binds all instance- and device-level function pointers.
 */
class DynamicLoader {
public:
    DynamicLoader();
    ~DynamicLoader();

    // Non-copyable, movable
    DynamicLoader(const DynamicLoader&) = delete;
    DynamicLoader& operator=(const DynamicLoader&) = delete;
    DynamicLoader(DynamicLoader&& other) noexcept;
    DynamicLoader& operator=(DynamicLoader&& other) noexcept;

    /**
     * @brief Load Vulkan library and resolve entry point vkGetInstanceProcAddr.
     * @param custom_lib_path Optional explicit path to Vulkan shared library.
     */
    void load_library(const std::string& custom_lib_path = "");

    /**
     * @brief Check if Vulkan library is successfully loaded.
     */
    [[nodiscard]] bool is_loaded() const noexcept { return library_handle_ != nullptr; }

    /**
     * @brief Load instance-level functions once VkInstance is created.
     */
    void load_instance_functions(VkInstance instance);

    /**
     * @brief Load device-level functions once VkDevice is created.
     */
    void load_device_functions(VkDevice device);

    /**
     * @brief Automatically configure SwiftShader ICD fallback if hardware driver is not registered.
     */
    static void configure_swiftshader_fallback();

    // Entry point
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr{nullptr};

    // Global / Pre-instance
    PFN_vkCreateInstance vkCreateInstance{nullptr};
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties{nullptr};
    PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties{nullptr};
    PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion{nullptr};

    // Instance-level
    PFN_vkDestroyInstance vkDestroyInstance{nullptr};
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices{nullptr};
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties{nullptr};
    PFN_vkGetPhysicalDeviceFeatures vkGetPhysicalDeviceFeatures{nullptr};
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties{nullptr};
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties{nullptr};
    PFN_vkCreateDevice vkCreateDevice{nullptr};
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr{nullptr};
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties{nullptr};

    // Device-level
    PFN_vkDestroyDevice vkDestroyDevice{nullptr};
    PFN_vkGetDeviceQueue vkGetDeviceQueue{nullptr};
    PFN_vkQueueSubmit vkQueueSubmit{nullptr};
    PFN_vkQueueWaitIdle vkQueueWaitIdle{nullptr};
    PFN_vkDeviceWaitIdle vkDeviceWaitIdle{nullptr};

    PFN_vkCreateBuffer vkCreateBuffer{nullptr};
    PFN_vkDestroyBuffer vkDestroyBuffer{nullptr};
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements{nullptr};
    PFN_vkAllocateMemory vkAllocateMemory{nullptr};
    PFN_vkFreeMemory vkFreeMemory{nullptr};
    PFN_vkBindBufferMemory vkBindBufferMemory{nullptr};
    PFN_vkMapMemory vkMapMemory{nullptr};
    PFN_vkUnmapMemory vkUnmapMemory{nullptr};
    PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges{nullptr};
    PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges{nullptr};

    PFN_vkCreateShaderModule vkCreateShaderModule{nullptr};
    PFN_vkDestroyShaderModule vkDestroyShaderModule{nullptr};
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout{nullptr};
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout{nullptr};
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout{nullptr};
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout{nullptr};
    PFN_vkCreateComputePipelines vkCreateComputePipelines{nullptr};
    PFN_vkDestroyPipeline vkDestroyPipeline{nullptr};

    PFN_vkCreateDescriptorPool vkCreateDescriptorPool{nullptr};
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool{nullptr};
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets{nullptr};
    PFN_vkFreeDescriptorSets vkFreeDescriptorSets{nullptr};
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets{nullptr};

    PFN_vkCreateCommandPool vkCreateCommandPool{nullptr};
    PFN_vkDestroyCommandPool vkDestroyCommandPool{nullptr};
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers{nullptr};
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers{nullptr};
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer{nullptr};
    PFN_vkEndCommandBuffer vkEndCommandBuffer{nullptr};
    PFN_vkResetCommandBuffer vkResetCommandBuffer{nullptr};

    PFN_vkCmdBindPipeline vkCmdBindPipeline{nullptr};
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets{nullptr};
    PFN_vkCmdPushConstants vkCmdPushConstants{nullptr};
    PFN_vkCmdDispatch vkCmdDispatch{nullptr};
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier{nullptr};
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer{nullptr};

    PFN_vkCreateFence vkCreateFence{nullptr};
    PFN_vkDestroyFence vkDestroyFence{nullptr};
    PFN_vkWaitForFences vkWaitForFences{nullptr};
    PFN_vkResetFences vkResetFences{nullptr};

private:
    void* library_handle_{nullptr};
    void unload();
};

} // namespace soar::vk
