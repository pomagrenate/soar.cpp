#pragma once

#include <soar/vulkan/dynamic_loader.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace soar::vk {

struct DeviceInfo {
    std::string device_name;
    uint32_t vendor_id{0};
    uint32_t device_id{0};
    VkPhysicalDeviceType device_type{VK_PHYSICAL_DEVICE_TYPE_OTHER};
    uint32_t api_version{0};
    uint32_t driver_version{0};
    size_t total_device_memory{0};
    size_t total_host_memory{0};
    uint32_t max_workgroup_invocations{0};
    uint32_t max_workgroup_size[3]{0, 0, 0};
    uint32_t max_compute_shared_memory_size{0};
};

/**
 * @brief Manages headless Vulkan instance, physical device, logical device, and compute queues.
 */
class VulkanContext {
public:
    explicit VulkanContext(bool enable_validation_layers = false, bool allow_cpu_fallback = false);
    ~VulkanContext();

    // Non-copyable, movable
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&& other) noexcept;
    VulkanContext& operator=(VulkanContext&& other) noexcept;

    [[nodiscard]] DynamicLoader& loader() noexcept { return loader_; }
    [[nodiscard]] const DynamicLoader& loader() const noexcept { return loader_; }

    [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const noexcept { return physical_device_; }
    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] VkQueue compute_queue() const noexcept { return compute_queue_; }
    [[nodiscard]] uint32_t compute_queue_family_index() const noexcept { return compute_queue_family_index_; }
    [[nodiscard]] VkCommandPool command_pool() const noexcept { return command_pool_; }

    [[nodiscard]] const DeviceInfo& device_info() const noexcept { return device_info_; }

    /**
     * @brief Find appropriate memory type index for allocation requirements.
     */
    [[nodiscard]] uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const;

    /**
     * @brief Block until all pending queue work is completed.
     */
    void wait_idle() const;

private:
    void init_instance(bool enable_validation);
    void pick_physical_device();
    void init_logical_device();
    void init_command_pool();
    void cleanup() noexcept;

    DynamicLoader loader_;
    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue compute_queue_{VK_NULL_HANDLE};
    uint32_t compute_queue_family_index_{0};
    VkCommandPool command_pool_{VK_NULL_HANDLE};

    VkPhysicalDeviceMemoryProperties memory_properties_{};
    DeviceInfo device_info_{};
    bool allow_cpu_fallback_{false};
};

} // namespace soar::vk
