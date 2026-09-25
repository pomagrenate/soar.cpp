#pragma once

#include <soar/vulkan/context.hpp>
#include <soar/core/types.hpp>
#include <span>
#include <memory>

namespace soar::vk {

enum class BufferUsageType {
    DeviceStorage,  // High-performance GPU local memory for compute shader I/O
    StagingHost,    // Host-visible, coherent staging memory for uploads/downloads
    UniformParam    // Read-only parameters/weights
};

/**
 * @brief RAII wrapper for Vulkan buffer and associated device memory.
 * 
 * Supports both directly allocated buffers and slices/views into pre-allocated memory slabs.
 */
class VulkanBuffer {
public:
    VulkanBuffer(VulkanContext& ctx, size_t size_bytes, BufferUsageType usage_type);
    
    // View/slice constructor for sub-allocation
    VulkanBuffer(VulkanContext& ctx, VkBuffer shared_buffer, size_t offset, size_t size_bytes);

    ~VulkanBuffer();

    // Non-copyable, movable
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceMemory memory() const noexcept { return memory_; }
    [[nodiscard]] size_t size() const noexcept { return size_bytes_; }
    [[nodiscard]] size_t offset() const noexcept { return offset_; }
    [[nodiscard]] bool is_view() const noexcept { return is_view_; }

    /**
     * @brief Map host-visible buffer into host address space.
     */
    void* map();
    void unmap();

    /**
     * @brief Copy data from host memory into this buffer (must be mapped or staging).
     */
    void upload_host(const void* src_data, size_t bytes, size_t dst_offset = 0);

    /**
     * @brief Copy data from this buffer into host memory (must be mapped or staging).
     */
    void download_host(void* dst_data, size_t bytes, size_t src_offset = 0);

    /**
     * @brief Asynchronous copy from another buffer using a command buffer.
     */
    void copy_from(VkCommandBuffer cmd, const VulkanBuffer& src, size_t bytes, size_t src_offset = 0, size_t dst_offset = 0);

private:
    void cleanup() noexcept;

    VulkanContext* ctx_{nullptr};
    VkBuffer buffer_{VK_NULL_HANDLE};
    VkDeviceMemory memory_{VK_NULL_HANDLE};
    size_t size_bytes_{0};
    size_t offset_{0};
    void* mapped_ptr_{nullptr};
    bool is_view_{false};
};

} // namespace soar::vk
