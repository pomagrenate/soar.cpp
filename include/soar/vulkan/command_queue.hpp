#pragma once

#include <soar/vulkan/context.hpp>
#include <soar/vulkan/buffer.hpp>
#include <functional>

namespace soar::vk {

/**
 * @brief Manages Vulkan command buffers, synchronization barriers, and queue submissions.
 */
class CommandQueue {
public:
    explicit CommandQueue(VulkanContext& ctx);
    ~CommandQueue();

    // Non-copyable, movable
    CommandQueue(const CommandQueue&) = delete;
    CommandQueue& operator=(const CommandQueue&) = delete;
    CommandQueue(CommandQueue&& other) noexcept;
    CommandQueue& operator=(CommandQueue&& other) noexcept;

    /**
     * @brief Execute a sequence of commands synchronously with a fence wait.
     */
    void execute_sync(const std::function<void(VkCommandBuffer)>& record_fn);

    /**
     * @brief Record a compute-to-compute execution and memory dependency barrier for a buffer.
     */
    static void memory_barrier(const VulkanContext& ctx,
                               VkCommandBuffer cmd,
                               const VulkanBuffer& buffer,
                               VkAccessFlags src_access = VK_ACCESS_SHADER_WRITE_BIT,
                               VkAccessFlags dst_access = VK_ACCESS_SHADER_READ_BIT);

    /**
     * @brief Global compute execution barrier.
     */
    static void global_compute_barrier(const VulkanContext& ctx, VkCommandBuffer cmd);

    /**
     * @brief Allocate a single command buffer from the context's pool.
     */
    [[nodiscard]] VkCommandBuffer allocate_command_buffer();

    /**
     * @brief Free an allocated command buffer.
     */
    void free_command_buffer(VkCommandBuffer cmd);

private:
    void cleanup() noexcept;

    VulkanContext* ctx_{nullptr};
    VkFence fence_{VK_NULL_HANDLE};
};

} // namespace soar::vk
