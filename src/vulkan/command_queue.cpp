#include <soar/vulkan/command_queue.hpp>
#include <soar/core/logging.hpp>
#include <soar/core/error.hpp>

namespace soar::vk {

CommandQueue::CommandQueue(VulkanContext& ctx) : ctx_(&ctx) {
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = 0;

    VkResult res = ctx_->loader().vkCreateFence(ctx_->device(), &fence_info, nullptr, &fence_);
    if (res != VK_SUCCESS) {
        throw DeviceError("Failed to create VkFence for CommandQueue. VkResult: " + std::to_string(res));
    }
}

CommandQueue::~CommandQueue() {
    cleanup();
}

CommandQueue::CommandQueue(CommandQueue&& other) noexcept {
    *this = std::move(other);
}

CommandQueue& CommandQueue::operator=(CommandQueue&& other) noexcept {
    if (this != &other) {
        cleanup();
        ctx_ = other.ctx_;
        fence_ = other.fence_;

        other.ctx_ = nullptr;
        other.fence_ = VK_NULL_HANDLE;
    }
    return *this;
}

void CommandQueue::cleanup() noexcept {
    if (ctx_ && fence_ != VK_NULL_HANDLE) {
        ctx_->loader().vkDestroyFence(ctx_->device(), fence_, nullptr);
        fence_ = VK_NULL_HANDLE;
    }
    ctx_ = nullptr;
}

VkCommandBuffer CommandQueue::allocate_command_buffer() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = ctx_->command_pool();
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkResult res = ctx_->loader().vkAllocateCommandBuffers(ctx_->device(), &alloc_info, &cmd);
    if (res != VK_SUCCESS) {
        throw DeviceError("Failed to allocate VkCommandBuffer. VkResult: " + std::to_string(res));
    }
    return cmd;
}

void CommandQueue::free_command_buffer(VkCommandBuffer cmd) {
    if (cmd != VK_NULL_HANDLE && ctx_) {
        ctx_->loader().vkFreeCommandBuffers(ctx_->device(), ctx_->command_pool(), 1, &cmd);
    }
}

void CommandQueue::execute_sync(const std::function<void(VkCommandBuffer)>& record_fn) {
    VkCommandBuffer cmd = allocate_command_buffer();

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult res = ctx_->loader().vkBeginCommandBuffer(cmd, &begin_info);
    if (res != VK_SUCCESS) {
        free_command_buffer(cmd);
        throw DeviceError("Failed to begin VkCommandBuffer. VkResult: " + std::to_string(res));
    }

    try {
        record_fn(cmd);
    } catch (...) {
        free_command_buffer(cmd);
        throw;
    }

    res = ctx_->loader().vkEndCommandBuffer(cmd);
    if (res != VK_SUCCESS) {
        free_command_buffer(cmd);
        throw DeviceError("Failed to end VkCommandBuffer. VkResult: " + std::to_string(res));
    }

    ctx_->loader().vkResetFences(ctx_->device(), 1, &fence_);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmd;

    res = ctx_->loader().vkQueueSubmit(ctx_->compute_queue(), 1, &submit_info, fence_);
    if (res != VK_SUCCESS) {
        free_command_buffer(cmd);
        throw DeviceError("Failed to submit command buffer to compute queue. VkResult: " + std::to_string(res));
    }

    // Wait for fence (timeout = 60 seconds)
    constexpr uint64_t TIMEOUT_NS = 60ULL * 1000ULL * 1000ULL * 1000ULL;
    res = ctx_->loader().vkWaitForFences(ctx_->device(), 1, &fence_, VK_TRUE, TIMEOUT_NS);
    if (res != VK_SUCCESS) {
        free_command_buffer(cmd);
        throw DeviceError("Timeout or failure waiting for compute fence. VkResult: " + std::to_string(res));
    }

    free_command_buffer(cmd);
}

void CommandQueue::memory_barrier(const VulkanContext& ctx,
                                 VkCommandBuffer cmd,
                                 const VulkanBuffer& buffer,
                                 VkAccessFlags src_access,
                                 VkAccessFlags dst_access) {
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.handle();
    barrier.offset = buffer.offset();
    barrier.size = buffer.size();

    VkPipelineStageFlags src_stages = 0;
    if (src_access & (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)) {
        src_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (src_access & (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT)) {
        src_stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    if (src_stages == 0) {
        src_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    VkPipelineStageFlags dst_stages = 0;
    if (dst_access & (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)) {
        dst_stages |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    if (dst_access & (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_UNIFORM_READ_BIT)) {
        dst_stages |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    }
    if (dst_stages == 0) {
        dst_stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    }

    ctx.loader().vkCmdPipelineBarrier(
        cmd,
        src_stages,
        dst_stages,
        0,
        0, nullptr,
        1, &barrier,
        0, nullptr
    );
}

void CommandQueue::global_compute_barrier(const VulkanContext& ctx, VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

    ctx.loader().vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1, &barrier,
        0, nullptr,
        0, nullptr
    );
}

} // namespace soar::vk
