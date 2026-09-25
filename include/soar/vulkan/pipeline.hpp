#pragma once

#include <soar/vulkan/context.hpp>
#include <soar/vulkan/buffer.hpp>
#include <span>
#include <vector>

namespace soar::vk {

struct PushConstantRangeConfig {
    uint32_t offset{0};
    uint32_t size{0};
};

/**
 * @brief High-performance Compute Pipeline wrapper for SPIR-V kernels.
 */
class ComputePipeline {
public:
    ComputePipeline(VulkanContext& ctx,
                    std::span<const uint32_t> spirv_code,
                    uint32_t num_storage_buffers,
                    uint32_t push_constant_size = 0,
                    const char* entry_point = "main");

    ~ComputePipeline();

    // Non-copyable, movable
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;
    ComputePipeline(ComputePipeline&& other) noexcept;
    ComputePipeline& operator=(ComputePipeline&& other) noexcept;

    [[nodiscard]] VkPipeline handle() const noexcept { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const noexcept { return pipeline_layout_; }
    [[nodiscard]] VkDescriptorSet descriptor_set() const noexcept { return descriptor_set_; }

    /**
     * @brief Bind storage buffers to descriptor set.
     */
    void bind_buffers(std::span<const VulkanBuffer*> buffers);

    /**
     * @brief Record pipeline bind, descriptor set bind, push constants, and dispatch into command buffer.
     */
    void record_dispatch(VkCommandBuffer cmd,
                         uint32_t group_x, uint32_t group_y, uint32_t group_z,
                         const void* push_constants = nullptr,
                         uint32_t push_constant_size = 0);

private:
    void cleanup() noexcept;

    VulkanContext* ctx_{nullptr};
    VkShaderModule shader_module_{VK_NULL_HANDLE};
    VkDescriptorSetLayout desc_layout_{VK_NULL_HANDLE};
    VkDescriptorPool desc_pool_{VK_NULL_HANDLE};
    VkDescriptorSet descriptor_set_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline pipeline_{VK_NULL_HANDLE};
    uint32_t num_storage_buffers_{0};
    uint32_t push_constant_size_{0};
};

} // namespace soar::vk
