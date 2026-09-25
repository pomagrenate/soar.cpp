#include <soar/vulkan/pipeline.hpp>
#include <soar/core/logging.hpp>
#include <soar/core/error.hpp>

namespace soar::vk {

ComputePipeline::ComputePipeline(VulkanContext& ctx,
                                 std::span<const uint32_t> spirv_code,
                                 uint32_t num_storage_buffers,
                                 uint32_t push_constant_size,
                                 const char* entry_point)
    : ctx_(&ctx), num_storage_buffers_(num_storage_buffers), push_constant_size_(push_constant_size) {
    if (spirv_code.empty()) {
        throw DeviceError("Cannot create ComputePipeline from empty SPIR-V code");
    }

    // 1. Create Shader Module
    VkShaderModuleCreateInfo module_info{};
    module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_info.codeSize = spirv_code.size_bytes();
    module_info.pCode = spirv_code.data();

    VkResult res = ctx_->loader().vkCreateShaderModule(ctx_->device(), &module_info, nullptr, &shader_module_);
    if (res != VK_SUCCESS) {
        throw DeviceError("Failed to create VkShaderModule. VkResult: " + std::to_string(res));
    }

    // 2. Create Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings(num_storage_buffers_);
    for (uint32_t i = 0; i < num_storage_buffers_; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    res = ctx_->loader().vkCreateDescriptorSetLayout(ctx_->device(), &layout_info, nullptr, &desc_layout_);
    if (res != VK_SUCCESS) {
        cleanup();
        throw DeviceError("Failed to create VkDescriptorSetLayout. VkResult: " + std::to_string(res));
    }

    // 3. Create Pipeline Layout
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size_;

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &desc_layout_;
    if (push_constant_size_ > 0) {
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
    }

    res = ctx_->loader().vkCreatePipelineLayout(ctx_->device(), &pipeline_layout_info, nullptr, &pipeline_layout_);
    if (res != VK_SUCCESS) {
        cleanup();
        throw DeviceError("Failed to create VkPipelineLayout. VkResult: " + std::to_string(res));
    }

    // 4. Create Compute Pipeline
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = shader_module_;
    pipeline_info.stage.pName = entry_point;
    pipeline_info.layout = pipeline_layout_;

    res = ctx_->loader().vkCreateComputePipelines(ctx_->device(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_);
    if (res != VK_SUCCESS) {
        cleanup();
        throw DeviceError("Failed to create VkComputePipeline. VkResult: " + std::to_string(res));
    }

    // 5. Create Descriptor Pool and allocate Set
    if (num_storage_buffers_ > 0) {
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = num_storage_buffers_;

        VkDescriptorPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;

        res = ctx_->loader().vkCreateDescriptorPool(ctx_->device(), &pool_info, nullptr, &desc_pool_);
        if (res != VK_SUCCESS) {
            cleanup();
            throw DeviceError("Failed to create VkDescriptorPool. VkResult: " + std::to_string(res));
        }

        VkDescriptorSetAllocateInfo set_info{};
        set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        set_info.descriptorPool = desc_pool_;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &desc_layout_;

        res = ctx_->loader().vkAllocateDescriptorSets(ctx_->device(), &set_info, &descriptor_set_);
        if (res != VK_SUCCESS) {
            cleanup();
            throw DeviceError("Failed to allocate VkDescriptorSet. VkResult: " + std::to_string(res));
        }
    }
}

ComputePipeline::~ComputePipeline() {
    cleanup();
}

ComputePipeline::ComputePipeline(ComputePipeline&& other) noexcept {
    *this = std::move(other);
}

ComputePipeline& ComputePipeline::operator=(ComputePipeline&& other) noexcept {
    if (this != &other) {
        cleanup();
        ctx_ = other.ctx_;
        shader_module_ = other.shader_module_;
        desc_layout_ = other.desc_layout_;
        desc_pool_ = other.desc_pool_;
        descriptor_set_ = other.descriptor_set_;
        pipeline_layout_ = other.pipeline_layout_;
        pipeline_ = other.pipeline_;
        num_storage_buffers_ = other.num_storage_buffers_;
        push_constant_size_ = other.push_constant_size_;

        other.ctx_ = nullptr;
        other.shader_module_ = VK_NULL_HANDLE;
        other.desc_layout_ = VK_NULL_HANDLE;
        other.desc_pool_ = VK_NULL_HANDLE;
        other.descriptor_set_ = VK_NULL_HANDLE;
        other.pipeline_layout_ = VK_NULL_HANDLE;
        other.pipeline_ = VK_NULL_HANDLE;
    }
    return *this;
}

void ComputePipeline::cleanup() noexcept {
    if (!ctx_) return;

    if (pipeline_ != VK_NULL_HANDLE) {
        ctx_->loader().vkDestroyPipeline(ctx_->device(), pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        ctx_->loader().vkDestroyPipelineLayout(ctx_->device(), pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (desc_pool_ != VK_NULL_HANDLE) {
        ctx_->loader().vkDestroyDescriptorPool(ctx_->device(), desc_pool_, nullptr);
        desc_pool_ = VK_NULL_HANDLE;
        descriptor_set_ = VK_NULL_HANDLE;
    }
    if (desc_layout_ != VK_NULL_HANDLE) {
        ctx_->loader().vkDestroyDescriptorSetLayout(ctx_->device(), desc_layout_, nullptr);
        desc_layout_ = VK_NULL_HANDLE;
    }
    if (shader_module_ != VK_NULL_HANDLE) {
        ctx_->loader().vkDestroyShaderModule(ctx_->device(), shader_module_, nullptr);
        shader_module_ = VK_NULL_HANDLE;
    }
    ctx_ = nullptr;
}

void ComputePipeline::bind_buffers(std::span<const VulkanBuffer*> buffers) {
    if (buffers.size() != num_storage_buffers_) {
        throw DeviceError("Descriptor buffer count mismatch: expected " + std::to_string(num_storage_buffers_) + ", got " + std::to_string(buffers.size()));
    }

    std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
    std::vector<VkWriteDescriptorSet> writes(buffers.size());

    for (size_t i = 0; i < buffers.size(); ++i) {
        buffer_infos[i].buffer = buffers[i]->handle();
        buffer_infos[i].offset = buffers[i]->offset();
        buffer_infos[i].range = buffers[i]->size();

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].pNext = nullptr;
        writes[i].dstSet = descriptor_set_;
        writes[i].dstBinding = static_cast<uint32_t>(i);
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buffer_infos[i];
        writes[i].pImageInfo = nullptr;
        writes[i].pTexelBufferView = nullptr;
    }

    ctx_->loader().vkUpdateDescriptorSets(ctx_->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void ComputePipeline::record_dispatch(VkCommandBuffer cmd,
                                      uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                      const void* push_constants,
                                      uint32_t push_constant_size) {
    ctx_->loader().vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);

    if (descriptor_set_ != VK_NULL_HANDLE) {
        ctx_->loader().vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_, 0, nullptr);
    }

    if (push_constants && push_constant_size > 0) {
        ctx_->loader().vkCmdPushConstants(cmd, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size, push_constants);
    }

    ctx_->loader().vkCmdDispatch(cmd, group_x, group_y, group_z);
}

} // namespace soar::vk
