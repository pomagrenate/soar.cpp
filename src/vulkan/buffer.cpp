#include <soar/vulkan/buffer.hpp>
#include <soar/core/logging.hpp>
#include <soar/core/error.hpp>
#include <cstring>

namespace soar::vk {

VulkanBuffer::VulkanBuffer(VulkanContext& ctx, size_t size_bytes, BufferUsageType usage_type)
    : ctx_(&ctx), size_bytes_(size_bytes), offset_(0), is_view_(false) {
    if (size_bytes == 0) {
        throw MemoryError("Cannot allocate VulkanBuffer of 0 bytes");
    }

    VkBufferCreateInfo buf_info{};
    buf_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buf_info.size = size_bytes;
    buf_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkMemoryPropertyFlags mem_flags = 0;

    switch (usage_type) {
        case BufferUsageType::DeviceStorage:
            buf_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        case BufferUsageType::StagingHost:
            buf_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case BufferUsageType::UniformParam:
            buf_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            mem_flags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
    }

    VkResult res = ctx_->loader().vkCreateBuffer(ctx_->device(), &buf_info, nullptr, &buffer_);
    if (res != VK_SUCCESS) {
        throw DeviceError("Failed to create VkBuffer. VkResult: " + std::to_string(res));
    }

    VkMemoryRequirements mem_reqs;
    ctx_->loader().vkGetBufferMemoryRequirements(ctx_->device(), buffer_, &mem_reqs);

    uint32_t mem_type_index = 0;
    try {
        mem_type_index = ctx_->find_memory_type(mem_reqs.memoryTypeBits, mem_flags);
    } catch (const MemoryError&) {
        // Fallback for software rasterizers or unified memory where DEVICE_LOCAL might not be separate
        mem_type_index = ctx_->find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_index;

    res = ctx_->loader().vkAllocateMemory(ctx_->device(), &alloc_info, nullptr, &memory_);
    if (res != VK_SUCCESS) {
        ctx_->loader().vkDestroyBuffer(ctx_->device(), buffer_, nullptr);
        buffer_ = VK_NULL_HANDLE;
        throw MemoryError("Failed to allocate VkDeviceMemory of " + std::to_string(mem_reqs.size) + " bytes. VkResult: " + std::to_string(res));
    }

    res = ctx_->loader().vkBindBufferMemory(ctx_->device(), buffer_, memory_, 0);
    if (res != VK_SUCCESS) {
        cleanup();
        throw DeviceError("Failed to bind VkBuffer to memory. VkResult: " + std::to_string(res));
    }
}

VulkanBuffer::VulkanBuffer(VulkanContext& ctx, VkBuffer shared_buffer, size_t offset, size_t size_bytes)
    : ctx_(&ctx), buffer_(shared_buffer), memory_(VK_NULL_HANDLE), size_bytes_(size_bytes), offset_(offset), is_view_(true) {
}

VulkanBuffer::~VulkanBuffer() {
    cleanup();
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
    if (this != &other) {
        cleanup();
        ctx_ = other.ctx_;
        buffer_ = other.buffer_;
        memory_ = other.memory_;
        size_bytes_ = other.size_bytes_;
        offset_ = other.offset_;
        mapped_ptr_ = other.mapped_ptr_;
        is_view_ = other.is_view_;

        other.ctx_ = nullptr;
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
        other.mapped_ptr_ = nullptr;
    }
    return *this;
}

void VulkanBuffer::cleanup() noexcept {
    if (!ctx_) return;

    if (mapped_ptr_) {
        unmap();
    }

    if (!is_view_) {
        if (buffer_ != VK_NULL_HANDLE) {
            ctx_->loader().vkDestroyBuffer(ctx_->device(), buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
        }
        if (memory_ != VK_NULL_HANDLE) {
            ctx_->loader().vkFreeMemory(ctx_->device(), memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
        }
    }
    ctx_ = nullptr;
}

void* VulkanBuffer::map() {
    if (is_view_) {
        throw DeviceError("Cannot map a buffer view directly");
    }
    if (mapped_ptr_) {
        return mapped_ptr_;
    }

    VkResult res = ctx_->loader().vkMapMemory(ctx_->device(), memory_, 0, size_bytes_, 0, &mapped_ptr_);
    if (res != VK_SUCCESS) {
        throw DeviceError("Failed to map VkDeviceMemory. VkResult: " + std::to_string(res));
    }
    return mapped_ptr_;
}

void VulkanBuffer::unmap() {
    if (mapped_ptr_ && !is_view_) {
        ctx_->loader().vkUnmapMemory(ctx_->device(), memory_);
        mapped_ptr_ = nullptr;
    }
}

void VulkanBuffer::upload_host(const void* src_data, size_t bytes, size_t dst_offset) {
    if (dst_offset + bytes > size_bytes_) {
        throw MemoryError("Out of bounds buffer upload");
    }
    void* ptr = map();
    std::memcpy(static_cast<char*>(ptr) + dst_offset, src_data, bytes);
}

void VulkanBuffer::download_host(void* dst_data, size_t bytes, size_t src_offset) {
    if (src_offset + bytes > size_bytes_) {
        throw MemoryError("Out of bounds buffer download");
    }
    void* ptr = map();
    std::memcpy(dst_data, static_cast<const char*>(ptr) + src_offset, bytes);
}

void VulkanBuffer::copy_from(VkCommandBuffer cmd, const VulkanBuffer& src, size_t bytes, size_t src_offset, size_t dst_offset) {
    VkBufferCopy copy_region{};
    copy_region.srcOffset = src.offset() + src_offset;
    copy_region.dstOffset = offset_ + dst_offset;
    copy_region.size = bytes;

    ctx_->loader().vkCmdCopyBuffer(cmd, src.handle(), buffer_, 1, &copy_region);
}

} // namespace soar::vk
