#include <soar/vulkan/caching_allocator.hpp>
#include <soar/core/logging.hpp>
#include <soar/core/error.hpp>
#include <algorithm>

namespace soar::vk {

CachingAllocator::CachingAllocator(VulkanContext& ctx) : ctx_(&ctx) {
    SOAR_LOG_INFO("CachingAllocator initialized for Vulkan device memory");
}

CachingAllocator::~CachingAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Clean up all free blocks
    for (auto& block : free_blocks_) {
        if (block.buffer != VK_NULL_HANDLE) {
            ctx_->loader().vkDestroyBuffer(ctx_->device(), block.buffer, nullptr);
        }
        if (block.memory != VK_NULL_HANDLE) {
            ctx_->loader().vkFreeMemory(ctx_->device(), block.memory, nullptr);
        }
    }
    free_blocks_.clear();
    
    // Clean up any remaining allocated blocks
    for (auto& [buffer_ptr, block] : allocated_blocks_) {
        if (block.buffer != VK_NULL_HANDLE) {
            ctx_->loader().vkDestroyBuffer(ctx_->device(), block.buffer, nullptr);
        }
        if (block.memory != VK_NULL_HANDLE) {
            ctx_->loader().vkFreeMemory(ctx_->device(), block.memory, nullptr);
        }
    }
    allocated_blocks_.clear();
    
    SOAR_LOG_INFO("CachingAllocator destroyed. Stats: allocations={}, hits={}, misses={}, peak_cached={}MB",
              stats_.total_allocations, stats_.cache_hits, stats_.cache_misses,
              stats_.peak_cached_bytes / (1024 * 1024));
}

std::unique_ptr<VulkanBuffer> CachingAllocator::allocate(size_t size_bytes, BufferUsageType usage_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    stats_.total_allocations++;
    
    // Try to find a suitable cached block
    MemoryBlock* cached_block = find_cached_block(size_bytes, usage_type);
    if (cached_block) {
        stats_.cache_hits++;
        auto buffer = std::make_unique<VulkanBuffer>(*ctx_, cached_block->buffer, 
                                                     cached_block->offset, size_bytes);
        allocated_blocks_[buffer.get()] = *cached_block;
        
        // Remove from free list (simplified approach)
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (&(*it) == cached_block) {
                free_blocks_.erase(it);
                break;
            }
        }
        
        return buffer;
    }
    
    // Cache miss - allocate new block
    stats_.cache_misses++;
    MemoryBlock new_block = allocate_new_block(size_bytes, usage_type);
    
    auto buffer = std::make_unique<VulkanBuffer>(*ctx_, new_block.buffer,
                                                     new_block.offset, size_bytes);
    allocated_blocks_[buffer.get()] = new_block;
    
    return buffer;
}

void CachingAllocator::deallocate(VulkanBuffer* buffer) {
    if (!buffer) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocated_blocks_.find(buffer);
    if (it != allocated_blocks_.end()) {
        return_block_to_cache(it->second);
        allocated_blocks_.erase(it);
    }
}

void CachingAllocator::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    SOAR_LOG_INFO("Emptying cache. Current cached bytes: {}MB", 
              stats_.current_cached_bytes / (1024 * 1024));
    
    for (auto& block : free_blocks_) {
        if (block.buffer != VK_NULL_HANDLE) {
            ctx_->loader().vkDestroyBuffer(ctx_->device(), block.buffer, nullptr);
        }
        if (block.memory != VK_NULL_HANDLE) {
            ctx_->loader().vkFreeMemory(ctx_->device(), block.memory, nullptr);
        }
    }
    free_blocks_.clear();
    stats_.current_cached_bytes = 0;
}

CachingAllocator::CacheStats CachingAllocator::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

MemoryBlock* CachingAllocator::find_cached_block(size_t size_bytes, BufferUsageType usage_type) {
    // Simplified: find first block that fits the size requirement
    // PyTorch uses more sophisticated size-based binning
    for (auto& block : free_blocks_) {
        if (!block.allocated && block.size >= size_bytes) {
            // For now, accept all blocks regardless of age
            // Full implementation would have age-based eviction
            return &block;
        }
    }
    return nullptr;
}

MemoryBlock CachingAllocator::allocate_new_block(size_t size_bytes, BufferUsageType usage_type) {
    MemoryBlock block;
    block.size = size_bytes;
    block.offset = 0;
    block.allocated = true;
    block.last_used = std::chrono::steady_clock::now();
    
    // Directly allocate Vulkan buffer and memory (copied from VulkanBuffer constructor)
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

    VkResult res = ctx_->loader().vkCreateBuffer(ctx_->device(), &buf_info, nullptr, &block.buffer);
    if (res != VK_SUCCESS) {
        throw DeviceError("Failed to create VkBuffer in caching allocator. VkResult: " + std::to_string(res));
    }

    VkMemoryRequirements mem_reqs;
    ctx_->loader().vkGetBufferMemoryRequirements(ctx_->device(), block.buffer, &mem_reqs);

    uint32_t mem_type_index = 0;
    try {
        mem_type_index = ctx_->find_memory_type(mem_reqs.memoryTypeBits, mem_flags);
    } catch (const MemoryError&) {
        mem_type_index = ctx_->find_memory_type(mem_reqs.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    }

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_reqs.size;
    alloc_info.memoryTypeIndex = mem_type_index;

    res = ctx_->loader().vkAllocateMemory(ctx_->device(), &alloc_info, nullptr, &block.memory);
    if (res != VK_SUCCESS) {
        ctx_->loader().vkDestroyBuffer(ctx_->device(), block.buffer, nullptr);
        block.buffer = VK_NULL_HANDLE;
        throw MemoryError("Failed to allocate VkDeviceMemory in caching allocator: " + std::to_string(mem_reqs.size) + " bytes. VkResult: " + std::to_string(res));
    }

    res = ctx_->loader().vkBindBufferMemory(ctx_->device(), block.buffer, block.memory, 0);
    if (res != VK_SUCCESS) {
        ctx_->loader().vkFreeMemory(ctx_->device(), block.memory, nullptr);
        ctx_->loader().vkDestroyBuffer(ctx_->device(), block.buffer, nullptr);
        block.buffer = VK_NULL_HANDLE;
        block.memory = VK_NULL_HANDLE;
        throw DeviceError("Failed to bind VkBuffer to memory in caching allocator. VkResult: " + std::to_string(res));
    }
    
    return block;
}

void CachingAllocator::return_block_to_cache(MemoryBlock block) {
    block.allocated = false;
    block.last_used = std::chrono::steady_clock::now();
    
    stats_.current_cached_bytes += block.size;
    stats_.peak_cached_bytes = std::max(stats_.peak_cached_bytes, stats_.current_cached_bytes);
    
    evict_if_needed(block.size);
    
    free_blocks_.push_back(std::move(block));
}

void CachingAllocator::evict_if_needed(size_t required_bytes) {
    if (stats_.current_cached_bytes + required_bytes <= MAX_CACHE_SIZE) {
        return; // No eviction needed
    }
    
    // Evict oldest blocks until we have enough space
    auto now = std::chrono::steady_clock::now();
    
    while (stats_.current_cached_bytes + required_bytes > MAX_CACHE_SIZE && !free_blocks_.empty()) {
        // Find oldest block
        auto oldest = std::min_element(free_blocks_.begin(), free_blocks_.end(),
            [&now](const MemoryBlock& a, const MemoryBlock& b) {
                auto age_a = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - a.last_used).count();
                auto age_b = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - b.last_used).count();
                return age_a > age_b;
            });
        
        if (oldest != free_blocks_.end()) {
            stats_.current_cached_bytes -= oldest->size;
            
            if (oldest->buffer != VK_NULL_HANDLE) {
                ctx_->loader().vkDestroyBuffer(ctx_->device(), oldest->buffer, nullptr);
            }
            if (oldest->memory != VK_NULL_HANDLE) {
                ctx_->loader().vkFreeMemory(ctx_->device(), oldest->memory, nullptr);
            }
            
            free_blocks_.erase(oldest);
        }
    }
}

} // namespace soar::vk