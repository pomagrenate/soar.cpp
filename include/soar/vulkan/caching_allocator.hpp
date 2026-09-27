#pragma once

#include <soar/vulkan/buffer.hpp>
#include <soar/vulkan/context.hpp>
#include <soar/core/types.hpp>
#include <unordered_map>
#include <list>
#include <memory>
#include <mutex>
#include <chrono>

namespace soar::vk {

/**
 * @brief Memory block for caching allocator.
 * 
 * Inspired by PyTorch's block-based caching, this represents a contiguous
 * memory region that can be split and coalesced for efficient reuse.
 */
struct MemoryBlock {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceMemory memory{VK_NULL_HANDLE};
    size_t size{0};
    size_t offset{0};
    bool allocated{false};
    std::chrono::time_point<std::chrono::steady_clock> last_used;
};

/**
 * @brief Simple caching allocator for Vulkan device memory.
 * 
 * Inspired by PyTorch's CachingDeviceAllocator, this provides:
 * - Block-level memory management
 * - Memory reuse through caching
 * - Simple age-based eviction
 * - Thread-safe allocation
 * 
 * Simplified compared to PyTorch's implementation but provides similar
 * benefits for 2048×2048 high-resolution workloads.
 */
class CachingAllocator {
public:
    explicit CachingAllocator(VulkanContext& ctx);
    ~CachingAllocator();

    // Non-copyable, movable
    CachingAllocator(const CachingAllocator&) = delete;
    CachingAllocator& operator=(const CachingAllocator&) = delete;
    CachingAllocator(CachingAllocator&&) = delete;
    CachingAllocator& operator=(CachingAllocator&&) = delete;

    /**
     * @brief Allocate a buffer of the specified size and usage type.
     */
    std::unique_ptr<VulkanBuffer> allocate(size_t size_bytes, BufferUsageType usage_type);

    /**
     * @brief Return a buffer to the cache for reuse.
     */
    void deallocate(VulkanBuffer* buffer);

    /**
     * @brief Clear all cached memory (useful for OOM situations).
     */
    void empty_cache();

    /**
     * @brief Get statistics about cache performance.
     */
    struct CacheStats {
        size_t total_allocations{0};
        size_t cache_hits{0};
        size_t cache_misses{0};
        size_t current_cached_bytes{0};
        size_t peak_cached_bytes{0};
    };
    
    CacheStats get_stats() const;

private:
    /**
     * @brief Try to find a suitable block in the cache.
     */
    MemoryBlock* find_cached_block(size_t size_bytes, BufferUsageType usage_type);

    /**
     * @brief Allocate a new block from Vulkan.
     */
    MemoryBlock allocate_new_block(size_t size_bytes, BufferUsageType usage_type);

    /**
     * @brief Return a block to the cache.
     */
    void return_block_to_cache(MemoryBlock block);

    /**
     * @brief Evict old blocks if cache is too large.
     */
    void evict_if_needed(size_t required_bytes);

    VulkanContext* ctx_;
    mutable std::mutex mutex_;
    
    // Free blocks organized by size (simplified)
    std::list<MemoryBlock> free_blocks_;
    
    // Allocated blocks tracking
    std::unordered_map<VulkanBuffer*, MemoryBlock> allocated_blocks_;
    
    CacheStats stats_;
    
    // Cache configuration
    static constexpr size_t MAX_CACHE_SIZE = 512 * 1024 * 1024; // 512MB
    static constexpr size_t EVICTION_AGE_MS = 5000; // 5 seconds
};

} // namespace soar::vk