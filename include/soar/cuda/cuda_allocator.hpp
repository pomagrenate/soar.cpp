#pragma once

#include <soar/cuda/cuda_runtime.hpp>
#include <soar/cuda/cuda_stream.hpp>
#include <set>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstddef>
#include <cstdint>

namespace soar::cuda {

struct Block;

struct BlockComparator {
    bool operator()(const Block* a, const Block* b) const noexcept;
};

struct Block {
    void* ptr{nullptr};
    size_t size{0};
    size_t requested_size{0};
    int device_index{0};
    cudaStream_t stream{nullptr};
    bool allocated{false};

    // Intrusive links within the same contiguous segment
    Block* prev{nullptr};
    Block* next{nullptr};

    // Owning parent segment base pointer
    void* segment_ptr{nullptr};
    size_t segment_size{0};
    bool is_small{false};

    Block(void* p, size_t sz, int dev, cudaStream_t str, bool small) noexcept
        : ptr(p), size(sz), requested_size(sz), device_index(dev),
          stream(str), allocated(false), is_small(small) {}
};

/**
 * @brief Pending free block with event for stream-aware reclamation.
 * 
 * When a block is freed on a stream, we record an event and only
 * return the block to the free pool after the event completes.
 * This prevents race conditions where a block freed on stream A
 * could be reused on stream B before stream A completes.
 */
struct PendingFree {
    Block* block{nullptr};
    cudaEvent_t event{nullptr};
};

/**
 * @brief High-performance CUDA caching memory pool inspired by PyTorch CUDACachingAllocator.
 *
 * Implements:
 * - 64-byte host SIMD alignment & 512-byte GPU device alignment
 * - Small (<1MB) and Large (>=1MB) allocation bins
 * - Dynamic block splitting to eliminate internal fragmentation
 * - Immediate intrusive neighbor coalescing on deallocation
 * - Zero cudaDeviceSynchronize requirement; stream-ordered reuse
 */
class CudaMemoryPool {
public:
    static constexpr size_t kSmallAllocationThreshold = 1048576; // 1 MB
    static constexpr size_t kSmallBuffer = 2097152;              // 2 MB
    static constexpr size_t kMinBlockSize = 512;                 // 512 bytes alignment
    static constexpr size_t kAlignment = 64;                     // 64-byte alignment

    explicit CudaMemoryPool(int device_index = 0);
    ~CudaMemoryPool();

    CudaMemoryPool(const CudaMemoryPool&) = delete;
    CudaMemoryPool& operator=(const CudaMemoryPool&) = delete;

    /**
     * @brief Allocate cached device memory for the given stream.
     */
    void* allocate(size_t size, cudaStream_t stream = nullptr);

    /**
     * @brief Return device memory to the pool cache (coalescing adjacent blocks).
     */
    void deallocate(void* ptr);

    /**
     * @brief Process pending frees and return completed blocks to the pool.
     * 
     * Should be called periodically (e.g., on allocation failure or periodically
     * during training) to reclaim memory from completed streams.
     */
    void process_pending_frees();

    /**
     * @brief Release all cached unallocated blocks back to CUDA driver/runtime.
     */
    void empty_cache();

    [[nodiscard]] size_t allocated_bytes() const noexcept { return allocated_bytes_; }
    [[nodiscard]] size_t reserved_bytes() const noexcept { return reserved_bytes_; }
    [[nodiscard]] size_t active_blocks() const noexcept { return active_blocks_count_; }

    static CudaMemoryPool& instance(int device_index = 0);

    /**
     * @brief Try to reclaim memory by processing pending frees on OOM.
     */
    void try_reclaim();

private:
    [[nodiscard]] static size_t round_size(size_t size) noexcept;
    Block* find_free_block(size_t size, cudaStream_t stream, bool is_small);
    Block* allocate_new_segment(size_t size, cudaStream_t stream, bool is_small);
    void split_block(Block* block, size_t size);
    void coalesce_block(Block* block);
    void record_pending_free(Block* block);

    int device_index_{0};
    mutable std::mutex mutex_;

    std::set<Block*, BlockComparator> small_blocks_;
    std::set<Block*, BlockComparator> large_blocks_;

    std::unordered_map<void*, Block*> allocated_blocks_;
    std::vector<Block*> all_segments_;
    
    // Per-stream pending free queues for stream-aware reclamation
    std::unordered_map<cudaStream_t, std::vector<PendingFree>> pending_frees_;

    size_t allocated_bytes_{0};
    size_t reserved_bytes_{0};
    size_t active_blocks_count_{0};
};

/**
 * @brief Pinned host memory allocator for async H→D transfers.
 * 
 * Uses cudaMallocHost/cudaFreeHost to allocate page-locked memory,
 * which enables DMA engine transfers that can overlap with GPU compute.
 * Critical for 2048×2048 workloads where H→D transfer latency
 * (~5ms at PCIe 4.0 for 16MB) can be hidden behind compute.
 */
class PinnedHostAllocator {
public:
    static PinnedHostAllocator& instance();

    void* allocate(size_t size);
    void deallocate(void* ptr);

    // No copy/move
    PinnedHostAllocator(const PinnedHostAllocator&) = delete;
    PinnedHostAllocator& operator=(const PinnedHostAllocator&) = delete;

private:
    PinnedHostAllocator() = default;
    ~PinnedHostAllocator();

    std::unordered_map<void*, size_t> blocks_;
    std::mutex mutex_;
};

} // namespace soar::cuda
