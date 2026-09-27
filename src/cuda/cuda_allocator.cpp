#include <soar/cuda/cuda_allocator.hpp>
#include <algorithm>
#include <cassert>

namespace soar::cuda {

bool BlockComparator::operator()(const Block* a, const Block* b) const noexcept {
    if (a->size != b->size) {
        return a->size < b->size;
    }
    if (a->stream != b->stream) {
        return a->stream < b->stream;
    }
    return a->ptr < b->ptr;
}

CudaMemoryPool::CudaMemoryPool(int device_index)
    : device_index_(device_index) {}

CudaMemoryPool::~CudaMemoryPool() {
    empty_cache();
}

CudaMemoryPool& CudaMemoryPool::instance(int device_index) {
    static CudaMemoryPool s_pool(device_index);
    return s_pool;
}

size_t CudaMemoryPool::round_size(size_t size) noexcept {
    if (size == 0) return kMinBlockSize;
    size_t align = kMinBlockSize;
    return (size + align - 1) & ~(align - 1);
}

Block* CudaMemoryPool::find_free_block(size_t size, cudaStream_t stream, bool is_small) {
    auto& pool_set = is_small ? small_blocks_ : large_blocks_;
    Block search_key(nullptr, size, device_index_, stream, is_small);
    auto it = pool_set.lower_bound(&search_key);

    // Strict stream matching: only reuse blocks on the same stream
    // or the default stream (null). This prevents race conditions.
    while (it != pool_set.end()) {
        Block* candidate = *it;
        if (candidate->stream == stream || 
            (candidate->stream == nullptr && stream == nullptr)) {
            pool_set.erase(it);
            return candidate;
        }
        ++it;
    }

    return nullptr;
}

Block* CudaMemoryPool::allocate_new_segment(size_t size, cudaStream_t stream, bool is_small) {
    size_t alloc_size = is_small ? kSmallBuffer : std::max(size, size_t(20 * 1024 * 1024));
    void* dev_ptr = nullptr;
    cudaError_t status = cudaMalloc(&dev_ptr, alloc_size);
    if (status != cudaSuccess) {
        // Try freeing unused cached segments first
        empty_cache();
        status = cudaMalloc(&dev_ptr, alloc_size);
        if (status != cudaSuccess) {
            // Fallback to exact requested size
            alloc_size = size;
            SOAR_CUDA_CHECK(cudaMalloc(&dev_ptr, alloc_size));
        }
    }

    reserved_bytes_ += alloc_size;

    auto block = new Block(dev_ptr, alloc_size, device_index_, stream, is_small);
    block->segment_ptr = dev_ptr;
    block->segment_size = alloc_size;
    all_segments_.push_back(block);
    return block;
}

void CudaMemoryPool::split_block(Block* block, size_t size) {
    if (block->size < size + kMinBlockSize) {
        return;
    }

    size_t remaining_size = block->size - size;
    void* remaining_ptr = static_cast<char*>(block->ptr) + size;

    auto remaining = new Block(remaining_ptr, remaining_size, block->device_index, block->stream, block->is_small);
    remaining->segment_ptr = block->segment_ptr;
    remaining->segment_size = block->segment_size;
    remaining->prev = block;
    remaining->next = block->next;
    if (block->next) {
        block->next->prev = remaining;
    }
    block->next = remaining;
    block->size = size;

    auto& pool_set = remaining->is_small ? small_blocks_ : large_blocks_;
    pool_set.insert(remaining);
}

void CudaMemoryPool::coalesce_block(Block* block) {
    auto& pool_set = block->is_small ? small_blocks_ : large_blocks_;

    // Coalesce with left (prev) neighbor if free
    if (block->prev && !block->prev->allocated && block->prev->segment_ptr == block->segment_ptr) {
        Block* prev = block->prev;
        pool_set.erase(prev);
        prev->size += block->size;
        prev->next = block->next;
        if (block->next) {
            block->next->prev = prev;
        }
        delete block;
        block = prev;
    }

    // Coalesce with right (next) neighbor if free
    if (block->next && !block->next->allocated && block->next->segment_ptr == block->segment_ptr) {
        Block* next = block->next;
        pool_set.erase(next);
        block->size += next->size;
        block->next = next->next;
        if (next->next) {
            next->next->prev = block;
        }
        delete next;
    }

    block->allocated = false;
    pool_set.insert(block);
}

void CudaMemoryPool::try_reclaim() {
    // Process pending frees to reclaim memory from completed streams
    process_pending_frees();
    
    // If still no free blocks, empty cache to release everything
    empty_cache();
}

void* CudaMemoryPool::allocate(size_t size, cudaStream_t stream) {
    if (size == 0) return nullptr;

    size_t aligned_size = round_size(size);
    bool is_small = aligned_size <= kSmallAllocationThreshold;

    std::lock_guard<std::mutex> lock(mutex_);

    Block* block = find_free_block(aligned_size, stream, is_small);
    if (!block) {
        block = allocate_new_segment(aligned_size, stream, is_small);
    }

    split_block(block, aligned_size);

    block->allocated = true;
    block->requested_size = size;
    block->stream = stream;

    allocated_blocks_[block->ptr] = block;
    allocated_bytes_ += block->size;
    active_blocks_count_++;

    return block->ptr;
}

void CudaMemoryPool::record_pending_free(Block* block) {
    // Record an event on the stream where the block was used
    cudaEvent_t event = nullptr;
    SOAR_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    SOAR_CUDA_CHECK(cudaEventRecord(event, block->stream));
    
    PendingFree pending;
    pending.block = block;
    pending.event = event;
    
    pending_frees_[block->stream].push_back(pending);
}

void CudaMemoryPool::process_pending_frees() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [stream, frees] : pending_frees_) {
        auto it = frees.begin();
        while (it != frees.end()) {
            if (cudaEventQuery(it->event) == cudaSuccess) {
                // Event completed - return block to pool
                cudaEventDestroy(it->event);
                coalesce_block(it->block);
                it = frees.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // Clean up empty pending free lists
    auto it = pending_frees_.begin();
    while (it != pending_frees_.end()) {
        if (it->second.empty()) {
            it = pending_frees_.erase(it);
        } else {
            ++it;
        }
    }
}

void CudaMemoryPool::deallocate(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        // Unknown pointer; fallback to cudaFree directly
        cudaFree(ptr);
        return;
    }

    Block* block = it->second;
    allocated_blocks_.erase(it);

    allocated_bytes_ -= block->size;
    active_blocks_count_--;

    // For stream-aware reclamation, record pending free instead of immediate coalescing
    if (block->stream != nullptr) {
        record_pending_free(block);
    } else {
        // Default stream can be immediately coalesced
        coalesce_block(block);
    }
}

void CudaMemoryPool::empty_cache() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Process pending frees first to reclaim as much as possible
    process_pending_frees();

    // Remove free contiguous segments
    std::vector<Block*> remaining_segments;
    for (Block* seg_head : all_segments_) {
        // If segment has no allocations and was not split, or all sub-blocks are free
        bool can_free = true;
        Block* curr = seg_head;
        while (curr) {
            if (curr->allocated) {
                can_free = false;
                break;
            }
            curr = curr->next;
        }

        if (can_free) {
            void* seg_ptr = seg_head->segment_ptr;
            size_t seg_size = seg_head->segment_size;
            // Remove sub-blocks from pool sets
            curr = seg_head;
            while (curr) {
                auto& pool_set = curr->is_small ? small_blocks_ : large_blocks_;
                pool_set.erase(curr);
                Block* nxt = curr->next;
                delete curr;
                curr = nxt;
            }
            reserved_bytes_ -= seg_size;
            cudaFree(seg_ptr);
        } else {
            remaining_segments.push_back(seg_head);
        }
    }
    all_segments_ = std::move(remaining_segments);
}

// ============================================================
//  PinnedHostAllocator implementation
// ============================================================

PinnedHostAllocator::~PinnedHostAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& [ptr, size] : blocks_) {
        cudaFreeHost(ptr);
    }
    blocks_.clear();
}

PinnedHostAllocator& PinnedHostAllocator::instance() {
    static PinnedHostAllocator s_instance;
    return s_instance;
}

void* PinnedHostAllocator::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    SOAR_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
    
    std::lock_guard<std::mutex> lock(mutex_);
    blocks_[ptr] = size;
    
    return ptr;
}

void PinnedHostAllocator::deallocate(void* ptr) {
    if (!ptr) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = blocks_.find(ptr);
    if (it != blocks_.end()) {
        blocks_.erase(it);
        cudaFreeHost(ptr);
    } else {
        // Unknown pointer - try to free anyway
        cudaFreeHost(ptr);
    }
}

} // namespace soar::cuda
