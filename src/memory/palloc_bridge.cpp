#include "soar/memory/palloc_bridge.hpp"
#include "palloc.h"
#include "palloc/arena_pomai.h"
#include <cassert>
#include <algorithm>

namespace soar::memory {

PallocArena::PallocArena(size_t capacity_bytes, bool thread_safe) noexcept
    : capacity_bytes_(capacity_bytes), allocated_bytes_(0) {
    arena_ = ::p_arena_create_for_vector_ex(capacity_bytes, thread_safe);
}

PallocArena::~PallocArena() noexcept {
    if (arena_ != nullptr) {
        ::p_arena_destroy(arena_);
        arena_ = nullptr;
    }
}

PallocArena::PallocArena(PallocArena&& other) noexcept
    : arena_(other.arena_),
      capacity_bytes_(other.capacity_bytes_),
      allocated_bytes_(other.allocated_bytes_) {
    other.arena_ = nullptr;
    other.capacity_bytes_ = 0;
    other.allocated_bytes_ = 0;
}

PallocArena& PallocArena::operator=(PallocArena&& other) noexcept {
    if (this != &other) {
        if (arena_ != nullptr) {
            ::p_arena_destroy(arena_);
        }
        arena_ = other.arena_;
        capacity_bytes_ = other.capacity_bytes_;
        allocated_bytes_ = other.allocated_bytes_;

        other.arena_ = nullptr;
        other.capacity_bytes_ = 0;
        other.allocated_bytes_ = 0;
    }
    return *this;
}

void* PallocArena::allocate_bytes(size_t bytes, size_t alignment) noexcept {
    if (arena_ == nullptr || bytes == 0) {
        return nullptr;
    }

    // p_arena_alloc_vector natively guarantees 64-byte alignment
    if (alignment <= 64) {
        void* ptr = ::p_arena_alloc_vector(arena_, bytes, 1);
        if (ptr != nullptr) {
            allocated_bytes_ += bytes;
        }
        return ptr;
    }

    // Over-allocate to satisfy alignment > 64 bytes (e.g. 256 for Vulkan uniform buffers)
    size_t padded_bytes = bytes + alignment;
    void* raw = ::p_arena_alloc_vector(arena_, padded_bytes, 1);
    if (raw == nullptr) {
        return nullptr;
    }

    uintptr_t addr = reinterpret_cast<uintptr_t>(raw);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    allocated_bytes_ += padded_bytes;
    return reinterpret_cast<void*>(aligned_addr);
}

void PallocArena::reset() noexcept {
    if (arena_ != nullptr) {
        ::p_arena_reset(arena_);
        allocated_bytes_ = 0;
    }
}

} // namespace soar::memory
