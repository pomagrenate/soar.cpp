#pragma once

#include <cstddef>
#include <limits>
#include <new>
#include <type_traits>
#include <vector>
#include "palloc.h"

namespace soar::memory {

/**
 * @brief C++20 STL-compatible allocator backed strictly by palloc.
 * Guarantees 64-byte alignment for SIMD vectorization (AVX2/AVX-512)
 * and directs all allocations/deallocations through palloc's core engine.
 */
template <typename T>
struct PallocAllocator {
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    constexpr PallocAllocator() noexcept = default;
    constexpr PallocAllocator(const PallocAllocator&) noexcept = default;
    template <typename U>
    constexpr PallocAllocator(const PallocAllocator<U>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = PallocAllocator<U>;
    };

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        constexpr std::size_t alignment = (alignof(T) > 64) ? alignof(T) : 64;
        void* ptr = ::pa_malloc_aligned(n * sizeof(T), alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        if (p) {
            constexpr std::size_t alignment = (alignof(T) > 64) ? alignof(T) : 64;
            ::pa_free_aligned(p, alignment);
        }
    }

    template <typename U>
    bool operator==(const PallocAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const PallocAllocator<U>&) const noexcept { return false; }
};

/// Type alias for std::vector using strict palloc allocation
template <typename T>
using PallocVector = std::vector<T, PallocAllocator<T>>;

/**
 * @brief Memory statistics queried directly from palloc runtime.
 */
struct PallocStats {
    size_t elapsed_msecs{0};
    size_t user_msecs{0};
    size_t system_msecs{0};
    size_t current_rss{0};
    size_t peak_rss{0};
    size_t current_commit{0};
    size_t peak_commit{0};
    size_t page_faults{0};
};

inline PallocStats get_palloc_process_info() noexcept {
    PallocStats stats;
    ::pa_process_info(&stats.elapsed_msecs, &stats.user_msecs, &stats.system_msecs,
                      &stats.current_rss, &stats.peak_rss,
                      &stats.current_commit, &stats.peak_commit, &stats.page_faults);
    return stats;
}

inline void print_palloc_stats() noexcept {
    ::pa_stats_print(nullptr);
}

} // namespace soar::memory
