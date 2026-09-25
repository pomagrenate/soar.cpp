#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include "soar/core/error.hpp"

// Forward declaration of palloc C structures
struct pa_arena_s;
typedef struct pa_arena_s pa_arena_t;

namespace soar::memory {

class PallocArena {
public:
    PallocArena() noexcept = default;
    explicit PallocArena(size_t capacity_bytes, bool thread_safe = false) noexcept;
    ~PallocArena() noexcept;

    // Move only
    PallocArena(const PallocArena&) = delete;
    PallocArena& operator=(const PallocArena&) = delete;
    PallocArena(PallocArena&& other) noexcept;
    PallocArena& operator=(PallocArena&& other) noexcept;

    [[nodiscard]] void* allocate_bytes(size_t bytes, size_t alignment = 64) noexcept;

    template <typename T>
    [[nodiscard]] T* allocate_elements(size_t count, size_t alignment = 64) noexcept {
        return static_cast<T*>(allocate_bytes(count * sizeof(T), alignment));
    }

    void reset() noexcept;
    [[nodiscard]] size_t capacity() const noexcept { return capacity_bytes_; }
    [[nodiscard]] size_t allocated() const noexcept { return allocated_bytes_; }
    [[nodiscard]] bool is_valid() const noexcept { return arena_ != nullptr; }

private:
    pa_arena_t* arena_{nullptr};
    size_t capacity_bytes_{0};
    size_t allocated_bytes_{0};
};

} // namespace soar::memory
