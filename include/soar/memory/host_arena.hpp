#pragma once

#include "palloc_bridge.hpp"
#include "soar/core/shape.hpp"
#include "soar/core/tensor_view.hpp"
#include "soar/core/types.hpp"

namespace soar::memory {

class HostArena {
public:
    explicit HostArena(size_t capacity_bytes) noexcept
        : arena_(capacity_bytes, false) {}

    template <typename T>
    [[nodiscard]] core::TensorView<T> create_tensor(const core::Shape& shape, size_t alignment = 64) noexcept {
        size_t total_elements = static_cast<size_t>(shape.numel());
        T* ptr = arena_.allocate_elements<T>(total_elements, alignment);
        if (!ptr) {
            return core::TensorView<T>(nullptr, core::Shape{0, 0, 0, 0});
        }
        return core::TensorView<T>(ptr, shape);
    }

    void reset() noexcept {
        arena_.reset();
    }

    [[nodiscard]] size_t capacity() const noexcept { return arena_.capacity(); }
    [[nodiscard]] size_t allocated() const noexcept { return arena_.allocated(); }

private:
    PallocArena arena_;
};

} // namespace soar::memory
