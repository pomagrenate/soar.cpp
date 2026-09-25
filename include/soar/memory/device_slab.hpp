#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>
#include "soar/core/shape.hpp"
#include "soar/core/types.hpp"
#include "soar/core/error.hpp"

namespace soar::memory {

struct SubAllocation {
    size_t offset{0};
    size_t size{0};
    core::Shape shape{};
    core::DataType dtype{core::DataType::Float32};
    std::string tag{};
};

class DeviceSlabSubAllocator {
public:
    explicit DeviceSlabSubAllocator(size_t total_capacity_bytes, size_t alignment = 256) noexcept
        : total_capacity_(total_capacity_bytes), alignment_(alignment) {}

    core::Result<SubAllocation> sub_allocate(
        size_t size_bytes,
        core::Shape shape,
        core::DataType dtype,
        std::string_view tag = ""
    ) noexcept;

    void reset_dynamic_allocations() noexcept;
    void freeze_static_parameters() noexcept;

    [[nodiscard]] size_t total_capacity() const noexcept { return total_capacity_; }
    [[nodiscard]] size_t current_offset() const noexcept { return current_offset_; }
    [[nodiscard]] size_t static_boundary() const noexcept { return static_boundary_; }

private:
    size_t total_capacity_{0};
    size_t alignment_{256};
    size_t current_offset_{0};
    size_t static_boundary_{0};
    std::vector<SubAllocation> allocations_{};
};

} // namespace soar::memory
