#include "soar/memory/device_slab.hpp"
#include <algorithm>

namespace soar::memory {

core::Result<SubAllocation> DeviceSlabSubAllocator::sub_allocate(
    size_t size_bytes,
    core::Shape shape,
    core::DataType dtype,
    std::string_view tag
) noexcept {
    if (size_bytes == 0) {
        return core::Status(core::StatusCode::InvalidArgument, "Cannot allocate 0 bytes in device slab");
    }

    // Align current offset to requested alignment bound (minStorageBufferOffsetAlignment)
    size_t aligned_offset = (current_offset_ + alignment_ - 1) & ~(alignment_ - 1);
    size_t required_end = aligned_offset + size_bytes;

    if (required_end > total_capacity_) {
        return core::Status(
            core::StatusCode::OutOfMemory,
            "Device VRAM slab capacity exceeded"
        );
    }

    SubAllocation alloc{
        .offset = aligned_offset,
        .size = size_bytes,
        .shape = shape,
        .dtype = dtype,
        .tag = std::string(tag),
    };

    allocations_.push_back(alloc);
    current_offset_ = required_end;
    return alloc;
}

void DeviceSlabSubAllocator::freeze_static_parameters() noexcept {
    // Round boundary up to alignment
    static_boundary_ = (current_offset_ + alignment_ - 1) & ~(alignment_ - 1);
    current_offset_ = static_boundary_;
}

void DeviceSlabSubAllocator::reset_dynamic_allocations() noexcept {
    // Reset offset back to static parameter boundary (O(1) bulk dynamic reset)
    current_offset_ = static_boundary_;

    // Remove dynamic sub-allocations
    std::erase_if(allocations_, [this](const SubAllocation& a) {
        return a.offset >= static_boundary_;
    });
}

} // namespace soar::memory
