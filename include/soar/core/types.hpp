#pragma once

#include <cstdint>
#include <cstddef>
#include <string_view>

namespace soar::core {

enum class DataType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Int32   = 2,
    UInt8   = 3,
};

constexpr size_t element_size(DataType dt) noexcept {
    switch (dt) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32:   return 4;
        case DataType::UInt8:   return 1;
        default: return 0;
    }
}

constexpr std::string_view data_type_name(DataType dt) noexcept {
    switch (dt) {
        case DataType::Float32: return "Float32";
        case DataType::Float16: return "Float16";
        case DataType::Int32:   return "Int32";
        case DataType::UInt8:   return "UInt8";
        default: return "Unknown";
    }
}

// 16-bit half precision float representation for storage
struct alignas(2) float16_t {
    uint16_t raw{0};

    constexpr float16_t() noexcept = default;
    constexpr explicit float16_t(uint16_t val) noexcept : raw(val) {}
};

} // namespace soar::core
