#pragma once

#include <span>
#include <cstdint>
#include <cassert>
#include "types.hpp"
#include "shape.hpp"

namespace soar::core {

template <typename T>
class TensorView {
public:
    using element_type = T;

    TensorView() noexcept : data_(nullptr), shape_() {}
    TensorView(T* ptr, Shape shape) noexcept : data_(ptr), shape_(std::move(shape)) {}

    [[nodiscard]] constexpr T* data() noexcept { return data_; }
    [[nodiscard]] constexpr const T* data() const noexcept { return data_; }

    [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] int64_t numel() const noexcept { return shape_.numel(); }
    [[nodiscard]] size_t bytes() const noexcept { return shape_.numel() * sizeof(T); }

    [[nodiscard]] constexpr T& operator()(int64_t batch, int64_t channel, int64_t y, int64_t x) noexcept {
        assert(data_ != nullptr);
        int64_t idx = ((batch * shape_.c + channel) * shape_.h + y) * shape_.w + x;
        return data_[idx];
    }

    [[nodiscard]] constexpr const T& operator()(int64_t batch, int64_t channel, int64_t y, int64_t x) const noexcept {
        assert(data_ != nullptr);
        int64_t idx = ((batch * shape_.c + channel) * shape_.h + y) * shape_.w + x;
        return data_[idx];
    }

    [[nodiscard]] constexpr std::span<T> as_span() noexcept {
        return std::span<T>(data_, static_cast<size_t>(numel()));
    }

    [[nodiscard]] constexpr std::span<const T> as_span() const noexcept {
        return std::span<const T>(data_, static_cast<size_t>(numel()));
    }

private:
    T* data_{nullptr};
    Shape shape_{};
};

} // namespace soar::core
