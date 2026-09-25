#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <string>
#include <initializer_list>
#include <numeric>
#include <sstream>

namespace soar::core {

class Shape {
public:
    int64_t n{1};
    int64_t c{1};
    int64_t h{1};
    int64_t w{1};

    constexpr Shape() noexcept : dims_{} {}

    Shape(int64_t n_, int64_t c_, int64_t h_, int64_t w_) noexcept
        : n(n_), c(c_), h(h_), w(w_), dims_{n_, c_, h_, w_} {}

    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {
        sync_legacy_fields();
    }

    explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {
        sync_legacy_fields();
    }

    [[nodiscard]] size_t ndim() const noexcept {
        return dims_.size();
    }

    [[nodiscard]] int64_t operator[](size_t index) const {
        return dims_[index];
    }

    [[nodiscard]] int64_t& operator[](size_t index) {
        return dims_[index];
    }

    [[nodiscard]] int64_t numel() const noexcept {
        if (dims_.empty()) return 0;
        int64_t total = 1;
        for (auto d : dims_) total *= d;
        return total;
    }

    [[nodiscard]] int64_t spatial_numel() const noexcept {
        return h * w;
    }

    [[nodiscard]] std::array<int64_t, 4> strides_nchw() const noexcept {
        return {c * h * w, h * w, w, 1};
    }

    [[nodiscard]] bool operator==(const Shape& other) const noexcept {
        return dims_ == other.dims_;
    }

    [[nodiscard]] bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] const std::vector<int64_t>& dims() const noexcept { return dims_; }

private:
    std::vector<int64_t> dims_;

    void sync_legacy_fields() noexcept {
        if (dims_.size() == 4) {
            n = dims_[0]; c = dims_[1]; h = dims_[2]; w = dims_[3];
        } else if (dims_.size() == 3) {
            n = 1; c = dims_[0]; h = dims_[1]; w = dims_[2];
        } else if (dims_.size() == 2) {
            n = 1; c = 1; h = dims_[0]; w = dims_[1];
        } else if (dims_.size() == 1) {
            n = 1; c = 1; h = 1; w = dims_[0];
        }
    }
};

} // namespace soar::core
