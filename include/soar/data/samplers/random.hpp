#pragma once

#include <soar/data/samplers/base.hpp>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

namespace soar::data::samplers {

/// A Sampler that yields randomly permuted indices for each epoch.
/// Matches PyTorch torch::data::samplers::RandomSampler.
class RandomSampler : public Sampler<> {
public:
    explicit RandomSampler(size_t size) : size_(size), index_(0) {
        reset_indices();
    }

    void reset(std::optional<size_t> new_size = std::nullopt) override {
        if (new_size.has_value()) {
            size_ = *new_size;
        }
        reset_indices();
    }

    std::optional<std::vector<size_t>> next(size_t batch_size) override {
        const auto remaining = size_ > index_ ? size_ - index_ : 0;
        if (remaining == 0) {
            return std::nullopt;
        }
        const size_t count = std::min(batch_size, remaining);
        std::vector<size_t> index_batch(indices_.begin() + index_, indices_.begin() + index_ + count);
        index_ += count;
        return index_batch;
    }

    [[nodiscard]] size_t index() const noexcept {
        return index_;
    }

private:
    void reset_indices() {
        indices_.resize(size_);
        std::iota(indices_.begin(), indices_.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
        index_ = 0;
    }

    size_t size_{0};
    size_t index_{0};
    std::vector<size_t> indices_;
};

} // namespace soar::data::samplers
