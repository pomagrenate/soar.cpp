#pragma once

#include <soar/data/samplers/base.hpp>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace soar::data::samplers {

/// A Sampler that returns indices sequentially from 0 to size - 1.
/// Matches PyTorch torch::data::samplers::SequentialSampler.
class SequentialSampler : public Sampler<> {
public:
    explicit SequentialSampler(size_t size) : size_(size), index_(0) {}

    void reset(std::optional<size_t> new_size = std::nullopt) override {
        if (new_size.has_value()) {
            size_ = *new_size;
        }
        index_ = 0;
    }

    std::optional<std::vector<size_t>> next(size_t batch_size) override {
        const auto remaining = size_ > index_ ? size_ - index_ : 0;
        if (remaining == 0) {
            return std::nullopt;
        }
        std::vector<size_t> index_batch(std::min(batch_size, remaining));
        for (auto& i : index_batch) {
            i = index_++;
        }
        return index_batch;
    }

    [[nodiscard]] size_t index() const noexcept {
        return index_;
    }

private:
    size_t size_{0};
    size_t index_{0};
};

} // namespace soar::data::samplers
