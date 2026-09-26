#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace soar::data::detail::sequencers {

namespace detail {
template <typename Result>
bool buffer_contains_result(const std::vector<std::optional<Result>>& buffer) {
    return std::any_of(
        buffer.begin(), buffer.end(), [](const std::optional<Result>& result) {
            return result.has_value();
        });
}
} // namespace detail

/// A Sequencer accepts a function that yields the next result of a DataLoader
/// and influences the order in which results are returned.
template <typename Result>
struct Sequencer {
    using ResultProducer = std::function<std::optional<Result>()>;
    virtual ~Sequencer() = default;
    virtual std::optional<Result> next(ResultProducer next_result) = 0;
};

/// A Sequencer that does not enforce ordering (identity function).
template <typename Result>
struct NoSequencer final : public Sequencer<Result> {
    using typename Sequencer<Result>::ResultProducer;
    std::optional<Result> next(ResultProducer next_result) override {
        return next_result();
    }
};

/// A Sequencer that buffers results and returns them in order of sequence numbers.
/// Matches PyTorch torch::data::detail::sequencers::OrderedSequencer.
template <typename Result>
struct OrderedSequencer : public Sequencer<Result> {
    using typename Sequencer<Result>::ResultProducer;

    explicit OrderedSequencer(size_t max_jobs) : buffer_(std::max(size_t(1), max_jobs)) {}

    std::optional<Result> next(ResultProducer next_result) override {
        // If we already have the result for the next sequence number, return it.
        if (auto& maybe_result = buffer(next_sequence_number_)) {
            auto result = std::move(*maybe_result);
            buffer(next_sequence_number_++).reset();
            return result;
        }

        // Otherwise wait for the next result.
        while (true) {
            auto result = next_result();
            if (!result) {
                assert(!detail::buffer_contains_result(buffer_));
                break;
            }
            if (result->sequence_number == next_sequence_number_) {
                ++next_sequence_number_;
                return result;
            }
            assert(!buffer(result->sequence_number).has_value());
            buffer(result->sequence_number) = std::move(result);
        }
        return std::nullopt;
    }

    std::optional<Result>& buffer(size_t index) {
        return buffer_.at(index % buffer_.size());
    }

    size_t next_sequence_number_ = 0;
    std::vector<std::optional<Result>> buffer_;
};

} // namespace soar::data::detail::sequencers
