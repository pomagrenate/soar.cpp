#pragma once

#include <soar/data/detail/queue.hpp>
#include <optional>
#include <chrono>
#include <utility>

namespace soar::data::detail {

/**
 * @brief Coordinates DataLoader jobs between main thread and worker threads.
 * Matches PyTorch torch::data::detail::DataShuttle<Job, Result>.
 */
template <typename Job, typename Result>
class DataShuttle {
public:
    void push_job(Job job) {
        new_jobs_.push(std::move(job));
        ++in_flight_jobs_;
    }

    void push_result(Result result) {
        results_.push(std::move(result));
    }

    Job pop_job() {
        return new_jobs_.pop();
    }

    std::optional<Result> pop_result(std::optional<std::chrono::milliseconds> timeout = std::nullopt) {
        if (in_flight_jobs_ > 0) {
            auto result = results_.pop(timeout);
            --in_flight_jobs_;
            return result;
        }
        return std::nullopt;
    }

    void drain() {
        auto number_cleared = new_jobs_.clear();
        in_flight_jobs_ -= number_cleared;
        while (in_flight_jobs_ > 0) {
            pop_result();
        }
    }

    [[nodiscard]] size_t in_flight_jobs() const noexcept {
        return in_flight_jobs_;
    }

private:
    Queue<Job> new_jobs_;
    size_t in_flight_jobs_ = 0;
    Queue<Result> results_;
};

} // namespace soar::data::detail
