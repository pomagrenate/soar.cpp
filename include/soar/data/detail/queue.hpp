#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>
#include <stdexcept>
#include <cassert>

namespace soar::data::detail {

/**
 * @brief Thread-safe blocking MPMC queue mirroring PyTorch torch::data::detail::Queue.
 */
template <typename T>
class Queue {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(value));
        }
        cv_.notify_one();
    }

    T pop(std::optional<std::chrono::milliseconds> timeout = std::nullopt) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout) {
            if (!cv_.wait_for(lock, *timeout, [this] { return !this->queue_.empty(); })) {
                throw std::runtime_error("Timeout in DataLoader queue while waiting for next batch");
            }
        } else {
            cv_.wait(lock, [this] { return !this->queue_.empty(); });
        }
        assert(!queue_.empty());
        T value = std::move(queue_.front());
        queue_.pop();
        return value;
    }

    size_t clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto sz = queue_.size();
        while (!queue_.empty()) {
            queue_.pop();
        }
        return sz;
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

} // namespace soar::data::detail
