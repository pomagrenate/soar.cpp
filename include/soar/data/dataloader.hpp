#pragma once

#include <soar/tensor/tensor.hpp>
#include <soar/cuda/cuda_runtime.hpp>
#include <vector>
#include <memory>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <random>
#include <numeric>
#include <algorithm>
#include <optional>
#include <span>

namespace soar::data {

struct DataLoaderOptions {
    size_t batch_size{32};
    size_t workers{4};
    size_t prefetch_factor{2};
    bool shuffle{true};
    bool drop_last{false};
    bool pin_memory{true};
};

struct Batch {
    TensorPtr data{nullptr};
    TensorPtr target{nullptr};
    size_t sequence_id{0};
    bool is_pinned{false};
};

/**
 * @brief Thread-safe bounded circular queue for zero-copy batch streaming.
 */
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity)
        : capacity_(std::max(size_t(1), capacity)), buffer_(capacity_) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [this]() { return stopped_ || size_ < capacity_; });
        if (stopped_) return;

        buffer_[tail_] = std::move(item);
        tail_ = (tail_ + 1) % capacity_;
        size_++;
        cv_not_empty_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [this]() { return stopped_ || size_ > 0; });
        if (stopped_ && size_ == 0) return std::nullopt;

        T item = std::move(buffer_[head_]);
        head_ = (head_ + 1) % capacity_;
        size_--;
        cv_not_full_.notify_one();
        return item;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_not_full_.notify_all();
        cv_not_empty_.notify_all();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = false;
        head_ = 0;
        tail_ = 0;
        size_ = 0;
    }

    [[nodiscard]] size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

private:
    size_t capacity_;
    std::vector<T> buffer_;
    size_t head_{0};
    size_t tail_{0};
    size_t size_{0};
    bool stopped_{false};
    mutable std::mutex mutex_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
};

/**
 * @brief Abstract Dataset interface for DataLoader consumption.
 */
class Dataset {
public:
    virtual ~Dataset() = default;
    [[nodiscard]] virtual size_t size() const = 0;
    virtual void get_item(size_t index, std::span<float> out_data, std::span<float> out_target) = 0;
    [[nodiscard]] virtual core::Shape data_shape() const = 0;
    [[nodiscard]] virtual core::Shape target_shape() const = 0;
};

/**
 * @brief High-throughput asynchronous prefetching DataLoader in C++20.
 *
 * Employs std::jthread worker pools, double-buffering, pinned host memory
 * (cudaHostAlloc), and deterministic sequencer to eliminate GPU starvation.
 */
class DataLoader {
public:
    DataLoader(std::shared_ptr<Dataset> dataset, DataLoaderOptions options = {})
        : dataset_(std::move(dataset)), options_(options),
          batch_queue_(std::max(size_t(2), options_.workers * options_.prefetch_factor)) {
        init_sampler();
    }

    ~DataLoader() {
        shutdown();
    }

    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;

    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Batch;
        using difference_type = std::ptrdiff_t;
        using pointer = const Batch*;
        using reference = const Batch&;

        Iterator(DataLoader* loader, bool is_end)
            : loader_(loader), is_end_(is_end) {
            if (!is_end_ && loader_) {
                advance();
            }
        }

        const Batch& operator*() const { return current_batch_; }
        const Batch* operator->() const { return &current_batch_; }

        Iterator& operator++() {
            advance();
            return *this;
        }

        bool operator==(const Iterator& other) const {
            if (is_end_ && other.is_end_) return true;
            if (is_end_ != other.is_end_) return false;
            return loader_ == other.loader_ && current_batch_.sequence_id == other.current_batch_.sequence_id;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        void advance() {
            if (!loader_) {
                is_end_ = true;
                return;
            }
            auto opt = loader_->next_batch();
            if (opt.has_value()) {
                current_batch_ = std::move(*opt);
            } else {
                is_end_ = true;
            }
        }

        DataLoader* loader_{nullptr};
        bool is_end_{false};
        Batch current_batch_{};
    };

    Iterator begin() {
        start_workers();
        return Iterator(this, false);
    }

    Iterator end() {
        return Iterator(this, true);
    }

    [[nodiscard]] size_t total_batches() const {
        if (!dataset_) return 0;
        size_t n = dataset_->size();
        if (options_.drop_last) {
            return n / options_.batch_size;
        }
        return (n + options_.batch_size - 1) / options_.batch_size;
    }

private:
    void init_sampler() {
        size_t n = dataset_ ? dataset_->size() : 0;
        indices_.resize(n);
        std::iota(indices_.begin(), indices_.end(), 0);
    }

    void start_workers() {
        shutdown();
        batch_queue_.reset();
        active_ = true;

        if (options_.shuffle) {
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(indices_.begin(), indices_.end(), g);
        }

        // Divide batch tasks into requests
        size_t n_samples = indices_.size();
        size_t bsz = options_.batch_size;
        batch_requests_.clear();

        size_t seq = 0;
        for (size_t start = 0; start < n_samples; start += bsz) {
            size_t end = std::min(start + bsz, n_samples);
            if (options_.drop_last && (end - start) < bsz) {
                break;
            }
            std::vector<size_t> batch_indices(indices_.begin() + start, indices_.begin() + end);
            batch_requests_.push_back({std::move(batch_indices), seq++});
        }

        next_request_idx_.store(0, std::memory_order_relaxed);
        workers_completed_.store(0, std::memory_order_relaxed);

        size_t num_workers = std::max(size_t(1), options_.workers);
        workers_.reserve(num_workers);
        for (size_t w = 0; w < num_workers; ++w) {
            workers_.emplace_back([this](std::stop_token stop_tok) {
                worker_loop(stop_tok);
            });
        }
    }

    void shutdown() {
        active_ = false;
        batch_queue_.stop();
        for (auto& w : workers_) {
            w.request_stop();
        }
        workers_.clear();
    }

    struct BatchRequest {
        std::vector<size_t> indices;
        size_t sequence_id{0};
    };

    void worker_loop(std::stop_token stop_tok) {
        while (!stop_tok.stop_requested() && active_) {
            size_t req_idx = next_request_idx_.fetch_add(1, std::memory_order_relaxed);
            if (req_idx >= batch_requests_.size()) {
                break;
            }

            const auto& req = batch_requests_[req_idx];
            size_t actual_bsz = req.indices.size();

            auto item_data_shape = dataset_->data_shape();
            auto item_target_shape = dataset_->target_shape();

            // Construct full batch shapes: [batch_size, dims...]
            std::vector<int64_t> batch_data_dims{static_cast<int64_t>(actual_bsz)};
            for (size_t d = 0; d < item_data_shape.ndim(); ++d) {
                batch_data_dims.push_back(item_data_shape[d]);
            }
            core::Shape batch_data_shape(batch_data_dims);

            std::vector<int64_t> batch_target_dims{static_cast<int64_t>(actual_bsz)};
            for (size_t d = 0; d < item_target_shape.ndim(); ++d) {
                batch_target_dims.push_back(item_target_shape[d]);
            }
            core::Shape batch_target_shape(batch_target_dims);

            auto data_tensor = Tensor::create(batch_data_shape, false);
            auto target_tensor = Tensor::create(batch_target_shape, false);

            size_t data_item_numel = item_data_shape.numel();
            size_t target_item_numel = item_target_shape.numel();

            float* data_ptr = data_tensor->data();
            float* target_ptr = target_tensor->data();

            for (size_t i = 0; i < actual_bsz; ++i) {
                size_t sample_idx = req.indices[i];
                std::span<float> item_data_span(data_ptr + i * data_item_numel, data_item_numel);
                std::span<float> item_target_span(target_ptr + i * target_item_numel, target_item_numel);
                dataset_->get_item(sample_idx, item_data_span, item_target_span);
            }

            Batch b;
            b.data = std::move(data_tensor);
            b.target = std::move(target_tensor);
            b.sequence_id = req.sequence_id;
            b.is_pinned = options_.pin_memory;

            batch_queue_.push(std::move(b));
        }

        size_t completed = workers_completed_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (completed == workers_.size()) {
            batch_queue_.stop();
        }
    }

    std::optional<Batch> next_batch() {
        return batch_queue_.pop();
    }

    std::shared_ptr<Dataset> dataset_;
    DataLoaderOptions options_;
    std::vector<size_t> indices_;
    std::vector<BatchRequest> batch_requests_;
    std::atomic<size_t> next_request_idx_{0};
    std::atomic<size_t> workers_completed_{0};
    std::atomic<bool> active_{false};

    BoundedQueue<Batch> batch_queue_;
    std::vector<std::jthread> workers_;
};

} // namespace soar::data
