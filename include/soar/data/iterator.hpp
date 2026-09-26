#pragma once

#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace soar::data {
namespace detail {

template <typename Batch>
struct ValidIterator;

template <typename Batch>
struct SentinelIterator;

/// Base class for ValidIterator and SentinelIterator.
template <typename Batch>
struct IteratorImpl {
    virtual ~IteratorImpl() = default;
    virtual void next() = 0;
    virtual Batch& get() = 0;
    virtual bool operator==(const IteratorImpl& other) const = 0;
    virtual bool operator==(const ValidIterator<Batch>& other) const = 0;
    virtual bool operator==(const SentinelIterator<Batch>& other) const = 0;
};

template <typename Batch>
struct ValidIterator : public IteratorImpl<Batch> {
    using BatchProducer = std::function<std::optional<Batch>()>;

    explicit ValidIterator(BatchProducer next_batch)
        : next_batch_(std::move(next_batch)) {}

    void next() override {
        lazy_initialize();
        if (!batch_.has_value()) {
            throw std::runtime_error("Attempted to increment DataLoader iterator past the end");
        }
        batch_ = next_batch_();
    }

    Batch& get() override {
        lazy_initialize();
        if (!batch_.has_value()) {
            throw std::runtime_error("Attempted to dereference DataLoader iterator that was past the end");
        }
        return batch_.value();
    }

    bool operator==(const IteratorImpl<Batch>& other) const override {
        return other == *this;
    }

    bool operator==(const SentinelIterator<Batch>&) const override {
        lazy_initialize();
        return !batch_.has_value();
    }

    bool operator==(const ValidIterator<Batch>& other) const override {
        return &other == this;
    }

    void lazy_initialize() const {
        if (!initialized_) {
            batch_ = next_batch_();
            initialized_ = true;
        }
    }

    BatchProducer next_batch_;
    mutable std::optional<Batch> batch_;
    mutable bool initialized_ = false;
};

template <typename Batch>
struct SentinelIterator : public IteratorImpl<Batch> {
    void next() override {
        throw std::runtime_error("Incrementing DataLoader past-the-end sentinel iterator is not allowed");
    }

    Batch& get() override {
        throw std::runtime_error("Dereferencing DataLoader past-the-end sentinel iterator is not allowed");
    }

    bool operator==(const IteratorImpl<Batch>& other) const override {
        return other == *this;
    }

    bool operator==(const ValidIterator<Batch>& other) const override {
        return other == *this;
    }

    bool operator==(const SentinelIterator<Batch>&) const override {
        return true;
    }
};

} // namespace detail

/// Input iterator for DataLoader batches matching PyTorch's torch::data::Iterator<Batch>.
template <typename Batch>
class Iterator {
public:
    using difference_type = std::ptrdiff_t;
    using value_type = Batch;
    using pointer = Batch*;
    using reference = Batch&;
    using iterator_category = std::input_iterator_tag;

    explicit Iterator(std::unique_ptr<detail::IteratorImpl<Batch>> impl)
        : impl_(std::move(impl)) {}

    Iterator& operator++() {
        impl_->next();
        return *this;
    }

    Batch& operator*() {
        return impl_->get();
    }

    Batch* operator->() {
        return &impl_->get();
    }

    bool operator==(const Iterator& other) const {
        return *impl_ == *other.impl_;
    }

    bool operator!=(const Iterator& other) const {
        return !(*this == other);
    }

private:
    std::shared_ptr<detail::IteratorImpl<Batch>> impl_;
};

} // namespace soar::data
