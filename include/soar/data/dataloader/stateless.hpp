#pragma once

#include <soar/data/dataloader/base.hpp>
#include <soar/data/worker_exception.hpp>
#include <cassert>
#include <cstddef>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

namespace soar::data {

/// A dataloader for stateless datasets.
/// Matches PyTorch torch::data::StatelessDataLoader<Dataset, Sampler>.
template <typename Dataset, typename Sampler>
class StatelessDataLoader : public DataLoaderBase<
                                Dataset,
                                typename Dataset::BatchType,
                                typename Sampler::BatchRequestType> {
public:
    using super = DataLoaderBase<
        Dataset,
        typename Dataset::BatchType,
        typename Sampler::BatchRequestType>;
    using typename super::BatchRequestType;

    StatelessDataLoader(
        Dataset dataset,
        Sampler sampler,
        DataLoaderOptions options)
        : super(options), sampler_(std::move(sampler)) {
        for (size_t w = 0; w < this->options_.workers; ++w) {
            // Each worker closure holds its own copy/reference to dataset
            this->workers_.emplace_back(
                [this, dataset]() mutable { this->worker_thread(dataset); });
        }
        if (this->options_.workers == 0) {
            this->main_thread_dataset_ =
                std::make_unique<Dataset>(std::move(dataset));
        }
    }

private:
    void reset() override {
        sampler_.reset();
        super::reset();
    }

    std::optional<BatchRequestType> get_batch_request() override {
        auto indices = sampler_.next(this->options_.batch_size);
        if (!indices ||
            (indices->size() < this->options_.batch_size &&
             this->options_.drop_last)) {
            return std::nullopt;
        }
        assert(!indices->empty());
        return indices;
    }

    Sampler sampler_;
};

} // namespace soar::data
