#pragma once

#include <soar/tensor/tensor.hpp>
#include <soar/cuda/cuda_runtime.hpp>
#include <soar/data/batch.hpp>
#include <soar/data/example.hpp>
#include <soar/data/iterator.hpp>
#include <soar/data/worker_exception.hpp>
#include <soar/data/dataloader_options.hpp>
#include <soar/data/samplers.hpp>
#include <soar/data/datasets/base.hpp>
#include <soar/data/dataloader/base.hpp>
#include <soar/data/dataloader/stateless.hpp>
#include <soar/data/coco_dataset.hpp>
#include <soar/data/yolo_dataset.hpp>

#include <vector>
#include <memory>
#include <thread>
#include <functional>
#include <cstring>
#include <span>
#include <optional>
#include <type_traits>
#include <cassert>

namespace soar::data {

/**
 * @brief Abstract Dataset interface for legacy / synthetic dataset consumption.
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
 * @brief Adapter to wrap any Dataset or SampleDataset into PyTorch's BatchDataset interface.
 */
struct DatasetBatchAdapter {
    using BatchType = Batch;
    using BatchRequestType = std::vector<size_t>;

    std::shared_ptr<Dataset> dataset{nullptr};
    std::function<DatasetSample(size_t)> sample_fetcher{nullptr};
    size_t total_size{0};
    bool pin_memory{false};

    [[nodiscard]] std::optional<size_t> size() const noexcept {
        if (dataset) return dataset->size();
        return total_size;
    }

    Batch get_batch(const std::vector<size_t>& indices) const {
        size_t actual_bsz = indices.size();
        Batch b;
        b.is_pinned = pin_memory;
        if (actual_bsz == 0) return b;

        if (sample_fetcher) {
            std::vector<DatasetSample> samples;
            samples.reserve(actual_bsz);
            for (size_t idx : indices) {
                samples.push_back(sample_fetcher(idx));
            }
            if (actual_bsz == 1) {
                b.data = samples[0].image;
                b.target = samples[0].mask;
                b.samples = std::move(samples);
            } else {
                size_t C = samples[0].image->dim(0);
                size_t H = samples[0].image->dim(1);
                size_t W = samples[0].image->dim(2);
                size_t item_numel = C * H * W;
                size_t mask_item_numel = H * W;

                b.data = Tensor::create({static_cast<int64_t>(actual_bsz),
                                        static_cast<int64_t>(C),
                                        static_cast<int64_t>(H),
                                        static_cast<int64_t>(W)}, false);
                b.target = Tensor::create({static_cast<int64_t>(actual_bsz),
                                          1,
                                          static_cast<int64_t>(H),
                                          static_cast<int64_t>(W)}, false);
                float* dst_img = b.data->data();
                float* dst_msk = b.target->data();

                for (size_t i = 0; i < actual_bsz; ++i) {
                    std::memcpy(dst_img + i * item_numel, samples[i].image->data(), item_numel * sizeof(float));
                    std::memcpy(dst_msk + i * mask_item_numel, samples[i].mask->data(), mask_item_numel * sizeof(float));
                }
                b.samples = std::move(samples);
            }
        } else if (dataset) {
            auto item_data_shape = dataset->data_shape();
            auto item_target_shape = dataset->target_shape();

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
                size_t sample_idx = indices[i];
                std::span<float> item_data_span(data_ptr + i * data_item_numel, data_item_numel);
                std::span<float> item_target_span(target_ptr + i * target_item_numel, target_item_numel);
                dataset->get_item(sample_idx, item_data_span, item_target_span);
            }

            b.data = std::move(data_tensor);
            b.target = std::move(target_tensor);
        }

        return b;
    }
};

/**
 * @brief High-throughput asynchronous prefetching DataLoader built on PyTorch's architecture.
 */
class DataLoader {
public:
    template <typename T>
        requires std::is_convertible_v<T*, Dataset*>
    DataLoader(std::shared_ptr<T> dataset, DataLoaderOptions options = {})
        : options_(options) {
        DatasetBatchAdapter adapter;
        adapter.dataset = std::static_pointer_cast<Dataset>(dataset);
        adapter.pin_memory = options.pin_memory;
        adapter.total_size = adapter.dataset ? adapter.dataset->size() : 0;
        init_loader(std::move(adapter));
    }

    template <typename SampleDataset>
        requires (!std::is_convertible_v<SampleDataset, std::shared_ptr<Dataset>>)
    DataLoader(const SampleDataset& ds, DataLoaderOptions options = {}, std::vector<size_t> subset_indices = {})
        : options_(options) {
        DatasetBatchAdapter adapter;
        adapter.pin_memory = options.pin_memory;
        if (subset_indices.empty()) {
            adapter.total_size = ds.size();
            adapter.sample_fetcher = [&ds](size_t idx) { return ds.get_sample(idx); };
        } else {
            adapter.total_size = subset_indices.size();
            adapter.sample_fetcher = [&ds, indices = std::move(subset_indices)](size_t idx) {
                return ds.get_sample(indices[idx]);
            };
        }
        init_loader(std::move(adapter));
    }

    Iterator<Batch> begin() {
        return loader_->begin();
    }

    Iterator<Batch> end() {
        return loader_->end();
    }

    [[nodiscard]] size_t total_batches() const {
        size_t n = total_samples();
        if (n == 0 || options_.batch_size == 0) return 0;
        if (options_.drop_last) {
            return n / options_.batch_size;
        }
        return (n + options_.batch_size - 1) / options_.batch_size;
    }

    [[nodiscard]] size_t total_samples() const {
        return total_samples_;
    }

    [[nodiscard]] const DataLoaderOptions& options() const noexcept {
        return options_;
    }

private:
    void init_loader(DatasetBatchAdapter adapter) {
        total_samples_ = adapter.size().value_or(0);
        if (options_.shuffle) {
            loader_ = std::make_unique<StatelessDataLoader<DatasetBatchAdapter, samplers::RandomSampler>>(
                adapter, samplers::RandomSampler(total_samples_), options_);
        } else {
            loader_ = std::make_unique<StatelessDataLoader<DatasetBatchAdapter, samplers::SequentialSampler>>(
                adapter, samplers::SequentialSampler(total_samples_), options_);
        }
    }

    DataLoaderOptions options_;
    size_t total_samples_{0};
    std::unique_ptr<DataLoaderBase<DatasetBatchAdapter, Batch, std::vector<size_t>>> loader_;
};

/**
 * @brief Creates a DataLoader instance for a dataset, sampler, and options.
 * Matches PyTorch torch::data::make_data_loader.
 */
template <typename DatasetType, typename Sampler>
inline std::unique_ptr<StatelessDataLoader<DatasetType, Sampler>> make_data_loader(
    DatasetType dataset, Sampler sampler, DataLoaderOptions options) {
    return std::make_unique<StatelessDataLoader<DatasetType, Sampler>>(
        std::move(dataset), std::move(sampler), options);
}

/**
 * @brief Creates a DataLoader instance for a dataset with default RandomSampler.
 * Matches PyTorch torch::data::make_data_loader.
 */
template <typename Sampler = samplers::RandomSampler, typename DatasetType>
inline std::unique_ptr<StatelessDataLoader<DatasetType, Sampler>> make_data_loader(
    DatasetType dataset, DataLoaderOptions options = DataLoaderOptions()) {
    std::optional<size_t> sz = dataset.size();
    assert(sz.has_value() && "Expected dataset to have a size to construct Sampler");
    return make_data_loader(std::move(dataset), Sampler(*sz), options);
}

/**
 * @brief Overload for DataLoader with subset indices.
 */
template <typename DatasetType>
inline DataLoader make_data_loader(
    const DatasetType& dataset,
    DataLoaderOptions options,
    std::vector<size_t> subset_indices) {
    return DataLoader(dataset, options, std::move(subset_indices));
}

} // namespace soar::data
