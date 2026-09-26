#pragma once

#include <soar/data/example.hpp>
#include <cstddef>
#include <optional>
#include <vector>

namespace soar::data::datasets {

/// A dataset that can yield data in batches.
/// Matches PyTorch torch::data::datasets::BatchDataset.
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename BatchRequest = std::vector<size_t>>
class BatchDataset {
public:
    using SelfType = Self;
    using BatchType = Batch;
    using BatchRequestType = BatchRequest;
    constexpr static bool is_stateful = false;

    virtual ~BatchDataset() = default;

    /// Returns a batch of data given a batch request.
    virtual Batch get_batch(BatchRequest request) = 0;

    /// Returns the size of the dataset, or std::nullopt if unsized.
    [[nodiscard]] virtual std::optional<size_t> size() const = 0;
};

/// A dataset supporting random access and batched access.
/// Matches PyTorch torch::data::datasets::Dataset.
template <typename Self, typename SingleExample = Example<>>
class Dataset : public BatchDataset<Self, std::vector<SingleExample>, std::vector<size_t>> {
public:
    using ExampleType = SingleExample;

    /// Returns the example at the given index.
    virtual ExampleType get(size_t index) = 0;

    /// Default implementation: gathers single examples into a batch.
    std::vector<ExampleType> get_batch(std::vector<size_t> indices) override {
        std::vector<ExampleType> batch;
        batch.reserve(indices.size());
        for (const auto i : indices) {
            batch.push_back(get(i));
        }
        return batch;
    }
};

} // namespace soar::data::datasets

namespace soar::data {
using datasets::BatchDataset;
} // namespace soar::data
