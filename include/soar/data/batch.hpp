#pragma once

#include <soar/tensor/tensor.hpp>
#include <soar/data/coco_dataset.hpp>
#include <vector>

namespace soar::data {

/// A batched tensor representation produced by the DataLoader.
struct Batch {
    TensorPtr data{nullptr};
    TensorPtr target{nullptr};
    std::vector<DatasetSample> samples{};
    size_t sequence_id{0};
    bool is_pinned{false};
};

} // namespace soar::data
