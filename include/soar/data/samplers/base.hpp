#pragma once

#include <cstddef>
#include <optional>
#include <vector>

namespace soar::data::samplers {

/// A Sampler yields batch requests (indices) to access a dataset.
/// Matches PyTorch torch::data::samplers::Sampler<BatchRequest>.
template <typename BatchRequest = std::vector<size_t>>
class Sampler {
public:
    using BatchRequestType = BatchRequest;

    virtual ~Sampler() = default;

    /// Resets the sampler's internal state, optionally updating total size.
    virtual void reset(std::optional<size_t> new_size = std::nullopt) = 0;

    /// Returns the next batch request or std::nullopt if exhausted.
    virtual std::optional<BatchRequest> next(size_t batch_size) = 0;
};

} // namespace soar::data::samplers
