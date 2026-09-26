#pragma once

#include <soar/tensor/tensor.hpp>
#include <utility>

namespace soar::data {

/// An Example from a dataset containing data and an associated target (label).
/// Matches PyTorch torch::data::Example<Data, Target>.
template <typename Data = TensorPtr, typename Target = TensorPtr>
struct Example {
    using DataType = Data;
    using TargetType = Target;

    Example() = default;
    Example(Data d, Target t) : data(std::move(d)), target(std::move(t)) {}

    Data data{};
    Target target{};
};

namespace example {
using NoTarget = void;
} // namespace example

/// Specialization for unlabeled examples.
template <typename Data>
struct Example<Data, example::NoTarget> {
    using DataType = Data;
    using TargetType = example::NoTarget;

    Example() = default;
    /* implicit */ Example(Data d) : data(std::move(d)) {}

    operator Data&() { return data; }
    operator const Data&() const { return data; }

    Data data{};
};

} // namespace soar::data
