#include <soar/tensor/tensor_iterator.hpp>
#include <soar/core/logging.hpp>
#include <algorithm>
#include <thread>
#include <omp.h>

namespace soar {

TensorIterator::TensorIterator() = default;

void TensorIterator::add_input(const TensorPtr& tensor) {
    if (!tensor) {
        throw std::invalid_argument("Cannot add null input tensor");
    }
    inputs_.push_back(tensor);
}

void TensorIterator::add_output(const TensorPtr& tensor) {
    if (!tensor) {
        throw std::invalid_argument("Cannot add null output tensor");
    }
    outputs_.push_back(tensor);
}

bool TensorIterator::is_contiguous() const {
    // Simplified: assume contiguous for now
    // Full implementation would check strides
    return true;
}

size_t TensorIterator::numel() const {
    if (total_elements_ == 0) {
        const_cast<TensorIterator*>(this)->compute_broadcast_shape();
    }
    return total_elements_;
}

void TensorIterator::compute_broadcast_shape() {
    if (inputs_.empty() && outputs_.empty()) {
        total_elements_ = 0;
        return;
    }

    // Start with first tensor's shape
    core::Shape result_shape = inputs_.empty() ? outputs_[0]->shape() : inputs_[0]->shape();
    
    // Broadcast all inputs
    for (const auto& tensor : inputs_) {
        const auto& shape = tensor->shape();
        // Simplified: only support same-size tensors for now
        // Full broadcasting would implement NumPy-style rules
        if (shape.numel() != result_shape.numel()) {
            SOAR_LOG_WARN("TensorIterator: input shapes don't match, using first tensor size");
        }
    }
    
    // Broadcast all outputs
    for (const auto& tensor : outputs_) {
        const auto& shape = tensor->shape();
        if (shape.numel() != result_shape.numel()) {
            SOAR_LOG_WARN("TensorIterator: output shape doesn't match inputs");
        }
    }
    
    broadcast_shape_ = result_shape;
    total_elements_ = result_shape.numel();
    
    // Check contiguity
    is_contiguous_ = is_contiguous();
}

void TensorIterator::for_each(const std::function<void(const std::vector<float*>& inputs, float* output)>& op,
                              bool parallel) {
    compute_broadcast_shape();
    
    if (total_elements_ == 0) {
        return;
    }
    
    if (is_contiguous_) {
        if (parallel && total_elements_ > 1024) {
            for_each_parallel(op, 0, total_elements_);
        } else {
            for_each_contiguous(op);
        }
    } else {
        for_each_strided(op);
    }
}

void TensorIterator::for_each_contiguous(const std::function<void(const std::vector<float*>& inputs, float* output)>& op) {
    // Fast path: all tensors are contiguous
    std::vector<float*> input_ptrs;
    input_ptrs.reserve(inputs_.size());
    
    for (const auto& tensor : inputs_) {
        input_ptrs.push_back(tensor->data());
    }
    
    float* output_ptr = outputs_[0]->data();
    
    for (size_t i = 0; i < total_elements_; ++i) {
        std::vector<float*> offset_ptrs;
        offset_ptrs.reserve(input_ptrs.size());
        for (auto ptr : input_ptrs) {
            offset_ptrs.push_back(ptr + i);
        }
        op(offset_ptrs, output_ptr + i);
    }
}

void TensorIterator::for_each_strided(const std::function<void(const std::vector<float*>& inputs, float* output)>& op) {
    // Slow path: strided access
    // For now, simplified to just use contiguous approach
    // Full implementation would compute multi-dimensional indices
    SOAR_LOG_WARN("TensorIterator: strided path not fully implemented, using simplified access");
    for_each_contiguous(op);
}

void TensorIterator::for_each_parallel(const std::function<void(const std::vector<float*>& inputs, float* output)>& op,
                                        size_t start, size_t end) {
    // Parallel execution using OpenMP
    std::vector<float*> input_ptrs;
    input_ptrs.reserve(inputs_.size());
    
    for (const auto& tensor : inputs_) {
        input_ptrs.push_back(tensor->data());
    }
    
    float* output_ptr = outputs_[0]->data();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = start; i < end; ++i) {
        std::vector<float*> offset_ptrs;
        offset_ptrs.reserve(input_ptrs.size());
        for (auto ptr : input_ptrs) {
            offset_ptrs.push_back(ptr + i);
        }
        op(offset_ptrs, output_ptr + i);
    }
}

void TensorIterator::for_each_binary(const std::function<void(float a, float b, float& out)>& op,
                                      bool parallel) {
    if (inputs_.size() != 2 || outputs_.size() != 1) {
        throw std::invalid_argument("Binary operation requires exactly 2 inputs and 1 output");
    }
    
    for_each([&op](const std::vector<float*>& inputs, float* output) {
        op(*inputs[0], *inputs[1], *output);
    }, parallel);
}

void TensorIterator::for_each_unary(const std::function<void(float a, float& out)>& op,
                                    bool parallel) {
    if (inputs_.size() != 1 || outputs_.size() != 1) {
        throw std::invalid_argument("Unary operation requires exactly 1 input and 1 output");
    }
    
    for_each([&op](const std::vector<float*>& inputs, float* output) {
        op(*inputs[0], *output);
    }, parallel);
}

TensorIterator make_binary_iterator(const TensorPtr& a, const TensorPtr& b, const TensorPtr& out) {
    TensorIterator iter;
    iter.add_input(a);
    iter.add_input(b);
    iter.add_output(out);
    return iter;
}

TensorIterator make_unary_iterator(const TensorPtr& a, const TensorPtr& out) {
    TensorIterator iter;
    iter.add_input(a);
    iter.add_output(out);
    return iter;
}

} // namespace soar