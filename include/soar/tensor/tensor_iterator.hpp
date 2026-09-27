#pragma once

#include <soar/tensor/tensor.hpp>
#include <soar/core/shape.hpp>
#include <vector>
#include <functional>
#include <memory>

namespace soar {

/**
 * @brief Simplified TensorIterator for elementwise operations.
 * 
 * Inspired by PyTorch's TensorIterator, this provides efficient
 * elementwise traversal for 2048×2048 high-resolution workloads.
 * 
 * Simplified compared to PyTorch's full implementation:
 * - Focuses on contiguous tensors first
 * - Supports basic broadcasting for common cases
 * - Optimized for float32 operations
 * - Provides parallel execution for large buffers
 */
class TensorIterator {
public:
    TensorIterator();
    ~TensorIterator() = default;

    // Non-copyable, movable
    TensorIterator(const TensorIterator&) = delete;
    TensorIterator& operator=(const TensorIterator&) = delete;
    TensorIterator(TensorIterator&&) = default;
    TensorIterator& operator=(TensorIterator&&) = default;

    /**
     * @brief Add an input tensor to the iterator.
     */
    void add_input(const TensorPtr& tensor);

    /**
     * @brief Add an output tensor to the iterator.
     */
    void add_output(const TensorPtr& tensor);

    /**
     * @brief Check if all tensors are contiguous.
     */
    bool is_contiguous() const;

    /**
     * @brief Get the total number of elements to iterate over.
     */
    size_t numel() const;

    /**
     * @brief Execute an elementwise operation.
     * 
     * @param op Function that takes pointers to input and output tensors
     * @param parallel Whether to use parallel execution for large buffers
     */
    void for_each(const std::function<void(const std::vector<float*>& inputs, float* output)>& op, 
                  bool parallel = true);

    /**
     * @brief Execute a binary operation with two inputs and one output.
     */
    void for_each_binary(const std::function<void(float a, float b, float& out)>& op,
                         bool parallel = true);

    /**
     * @brief Execute a unary operation with one input and one output.
     */
    void for_each_unary(const std::function<void(float a, float& out)>& op,
                        bool parallel = true);

private:
    /**
     * @brief Validate tensor shapes and compute broadcast shape.
     */
    void compute_broadcast_shape();

    /**
     * @brief Execute operation on contiguous tensors (fast path).
     */
    void for_each_contiguous(const std::function<void(const std::vector<float*>& inputs, float* output)>& op);

    /**
     * @brief Execute operation with strided access (slow path).
     */
    void for_each_strided(const std::function<void(const std::vector<float*>& inputs, float* output)>& op);

    /**
     * @brief Parallel execution for large contiguous buffers.
     */
    void for_each_parallel(const std::function<void(const std::vector<float*>& inputs, float* output)>& op,
                           size_t start, size_t end);

    std::vector<TensorPtr> inputs_;
    std::vector<TensorPtr> outputs_;
    core::Shape broadcast_shape_;
    size_t total_elements_{0};
    bool is_contiguous_{true};
};

/**
 * @brief Helper to create a TensorIterator for binary operations.
 */
TensorIterator make_binary_iterator(const TensorPtr& a, const TensorPtr& b, const TensorPtr& out);

/**
 * @brief Helper to create a TensorIterator for unary operations.
 */
TensorIterator make_unary_iterator(const TensorPtr& a, const TensorPtr& out);

} // namespace soar