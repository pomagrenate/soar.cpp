#pragma once

#include <soar/autograd/node.hpp>
#include <soar/tensor/tensor.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <memory>
#include <mutex>

namespace soar::autograd {

/**
 * @brief AccumulateGrad leaf node mirroring PyTorch torch::autograd::AccumulateGrad.
 *
 * Implements zero-overhead in-place gradient accumulation directly into
 * the leaf parameter's gradient buffer, avoiding any temporary memory allocations.
 */
class AccumulateGradNode : public AutogradNode {
public:
    explicit AccumulateGradNode(TensorPtr variable)
        : variable_(std::move(variable)) {}

    void backward(const TensorPtr& grad_output) override;

    [[nodiscard]] std::string name() const override { return "AccumulateGrad"; }

    [[nodiscard]] TensorPtr variable() const noexcept { return variable_; }

    void release_variables() override {
        // Keep variable_ reference alive for parameter updates
    }

private:
    TensorPtr variable_;
    std::mutex mutex_;
};

/**
 * @brief Production autograd backward execution engine mirroring torch/csrc/autograd/engine.cpp.
 *
 * Implements:
 * - Topological dependency computation (`compute_dependencies`)
 * - Kahn's topological queue scheduling
 * - Thread-safe ready queue dispatching
 * - Immediate activation release to maximize memory reuse
 */
class Engine {
public:
    static Engine& get_default_engine();

    /**
     * @brief Computes in-degrees for all reachable functions in the backward graph DAG.
     */
    void compute_dependencies(
        AutogradNode* root,
        std::unordered_map<AutogradNode*, size_t>& dependencies,
        std::unordered_map<AutogradNode*, std::vector<std::shared_ptr<AutogradNode>>>& graph_edges
    );

    /**
     * @brief Execute the reverse-mode automatic differentiation DAG.
     * @param root The root backward node (typically the scalar loss grad_fn).
     * @param root_grad The initial gradient (typically 1.0 for scalar losses).
     */
    void execute(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad);

private:
    std::mutex engine_mutex_;
};

} // namespace soar::autograd
