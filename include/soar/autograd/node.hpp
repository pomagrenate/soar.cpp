#pragma once

#include <memory>
#include <vector>
#include <string>

namespace soar {

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

/**
 * @brief Base class for nodes in the dynamic automatic differentiation computational graph.
 */
struct AutogradNode : public std::enable_shared_from_this<AutogradNode> {
    virtual ~AutogradNode() = default;

    /**
     * @brief Execute reverse-mode differentiation for this node.
     * @param grad_output Gradient of the scalar loss with respect to this node's output.
     */
    virtual void backward(const TensorPtr& grad_output) = 0;

    [[nodiscard]] virtual std::string name() const { return "AutogradNode"; }

    // List of input nodes that feed into this node
    std::vector<std::shared_ptr<AutogradNode>> next_nodes;
};

/**
 * @brief Helper to accumulate gradients safely.
 */
inline void accumulate_grad(const TensorPtr& target, const TensorPtr& incoming) {
    if (target) {
        target->add_grad(incoming);
    }
}

} // namespace soar
