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

    /**
     * @brief Return the input nodes in the forward graph (which are the dependent nodes in reverse pass).
     */
    virtual std::vector<std::shared_ptr<AutogradNode>> get_inputs() const { return {}; }

    [[nodiscard]] virtual std::string name() const { return "AutogradNode"; }

    /**
     * @brief Release saved forward variables to reclaim memory immediately upon backward completion.
     */
    virtual void release_variables() {}

    // List of input nodes that feed into this node
    std::vector<std::shared_ptr<AutogradNode>> next_nodes;

    // Accumulated gradient arriving at this node during reverse traversal
    TensorPtr grad_output_{nullptr};
    void add_grad_output(const TensorPtr& incoming);
};

void propagate_grad(const TensorPtr& tensor, const TensorPtr& incoming);
void run_backward(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad);

/**
 * @brief Helper to accumulate gradients safely.
 */
inline void accumulate_grad(const TensorPtr& target, const TensorPtr& incoming) {
    propagate_grad(target, incoming);
}

/**
 * @brief Thread-safe and scoped control over automatic differentiation graph building.
 */
class GradMode {
private:
    static inline thread_local bool enabled_{true};
public:
    static bool is_enabled() noexcept { return enabled_; }
    static void set_enabled(bool enabled) noexcept { enabled_ = enabled; }
};

class NoGradGuard {
private:
    bool prev_;
public:
    NoGradGuard() noexcept : prev_(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }
    ~NoGradGuard() noexcept {
        GradMode::set_enabled(prev_);
    }
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
};

} // namespace soar
