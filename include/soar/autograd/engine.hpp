#pragma once

#include <soar/autograd/node.hpp>
#include <soar/tensor/tensor.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <memory>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <future>

namespace soar::autograd {

// Maximum reentrant backward depth before switching to a new thread
// This limit prevents excessive lock nesting and potential deadlocks
static constexpr int MAX_REENTRANT_DEPTH = 60;

/**
 * @brief Graph task metadata for backward execution coordination.
 *
 * Similar to PyTorch's GraphTask, this holds metadata needed for a single
 * execution of backward(), including dependency tracking and error handling.
 */
struct GraphTask {
    std::atomic<uint64_t> outstanding_tasks_{0};
    std::atomic_bool has_error_{false};
    bool keep_graph_{true};
    
    std::mutex mutex_;
    std::unordered_map<AutogradNode*, size_t> dependencies_;
    std::unordered_set<AutogradNode*> nodes_in_graph_;
    
    void mark_error() { has_error_ = true; }
    bool has_error() const { return has_error_.load(); }
};

/**
 * @brief Thread-safe ready queue for autograd node execution.
 *
 * Implements priority-based task scheduling similar to PyTorch's ReadyQueue,
 * with support for device-specific queues and shutdown signaling.
 */
class ReadyQueue {
public:
    ReadyQueue() = default;
    ~ReadyQueue() = default;
    
    void push(std::shared_ptr<AutogradNode> node);
    std::shared_ptr<AutogradNode> pop();
    void shutdown();
    bool is_shutdown() const { return shutdown_.load(); }
    
private:
    std::queue<std::shared_ptr<AutogradNode>> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_{false};
};

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
 * @brief Enhanced autograd backward execution engine with thread pool support.
 *
 * Implements PyTorch-inspired optimizations:
 * - Thread pool execution with device-specific ready queues
 * - Reentrant backward support with depth limiting
 * - Graph task metadata for better coordination
 * - Thread-safe ready queue dispatching
 * - Immediate activation release to maximize memory reuse
 * - Phase 3.2: Eager activation release during backward
 */
class Engine {
public:
    static Engine& get_default_engine();
    
    ~Engine();

    /**
     * @brief Computes in-degrees for all reachable functions in the backward graph DAG.
     */
    void compute_dependencies(
        AutogradNode* root,
        std::unordered_map<AutogradNode*, size_t>& dependencies,
        std::unordered_map<AutogradNode*, std::vector<std::shared_ptr<AutogradNode>>>& graph_edges
    );

    /**
     * @brief Execute the reverse-mode automatic differentiation DAG with thread pool.
     * @param root The root backward node (typically the scalar loss grad_fn).
     * @param root_grad The initial gradient (typically 1.0 for scalar losses).
     */
    void execute(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad);
    
    /**
     * @brief Execute a single node task (used by worker threads).
     */
    void execute_node_task(std::shared_ptr<AutogradNode> node, std::shared_ptr<GraphTask> graph_task);
    
    /**
     * @brief Sequential execution fallback for reentrant backward or simple cases.
     */
    void execute_sequential(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad);
    
    /**
     * @brief Enable/disable eager activation release during backward.
     * @param enable If true, releases activations immediately after all consumers have received gradients.
     */
    void set_eager_activation_release(bool enable) noexcept { eager_activation_release_ = enable; }
    [[nodiscard]] bool eager_activation_release_enabled() const noexcept { return eager_activation_release_; }

private:
    Engine();
    
    /**
     * @brief Initialize the thread pool for parallel backward execution.
     */
    void init_thread_pool();
    
    /**
     * @brief Worker thread function for processing ready queue.
     */
    void worker_thread_func(std::shared_ptr<ReadyQueue> queue);
    
    /**
     * @brief Get the CPU-specific ready queue.
     */
    std::shared_ptr<ReadyQueue> get_cpu_ready_queue();
    
    /**
     * @brief Release activations for a node after all consumers have been processed.
     * This is critical for 2048×2048 workloads to reduce VRAM usage.
     */
    void release_node_activations(std::shared_ptr<AutogradNode> node);
    
    std::recursive_mutex engine_mutex_;
    std::shared_ptr<ReadyQueue> cpu_ready_queue_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutdown_{false};
    
    // Phase 3.2: Eager activation release state
    bool eager_activation_release_{true};
    std::unordered_set<AutogradNode*> released_nodes_;
    
    // Thread-local reentrant depth tracking
    static thread_local int current_depth;
};

} // namespace soar::autograd
