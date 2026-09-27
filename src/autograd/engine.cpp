#include <soar/autograd/engine.hpp>
#include <iostream>
#include <chrono>

namespace soar::autograd {

// Thread-local reentrant depth tracking
thread_local int Engine::current_depth = 0;

// ReadyQueue implementation
void ReadyQueue::push(std::shared_ptr<AutogradNode> node) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!shutdown_.load()) {
        queue_.push(std::move(node));
        cv_.notify_one();
    }
}

std::shared_ptr<AutogradNode> ReadyQueue::pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || shutdown_.load(); });
    
    if (shutdown_.load() && queue_.empty()) {
        return nullptr;
    }
    
    auto node = queue_.front();
    queue_.pop();
    return node;
}

void ReadyQueue::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = true;
    cv_.notify_all();
}

void AccumulateGradNode::backward(const TensorPtr& grad_output) {
    if (!variable_ || !grad_output) return;

    std::lock_guard<std::mutex> lock(mutex_);
    auto& param_grad = variable_->grad_ref();

    if (!param_grad) {
        // Zero-overhead steal or direct clone
        param_grad = grad_output->clone();
    } else {
        // In-place gradient accumulation without any new tensor allocation
        float* dst = param_grad->data();
        const float* src = grad_output->data();
        size_t n = std::min(param_grad->numel(), grad_output->numel());

        #pragma omp parallel for if (n > 4096)
        for (size_t i = 0; i < n; ++i) {
            dst[i] += src[i];
        }
    }
}

Engine::Engine() {
    // Initialize thread pool on first use
    init_thread_pool();
}

Engine::~Engine() {
    shutdown_ = true;
    if (cpu_ready_queue_) {
        cpu_ready_queue_->shutdown();
    }
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

Engine& Engine::get_default_engine() {
    static Engine s_engine;
    return s_engine;
}

void Engine::init_thread_pool() {
    if (initialized_.exchange(true)) {
        return; // Already initialized
    }
    
    // Create CPU ready queue
    cpu_ready_queue_ = std::make_shared<ReadyQueue>();
    
    // Determine number of worker threads (use hardware concurrency, but cap at 4 for now)
    unsigned int num_threads = std::min(4u, std::thread::hardware_concurrency());
    if (num_threads == 0) num_threads = 1;
    
    // Launch worker threads
    for (unsigned int i = 0; i < num_threads; ++i) {
        worker_threads_.emplace_back(&Engine::worker_thread_func, this, cpu_ready_queue_);
    }
}

void Engine::worker_thread_func(std::shared_ptr<ReadyQueue> queue) {
    while (!shutdown_.load()) {
        auto node = queue->pop();
        if (!node) {
            break; // Shutdown signal
        }
        
        // Execute the node (this is a simplified version - in full implementation
        // we'd need to associate this with a GraphTask)
        try {
            // For now, execute directly - full thread pool integration would need
            // more sophisticated task management
            TensorPtr grad_output = node->grad_output_;
            node->grad_output_ = nullptr;
            node->backward(grad_output);
            node->release_variables();
        } catch (const std::exception& e) {
            std::cerr << "Error in autograd worker thread: " << e.what() << std::endl;
        }
    }
}

std::shared_ptr<ReadyQueue> Engine::get_cpu_ready_queue() {
    return cpu_ready_queue_;
}

void Engine::compute_dependencies(
    AutogradNode* root,
    std::unordered_map<AutogradNode*, size_t>& dependencies,
    std::unordered_map<AutogradNode*, std::vector<std::shared_ptr<AutogradNode>>>& graph_edges) {

    if (!root) return;

    std::vector<AutogradNode*> queue{root};
    std::unordered_set<AutogradNode*> seen{root};

    while (!queue.empty()) {
        auto fn = queue.back();
        queue.pop_back();

        for (const auto& next_fn : fn->get_inputs()) {
            if (!next_fn) continue;
            graph_edges[fn].push_back(next_fn);
            dependencies[next_fn.get()] += 1;
            if (seen.insert(next_fn.get()).second) {
                queue.push_back(next_fn.get());
            }
        }
    }
}

void Engine::execute(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad) {
    if (!root) return;

    // Check reentrant depth - if too deep, execute sequentially to avoid excessive locking
    if (current_depth >= MAX_REENTRANT_DEPTH) {
        execute_sequential(root, root_grad);
        return;
    }

    // Create graph task for this execution
    auto graph_task = std::make_shared<GraphTask>();
    
    std::lock_guard<std::mutex> lock(engine_mutex_);

    std::unordered_map<AutogradNode*, size_t> dependencies;
    std::unordered_map<AutogradNode*, std::vector<std::shared_ptr<AutogradNode>>> graph_edges;

    compute_dependencies(root.get(), dependencies, graph_edges);
    
    // Store dependencies in graph task
    graph_task->dependencies_ = dependencies;
    graph_task->nodes_in_graph_.insert(root.get());
    for (const auto& [node, deps] : dependencies) {
        graph_task->nodes_in_graph_.insert(node);
    }

    root->add_grad_output(root_grad);
    
    // For now, use sequential execution as baseline
    // Full thread pool integration would require more sophisticated coordination
    execute_sequential(root, root_grad);
}

void Engine::release_node_activations(std::shared_ptr<AutogradNode> node) {
    if (!eager_activation_release_) return;
    if (!node) return;
    
    std::lock_guard<std::mutex> lock(engine_mutex_);
    
    // Mark this node as released to prevent double-release
    if (released_nodes_.find(node.get()) != released_nodes_.end()) {
        return;
    }
    released_nodes_.insert(node.get());
    
    // Release the node's saved activations
    node->release_variables();
}

void Engine::execute_sequential(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad) {
    current_depth++;
    
    std::unordered_map<AutogradNode*, size_t> dependencies;
    std::unordered_map<AutogradNode*, std::vector<std::shared_ptr<AutogradNode>>> graph_edges;

    compute_dependencies(root.get(), dependencies, graph_edges);

    root->add_grad_output(root_grad);

    std::queue<std::shared_ptr<AutogradNode>> ready_queue;
    ready_queue.push(root);

    while (!ready_queue.empty()) {
        auto curr = ready_queue.front();
        ready_queue.pop();

        TensorPtr go = curr->grad_output_;
        curr->grad_output_ = nullptr; // Immediately release intermediate gradient

        curr->backward(go);
        
        // Phase 3.2: Eager activation release
        // Release this node's activations immediately after backward completes
        // This is critical for 2048×2048 workloads to reduce VRAM usage
        if (eager_activation_release_) {
            release_node_activations(curr);
        } else {
            curr->release_variables();    // Traditional: release at end of backward
        }

        auto it = graph_edges.find(curr.get());
        if (it != graph_edges.end()) {
            for (const auto& next_node : it->second) {
                auto dep_it = dependencies.find(next_node.get());
                if (dep_it != dependencies.end()) {
                    dep_it->second -= 1;
                    if (dep_it->second == 0) {
                        ready_queue.push(next_node);
                    }
                }
            }
        }
    }
    
    // Clear released nodes set for next backward pass
    if (eager_activation_release_) {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        released_nodes_.clear();
    }
    
    current_depth--;
}

void Engine::execute_node_task(std::shared_ptr<AutogradNode> node, std::shared_ptr<GraphTask> graph_task) {
    // This would be used by worker threads in full thread pool implementation
    // For now, it's a placeholder for future enhancement
    try {
        TensorPtr grad_output = node->grad_output_;
        node->grad_output_ = nullptr;
        node->backward(grad_output);
        node->release_variables();
    } catch (const std::exception& e) {
        graph_task->mark_error();
        std::cerr << "Error executing node task: " << e.what() << std::endl;
    }
}

} // namespace soar::autograd

namespace soar {

void run_backward(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad) {
    autograd::Engine::get_default_engine().execute(std::move(root), root_grad);
}

} // namespace soar
