#include <soar/autograd/engine.hpp>
#include <iostream>

namespace soar::autograd {

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

Engine& Engine::get_default_engine() {
    static Engine s_engine;
    return s_engine;
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

    std::lock_guard<std::mutex> lock(engine_mutex_);

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
        curr->release_variables();    // Immediately free saved forward activations

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
}

} // namespace soar::autograd

namespace soar {

void run_backward(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad) {
    autograd::Engine::get_default_engine().execute(std::move(root), root_grad);
}

} // namespace soar
