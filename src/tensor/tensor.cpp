#include <soar/tensor/tensor.hpp>
#include <soar/autograd/node.hpp>
#include <soar/core/logging.hpp>

#include <random>
#include <cstring>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace soar {

Tensor::Tensor() : shape_(), strides_(), host_data_() {}

Tensor::Tensor(const core::Shape& shape, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad) {
    compute_strides();
    host_data_.resize(shape_.numel(), 0.0f);
}

Tensor::Tensor(const core::Shape& shape, std::span<const float> initial_data, bool requires_grad)
    : shape_(shape), requires_grad_(requires_grad) {
    compute_strides();
    if (initial_data.size() != static_cast<size_t>(shape_.numel())) {
        throw ShapeError("Initial data size does not match tensor numel");
    }
    host_data_.assign(initial_data.begin(), initial_data.end());
}

Tensor::~Tensor() = default;

void Tensor::compute_strides() {
    size_t nd = shape_.ndim();
    strides_.resize(nd);
    if (nd == 0) return;

    size_t cur = 1;
    for (int i = static_cast<int>(nd) - 1; i >= 0; --i) {
        strides_[i] = cur;
        cur *= shape_[i];
    }
}

TensorPtr Tensor::create(const core::Shape& shape, bool requires_grad) {
    return std::make_shared<Tensor>(shape, requires_grad);
}

TensorPtr Tensor::zeros(const core::Shape& shape, bool requires_grad) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    t->zero_();
    return t;
}

TensorPtr Tensor::ones(const core::Shape& shape, bool requires_grad) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    t->fill_(1.0f);
    return t;
}

TensorPtr Tensor::randn(const core::Shape& shape, float mean, float std, bool requires_grad) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d(mean, std);
    for (auto& val : t->host_data_) {
        val = d(gen);
    }
    return t;
}

TensorPtr Tensor::from_blob(const core::Shape& shape, const float* data, bool copy) {
    if (!copy) {
        throw DeviceError("Zero-copy external blob not supported directly; must copy to maintain lifetime");
    }
    return std::make_shared<Tensor>(shape, std::span<const float>(data, shape.numel()));
}

void Tensor::zero_() {
    std::fill(host_data_.begin(), host_data_.end(), 0.0f);
    if (on_device_ && device_buffer_) {
        sync_to_device();
    }
}

void Tensor::fill_(float val) {
    std::fill(host_data_.begin(), host_data_.end(), val);
    if (on_device_ && device_buffer_) {
        sync_to_device();
    }
}

void Tensor::to_device(vk::VulkanContext& ctx, vk::BufferUsageType usage) {
    if (!on_device_ || !device_buffer_) {
        device_buffer_ = std::make_shared<vk::VulkanBuffer>(ctx, bytes(), usage);
        on_device_ = true;
    }
    sync_to_device();
}

void Tensor::to_host() {
    if (on_device_ && device_buffer_) {
        sync_to_host();
    }
}

void Tensor::sync_to_device() {
    if (on_device_ && device_buffer_) {
        device_buffer_->upload_host(host_data_.data(), bytes());
    }
}

void Tensor::sync_to_host() {
    if (on_device_ && device_buffer_) {
        device_buffer_->download_host(host_data_.data(), bytes());
    }
}

void Tensor::zero_grad() {
    if (grad_) {
        grad_->zero_();
    }
}

void Tensor::add_grad(const TensorPtr& incoming) {
    if (!incoming) return;
    if (!grad_) {
        grad_ = incoming->clone();
        return;
    }
    float* t_data = grad_->data();
    const float* in_data = incoming->data();
    size_t n = grad_->numel();
    for (size_t i = 0; i < n; ++i) {
        t_data[i] += in_data[i];
    }
}

void AutogradNode::add_grad_output(const TensorPtr& incoming) {
    if (!incoming) return;
    if (!grad_output_) {
        grad_output_ = incoming;
    } else {
        grad_output_->add_grad(incoming);
    }
}

void propagate_grad(const TensorPtr& tensor, const TensorPtr& incoming) {
    if (!tensor || !incoming) return;
    if (tensor->requires_grad()) {
        if (auto fn = tensor->grad_fn()) {
            fn->add_grad_output(incoming);
        } else {
            tensor->add_grad(incoming);
        }
    }
}

void run_backward(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad) {
    if (!root) return;

    // 1. Build reachability graph and compute in-degree in backward DAG
    std::unordered_map<AutogradNode*, int> in_degree;
    std::unordered_map<AutogradNode*, std::vector<std::shared_ptr<AutogradNode>>> next_nodes;
    std::unordered_set<AutogradNode*> visited;
    std::vector<std::shared_ptr<AutogradNode>> stack;
    stack.push_back(root);
    visited.insert(root.get());

    while (!stack.empty()) {
        auto curr = stack.back();
        stack.pop_back();

        for (const auto& parent : curr->get_inputs()) {
            if (!parent) continue;
            next_nodes[curr.get()].push_back(parent);
            in_degree[parent.get()]++;
            if (visited.insert(parent.get()).second) {
                stack.push_back(parent);
            }
        }
    }

    // 2. Initialize root's incoming gradient
    root->add_grad_output(root_grad);

    std::queue<std::shared_ptr<AutogradNode>> ready_queue;
    ready_queue.push(root);

    // 3. Process nodes in topological order using Kahn's algorithm
    while (!ready_queue.empty()) {
        auto node = ready_queue.front();
        ready_queue.pop();

        TensorPtr go = node->grad_output_;
        node->grad_output_ = nullptr; // release intermediate gradient memory

        // Execute backward for this node
        node->backward(go);
        node->release_variables(); // release saved activations immediately

        // Decrement in-degree for dependent nodes using cached graph edges
        auto it_edges = next_nodes.find(node.get());
        if (it_edges != next_nodes.end()) {
            for (const auto& parent : it_edges->second) {
                auto it = in_degree.find(parent.get());
                if (it != in_degree.end()) {
                    it->second--;
                    if (it->second == 0) {
                        ready_queue.push(parent);
                    }
                }
            }
        }
    }
}

void Tensor::backward(TensorPtr gradient) {
    if (!requires_grad_) {
        return;
    }

    if (!gradient) {
        if (numel() != 1) {
            throw ShapeError("grad can be implicitly created only for scalar outputs");
        }
        gradient = Tensor::ones(shape_);
    }

    add_grad(gradient);

    if (grad_fn_) {
        auto root_fn = std::move(grad_fn_);
        grad_fn_ = nullptr;
        run_backward(root_fn, gradient);
    }
}

TensorPtr Tensor::reshape(const core::Shape& new_shape) {
    if (new_shape.numel() != static_cast<int64_t>(numel())) {
        throw ShapeError("Cannot reshape tensor: numel mismatch");
    }
    auto t = std::make_shared<Tensor>(new_shape, requires_grad_);
    t->host_data_ = host_data_;
    t->on_device_ = on_device_;
    t->device_buffer_ = device_buffer_;
    return t;
}

TensorPtr Tensor::clone() const {
    auto t = std::make_shared<Tensor>(shape_, requires_grad_);
    t->host_data_ = host_data_;
    return t;
}

} // namespace soar
