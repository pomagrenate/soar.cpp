#include <soar/tensor/tensor.hpp>
#include <soar/autograd/node.hpp>
#include <soar/core/logging.hpp>
#include <soar/cuda/cuda_allocator.hpp>
#include <soar/cuda/cuda_runtime.hpp>
#include <soar/cuda/cuda_kernels.hpp>

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

Tensor::~Tensor() {
    if (cuda_data_) {
        cuda::CudaMemoryPool::instance(cuda_device_id_).deallocate(cuda_data_);
        cuda_data_ = nullptr;
    }
}

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

TensorPtr Tensor::create(const core::Shape& shape, bool requires_grad, bool is_cuda, int device_id) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    if (is_cuda) {
        t->to_cuda(device_id, /*sync_from_host=*/false);
    }
    return t;
}

TensorPtr Tensor::zeros(const core::Shape& shape, bool requires_grad, bool is_cuda, int device_id) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    t->zero_();
    if (is_cuda) {
        t->to_cuda(device_id, /*sync_from_host=*/false);
        t->zero_();
    }
    return t;
}

TensorPtr Tensor::ones(const core::Shape& shape, bool requires_grad, bool is_cuda, int device_id) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    t->fill_(1.0f);
    if (is_cuda) {
        t->to_cuda(device_id, /*sync_from_host=*/false);
        t->fill_(1.0f);
    }
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
    if (cuda_data_) {
        cuda::kernels::cuda_zero(cuda_data_, numel());
    }
}

void Tensor::fill_(float val) {
    std::fill(host_data_.begin(), host_data_.end(), val);
    if (on_device_ && device_buffer_) {
        sync_to_device();
    }
    if (cuda_data_) {
        cuda::kernels::cuda_fill(cuda_data_, val, numel());
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
    sync_to_host();
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
    if (cuda_data_) {
        sync_cuda_to_host();
    }
}

void Tensor::to_cuda(int device_id, bool sync_from_host) {
    if (cuda_data_) return;
    size_t sz = bytes();
    if (sz == 0) return;
    cuda_device_id_ = device_id;
    cuda_data_ = static_cast<float*>(cuda::CudaMemoryPool::instance(device_id).allocate(sz));
    if (sync_from_host) {
        sync_to_cuda();
    }
}

void Tensor::sync_to_cuda(void* stream) {
    if (!cuda_data_) return;
    cuda::cudaMemcpyAsync(cuda_data_, host_data_.data(), bytes(), cuda::cudaMemcpyHostToDevice, static_cast<cuda::cudaStream_t>(stream));
}

void Tensor::sync_cuda_to_host(void* stream) {
    if (!cuda_data_) return;
    cuda::cudaMemcpyAsync(host_data_.data(), cuda_data_, bytes(), cuda::cudaMemcpyDeviceToHost, static_cast<cuda::cudaStream_t>(stream));
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
        if (incoming->is_cuda() && !grad_->is_cuda()) {
            grad_->to_cuda(cuda_device_id_);
        }
        return;
    }
    if (grad_->is_cuda() && incoming->is_cuda()) {
        cuda::kernels::add_backward(incoming->cuda_data(), grad_->cuda_data(), nullptr, grad_->numel());
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
        grad_output_ = incoming->clone();
        if (incoming->is_cuda() && !grad_output_->is_cuda()) {
            grad_output_->to_cuda();
        }
    } else {
        if (grad_output_->is_cuda() && incoming->is_cuda()) {
            cuda::kernels::add_backward(incoming->cuda_data(), grad_output_->cuda_data(), nullptr, grad_output_->numel());
            return;
        }
        float* d = grad_output_->data();
        const float* s = incoming->data();
        size_t n = std::min(grad_output_->numel(), incoming->numel());
        for (size_t i = 0; i < n; ++i) {
            d[i] += s[i];
        }
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

void run_backward(std::shared_ptr<AutogradNode> root, const TensorPtr& root_grad);


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
    if (cuda_data_) {
        t->to_cuda(cuda_device_id_, /*sync_from_host=*/false);
        cuda::cudaMemcpyAsync(t->cuda_data_, cuda_data_, bytes(), cuda::cudaMemcpyDeviceToDevice, nullptr);
    }
    return t;
}

} // namespace soar
