#pragma once

#include <soar/core/types.hpp>
#include <soar/core/shape.hpp>
#include <soar/core/error.hpp>
#include <soar/vulkan/context.hpp>
#include <soar/vulkan/buffer.hpp>

#include <soar/memory/palloc_allocator.hpp>
#include <soar/autograd/node.hpp>
#include <vector>
#include <memory>
#include <string>
#include <span>
#include <functional>

namespace soar {

class Tensor;
struct AutogradNode;

using TensorPtr = std::shared_ptr<Tensor>;
using ConstTensorPtr = std::shared_ptr<const Tensor>;

/**
 * @brief Unified C++20 Tensor class supporting host CPU buffers, Vulkan GPU buffers, and Autograd.
 */
class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    Tensor();
    explicit Tensor(const core::Shape& shape, bool requires_grad = false);
    Tensor(const core::Shape& shape, std::span<const float> initial_data, bool requires_grad = false);
    ~Tensor();

    // Factory methods
    static TensorPtr create(const core::Shape& shape, bool requires_grad = false, bool is_cuda = false, int device_id = 0);
    static TensorPtr zeros(const core::Shape& shape, bool requires_grad = false, bool is_cuda = false, int device_id = 0);
    static TensorPtr ones(const core::Shape& shape, bool requires_grad = false, bool is_cuda = false, int device_id = 0);
    static TensorPtr randn(const core::Shape& shape, float mean = 0.0f, float std = 1.0f, bool requires_grad = false);
    static TensorPtr from_blob(const core::Shape& shape, const float* data, bool copy = true);

    // Shape and Layout
    [[nodiscard]] const core::Shape& shape() const noexcept { return shape_; }
    [[nodiscard]] const memory::PallocVector<size_t>& strides() const noexcept { return strides_; }
    [[nodiscard]] size_t ndim() const noexcept { return shape_.ndim(); }
    [[nodiscard]] size_t dim(size_t index) const { return shape_[index]; }
    [[nodiscard]] size_t numel() const noexcept { return shape_.numel(); }
    [[nodiscard]] size_t bytes() const noexcept { return shape_.numel() * sizeof(float); }

    // Memory Access
    [[nodiscard]] float* data() noexcept { return host_data_.data(); }
    [[nodiscard]] const float* data() const noexcept { return host_data_.data(); }
    [[nodiscard]] std::span<float> span() noexcept { return host_data_; }
    [[nodiscard]] std::span<const float> span() const noexcept { return host_data_; }

    [[nodiscard]] float& item() {
        if (numel() != 1) throw ShapeError("item() called on tensor with numel != 1");
        return host_data_[0];
    }
    [[nodiscard]] float item() const {
        if (numel() != 1) throw ShapeError("item() called on tensor with numel != 1");
        return host_data_[0];
    }

    float& operator[](size_t index) { return host_data_[index]; }
    const float& operator[](size_t index) const { return host_data_[index]; }

    // Vulkan Device Management
    [[nodiscard]] bool is_on_device() const noexcept { return on_device_; }
    [[nodiscard]] vk::VulkanBuffer* device_buffer() noexcept { return device_buffer_.get(); }
    [[nodiscard]] const vk::VulkanBuffer* device_buffer() const noexcept { return device_buffer_.get(); }

    void to_device(vk::VulkanContext& ctx, vk::BufferUsageType usage = vk::BufferUsageType::DeviceStorage);
    void to_host();
    void sync_to_device();
    void sync_to_host();

    // CUDA Device Management
    [[nodiscard]] bool is_cuda() const noexcept { return cuda_data_ != nullptr; }
    [[nodiscard]] float* cuda_data() noexcept { return cuda_data_; }
    [[nodiscard]] const float* cuda_data() const noexcept { return cuda_data_; }

    void to_cuda(int device_id = 0, bool sync_from_host = true);
    void sync_to_cuda(void* stream = nullptr);
    void sync_cuda_to_host(void* stream = nullptr);

    // In-place operations
    void zero_();
    void fill_(float val);

    // Autograd
    [[nodiscard]] bool requires_grad() const noexcept { return GradMode::is_enabled() && requires_grad_; }
    void set_requires_grad(bool req) noexcept { requires_grad_ = req; }

    [[nodiscard]] TensorPtr grad() const noexcept { return grad_; }
    [[nodiscard]] TensorPtr& grad_ref() noexcept { return grad_; }
    void set_grad(TensorPtr g) noexcept { grad_ = std::move(g); }
    void zero_grad();
    void add_grad(const TensorPtr& incoming);

    [[nodiscard]] std::shared_ptr<AutogradNode> grad_fn() const noexcept { return grad_fn_; }
    void set_grad_fn(std::shared_ptr<AutogradNode> fn) noexcept {
        if (GradMode::is_enabled()) {
            grad_fn_ = std::move(fn);
        }
    }

    void backward(TensorPtr gradient = nullptr);

    // Reshape & View
    TensorPtr reshape(const core::Shape& new_shape);
    TensorPtr clone() const;

private:
    void compute_strides();

    core::Shape shape_{};
    memory::PallocVector<size_t> strides_{};
    memory::PallocVector<float> host_data_{};

    bool on_device_{false};
    std::shared_ptr<vk::VulkanBuffer> device_buffer_{nullptr};

    float* cuda_data_{nullptr};
    int cuda_device_id_{0};

    bool requires_grad_{false};
    TensorPtr grad_{nullptr};
    std::shared_ptr<AutogradNode> grad_fn_{nullptr};
};

} // namespace soar
