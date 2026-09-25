#pragma once

#include <soar/nn/module.hpp>
#include <soar/vulkan/context.hpp>
#include <cmath>

namespace soar::nn {

inline int select_group_count(int channels, int max_groups = 8) {
    for (int g = std::min(max_groups, channels); g > 0; --g) {
        if (channels % g == 0) return g;
    }
    return 1;
}

inline int autopad(int k, int p = -1, int d = 1) {
    if (d > 1) k = d * (k - 1) + 1;
    return (p < 0) ? (k / 2) : p;
}

/**
 * @brief 2D Convolution Layer supporting pointwise (1x1), depthwise, and general convolutions.
 */
class Conv2d : public Module {
public:
    Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
           size_t stride = 1, int padding = -1, size_t dilation = 1,
           size_t groups = 1, bool bias = false);

    TensorPtr forward(const TensorPtr& input) override;

    [[nodiscard]] size_t in_channels() const noexcept { return in_channels_; }
    [[nodiscard]] size_t out_channels() const noexcept { return out_channels_; }
    [[nodiscard]] size_t kernel_size() const noexcept { return kernel_size_; }
    [[nodiscard]] size_t stride() const noexcept { return stride_; }
    [[nodiscard]] size_t padding() const noexcept { return padding_; }
    [[nodiscard]] size_t dilation() const noexcept { return dilation_; }
    [[nodiscard]] size_t groups() const noexcept { return groups_; }
    [[nodiscard]] bool has_bias() const noexcept { return has_bias_; }

    [[nodiscard]] TensorPtr weight() const { return weight_; }
    [[nodiscard]] TensorPtr bias() const { return bias_; }

private:
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
    size_t dilation_;
    size_t groups_;
    bool has_bias_;

    TensorPtr weight_{nullptr};
    TensorPtr bias_{nullptr};
};

/**
 * @brief Group Normalization layer with PyTorch-exact biased sample variance.
 */
class GroupNorm : public Module {
public:
    GroupNorm(size_t num_groups, size_t num_channels, float eps = 1e-5f, bool affine = true);

    TensorPtr forward(const TensorPtr& input) override;

    [[nodiscard]] size_t num_groups() const noexcept { return num_groups_; }
    [[nodiscard]] size_t num_channels() const noexcept { return num_channels_; }
    [[nodiscard]] float eps() const noexcept { return eps_; }

    [[nodiscard]] TensorPtr weight() const { return weight_; }
    [[nodiscard]] TensorPtr bias() const { return bias_; }

private:
    size_t num_groups_;
    size_t num_channels_;
    float eps_;
    bool affine_;

    TensorPtr weight_{nullptr};
    TensorPtr bias_{nullptr};
};

/**
 * @brief SiLU activation layer: x / (1 + exp(-x)).
 */
class SiLU : public Module {
public:
    SiLU() : Module("SiLU") {}
    TensorPtr forward(const TensorPtr& input) override;
};

/**
 * @brief Sigmoid activation layer: 1 / (1 + exp(-x)).
 */
class Sigmoid : public Module {
public:
    Sigmoid() : Module("Sigmoid") {}
    TensorPtr forward(const TensorPtr& input) override;
};

/**
 * @brief PixelShuffle layer for sub-pixel resolution upsampling.
 */
class PixelShuffle : public Module {
public:
    explicit PixelShuffle(size_t upscale_factor = 2)
        : Module("PixelShuffle"), upscale_factor_(upscale_factor) {}

    TensorPtr forward(const TensorPtr& input) override;

    [[nodiscard]] size_t upscale_factor() const noexcept { return upscale_factor_; }

private:
    size_t upscale_factor_;
};

/**
 * @brief Bilinear upsampling with align_corners=false.
 */
class Upsample : public Module {
public:
    explicit Upsample(float scale_factor = 2.0f)
        : Module("Upsample"), scale_factor_(scale_factor) {}

    TensorPtr forward(const TensorPtr& input) override;

private:
    float scale_factor_;
};

/**
 * @brief 2D Max Pooling layer.
 */
class MaxPool2d : public Module {
public:
    explicit MaxPool2d(size_t kernel_size = 2, size_t stride = 1, size_t padding = 0)
        : Module("MaxPool2d"), kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

    TensorPtr forward(const TensorPtr& input) override;

    [[nodiscard]] size_t kernel_size() const noexcept { return kernel_size_; }
    [[nodiscard]] size_t stride() const noexcept { return stride_; }
    [[nodiscard]] size_t padding() const noexcept { return padding_; }

private:
    size_t kernel_size_;
    size_t stride_;
    size_t padding_;
};

/**
 * @brief Global average pooling layer over H x W -> 1 x 1.
 */
class AdaptiveAvgPool2d : public Module {
public:
    AdaptiveAvgPool2d() : Module("AdaptiveAvgPool2d") {}
    TensorPtr forward(const TensorPtr& input) override;
};

/**
 * @brief Sequential module container.
 */
class Sequential : public Module {
public:
    Sequential() : Module("Sequential") {}

    template <typename... Modules>
    explicit Sequential(Modules&&... mods) : Module("Sequential") {
        (add(std::forward<Modules>(mods)), ...);
    }

    void add(std::shared_ptr<Module> module) {
        std::string key = std::to_string(layers_.size());
        register_submodule(key, module);
        layers_.push_back(std::move(module));
    }

    TensorPtr forward(const TensorPtr& input) override {
        TensorPtr cur = input;
        for (auto& layer : layers_) {
            cur = layer->forward(cur);
        }
        return cur;
    }

    [[nodiscard]] size_t size() const noexcept { return layers_.size(); }
    [[nodiscard]] std::shared_ptr<Module> operator[](size_t i) const { return layers_[i]; }

private:
    std::vector<std::shared_ptr<Module>> layers_;
};

} // namespace soar::nn
