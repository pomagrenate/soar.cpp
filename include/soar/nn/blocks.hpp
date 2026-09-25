#pragma once

#include <soar/nn/layers.hpp>
#include <vector>
#include <memory>

namespace soar::nn {

// Mathematical tensor helper functions with autograd support
TensorPtr add_tensors(const TensorPtr& a, const TensorPtr& b);
TensorPtr mul_tensors(const TensorPtr& a, const TensorPtr& b);
TensorPtr convex_combination(const TensorPtr& g, const TensorPtr& a, const TensorPtr& b);
TensorPtr concat_channels(const std::vector<TensorPtr>& tensors);
TensorPtr resize_bilinear(const TensorPtr& input, size_t target_h, size_t target_w);

/**
 * @brief Conv2d -> GroupNorm -> SiLU block.
 */
class CBA : public Module {
public:
    CBA(size_t c1, size_t c2, size_t k = 1, size_t s = 1, int p = -1,
        size_t d = 1, size_t g = 1, bool act = true);

    TensorPtr forward(const TensorPtr& input) override;

    [[nodiscard]] std::shared_ptr<Conv2d> conv() const { return conv_; }
    [[nodiscard]] std::shared_ptr<GroupNorm> norm() const { return norm_; }

private:
    std::shared_ptr<Conv2d> conv_;
    std::shared_ptr<GroupNorm> norm_;
    std::shared_ptr<SiLU> act_{nullptr};
};

/**
 * @brief Depthwise stride-2 downsampling with pointwise projection.
 */
class Down : public Module {
public:
    Down(size_t c1, size_t c2);
    TensorPtr forward(const TensorPtr& input) override;

private:
    std::shared_ptr<CBA> dw_;
    std::shared_ptr<CBA> pw_;
};

/**
 * @brief Large-Kernel Residual block with identity residual connection.
 */
class LKR : public Module {
public:
    LKR(size_t c, size_t k = 7, float e = 2.0f);
    TensorPtr forward(const TensorPtr& input) override;

private:
    std::shared_ptr<CBA> dw_;
    std::shared_ptr<CBA> pw1_;
    std::shared_ptr<CBA> pw2_;
};

/**
 * @brief Multi-scale context module with dilated depthwise branches and global avg pool gate.
 */
class Ctx : public Module {
public:
    Ctx(size_t c1, size_t c2, const std::vector<size_t>& dilations = {1, 3, 5});
    TensorPtr forward(const TensorPtr& input) override;

private:
    size_t c1_;
    size_t c2_;
    bool add_;
    std::shared_ptr<CBA> cv1_;
    std::vector<std::shared_ptr<CBA>> branches_;
    std::shared_ptr<CBA> cv2_;
    std::shared_ptr<AdaptiveAvgPool2d> pool_;
    std::shared_ptr<Conv2d> gate_conv_;
    std::shared_ptr<Sigmoid> gate_sig_;
};

/**
 * @brief Gated top-down cross-attention convex fusion module.
 */
class Fuse : public Module {
public:
    Fuse(size_t c_low, size_t c_skip, size_t c2);
    TensorPtr forward(const std::vector<TensorPtr>& inputs) override;

private:
    std::shared_ptr<CBA> low_;
    std::shared_ptr<CBA> skip_;
    std::shared_ptr<CBA> gate_dw_;
    std::shared_ptr<Conv2d> gate_conv_;
    std::shared_ptr<Sigmoid> gate_sig_;
    std::shared_ptr<CBA> out_dw_;
    std::shared_ptr<CBA> out_pw_;
};

/**
 * @brief Multi-scale semantic aggregation module at stride 4.
 */
class Agg : public Module {
public:
    Agg(const std::vector<size_t>& chs, size_t c2);
    TensorPtr forward(const std::vector<TensorPtr>& inputs) override;

private:
    std::vector<std::shared_ptr<CBA>> proj_;
    std::shared_ptr<CBA> fuse_;
};

/**
 * @brief Sub-pixel segmentation head with PixelShuffle.
 */
class SegHead : public Module {
public:
    SegHead(size_t c1, size_t nc, size_t mid = 32, size_t r = 2, float prior = 0.01f);
    TensorPtr forward(const TensorPtr& input) override;

private:
    std::shared_ptr<CBA> refine_dw_;
    std::shared_ptr<CBA> refine_pw_;
    std::shared_ptr<Conv2d> pred_;
    std::shared_ptr<PixelShuffle> shuffle_;
};

/**
 * @brief Standard Bottleneck block matching YOLO/segres.
 */
class Bottleneck : public Module {
public:
    Bottleneck(size_t c1, size_t c2, bool shortcut = true, size_t g = 1, size_t k = 3, float e = 0.5f);
    TensorPtr forward(const TensorPtr& input) override;

private:
    bool add_;
    std::shared_ptr<CBA> cv1_;
    std::shared_ptr<CBA> cv2_;
};

/**
 * @brief CSP bottleneck block (C3k2) matching YOLO/segres.
 */
class C3k2 : public Module {
public:
    C3k2(size_t c1, size_t c2, size_t n = 1, bool shortcut = true, size_t g = 1, float e = 0.5f);
    TensorPtr forward(const TensorPtr& input) override;

private:
    std::shared_ptr<CBA> cv1_;
    std::shared_ptr<CBA> cv2_;
    std::shared_ptr<CBA> cv3_;
    std::vector<std::shared_ptr<Bottleneck>> m_;
};

/**
 * @brief Spatial Pyramid Pooling - Fast (SPPF).
 */
class SPPF : public Module {
public:
    SPPF(size_t c1, size_t c2, size_t k = 5);
    TensorPtr forward(const TensorPtr& input) override;

private:
    std::shared_ptr<CBA> cv1_;
    std::shared_ptr<CBA> cv2_;
    std::shared_ptr<MaxPool2d> m_;
};

/**
 * @brief Multi-tensor channel concatenation module.
 */
class Concat : public Module {
public:
    explicit Concat(size_t dimension = 1) : Module("Concat"), dim_(dimension) {}
    TensorPtr forward(const std::vector<TensorPtr>& inputs);
    TensorPtr forward(const TensorPtr& input) override { return input; }

private:
    size_t dim_{1};
};

} // namespace soar::nn

