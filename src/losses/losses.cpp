#include <soar/losses/losses.hpp>
#include <soar/cuda/cuda_kernels.hpp>

#include <soar/autograd/node.hpp>
#include <soar/core/logging.hpp>

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstring>

namespace soar::losses {

namespace {

inline float sigmoid_f(float z) noexcept {
    return 1.0f / (1.0f + std::exp(-z));
}

} // anonymous namespace

// -------------------------------------------------------------
// BCE With Logits Loss
// -------------------------------------------------------------
struct BCENode : public AutogradNode {
    TensorPtr logits;
    TensorPtr targets;
    float weight;
    float pos_weight;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (logits && logits->grad_fn()) return {logits->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!logits || !logits->requires_grad()) return;
        if (logits->is_cuda()) {
            if (targets && !targets->is_cuda()) targets->to_cuda();
            TensorPtr grad_z = Tensor::zeros(logits->shape(), false, true);
            float go = grad_output->item();
            soar::cuda::kernels::bce_with_logits_backward(
                logits->cuda_data(), targets->cuda_data(), grad_z->cuda_data(),
                go, weight, pos_weight, logits->numel());
            propagate_grad(logits, grad_z);
            return;
        }
        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* y = targets->data();
        float* gz = grad_z->data();
        float go = grad_output->item();

        size_t n = logits->numel();
        float scale = (go * weight) / static_cast<float>(n);

        for (size_t i = 0; i < n; ++i) {
            float sig = sigmoid_f(z[i]);
            // PyTorch aten: dL/dz = (sig * (1 + (pos_weight - 1) * y) - pos_weight * y) * scale
            float w = 1.0f + (pos_weight - 1.0f) * y[i];
            gz[i] = (sig * w - pos_weight * y[i]) * scale;
        }

        if (logits->is_cuda()) {
            grad_z->to_cuda();
        }
        propagate_grad(logits, grad_z);
    }

    void release_variables() override {
        logits = nullptr;
        targets = nullptr;
    }
};

BCEWithLogitsLoss::BCEWithLogitsLoss(float weight, float pos_weight)
    : BaseLoss(weight), pos_weight_(pos_weight) {}

TensorPtr BCEWithLogitsLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    if (logits->shape() != targets->shape()) {
        if (logits->numel() != targets->numel()) {
            throw ShapeError("BCEWithLogitsLoss: shape mismatch between logits and targets");
        }
    }
    TensorPtr loss = Tensor::create({1}, logits->requires_grad());

    if (logits->is_cuda()) {
        if (targets && !targets->is_cuda()) targets->to_cuda();
        float loss_val = soar::cuda::kernels::bce_with_logits_forward(
            logits->cuda_data(), targets->cuda_data(), weight_, pos_weight_, logits->numel());
        loss->item() = loss_val;

        if (logits->requires_grad()) {
            auto node = std::make_shared<BCENode>();
            node->logits = logits;
            node->targets = targets;
            node->weight = weight_;
            node->pos_weight = pos_weight_;
            loss->set_grad_fn(node);
        }
        return loss;
    }

    if (targets->is_cuda()) targets->sync_to_host();
    const float* z = logits->data();
    const float* y = targets->data();
    size_t n = logits->numel();

    double total_loss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float zi = z[i];
        float yi = y[i];
        float w = 1.0f + (pos_weight_ - 1.0f) * yi;
        float term = std::max(-zi, 0.0f) + std::log(1.0f + std::exp(-std::abs(zi)));
        float loss_i = (1.0f - yi) * zi + w * term;
        total_loss += loss_i;
    }

    loss->item() = static_cast<float>(weight_ * (total_loss / static_cast<double>(n)));

    if (logits->requires_grad()) {
        auto node = std::make_shared<BCENode>();
        node->logits = logits;
        node->targets = targets;
        node->weight = weight_;
        node->pos_weight = pos_weight_;
        loss->set_grad_fn(node);
    }

    return loss;
}

// -------------------------------------------------------------
// Soft Dice Loss
// -------------------------------------------------------------
struct DiceNode : public AutogradNode {
    TensorPtr logits;
    TensorPtr targets;
    float weight{1.0f};
    float smooth{1.0f};
    float inter{0.0f};
    float sum_p{0.0f};
    float sum_y{0.0f};

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (logits && logits->grad_fn()) return {logits->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!logits || !logits->requires_grad()) return;
        if (logits->is_cuda()) {
            if (targets && !targets->is_cuda()) targets->to_cuda();
            TensorPtr grad_z = Tensor::zeros(logits->shape(), false, true);
            float go = grad_output->item();
            soar::cuda::kernels::dice_loss_backward(
                logits->cuda_data(), targets->cuda_data(), grad_z->cuda_data(),
                go, weight, smooth, inter, sum_p, sum_y, logits->numel());
            propagate_grad(logits, grad_z);
            return;
        }

        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* y = targets->data();
        float* gz = grad_z->data();
        float go = grad_output->item();
        size_t n = logits->numel();

        double inter_val = 0.0;
        double sum_p_val = 0.0;
        double sum_y_val = 0.0;

        for (size_t i = 0; i < n; ++i) {
            float pi = sigmoid_f(z[i]);
            inter_val += pi * y[i];
            sum_p_val += pi;
            sum_y_val += y[i];
        }

        double denom = sum_p_val + sum_y_val + smooth;
        double denom_sq = denom * denom;
        double numer = 2.0 * inter_val + smooth;
        float factor = go * weight;

        for (size_t i = 0; i < n; ++i) {
            float pi = sigmoid_f(z[i]);
            float dp_dz = pi * (1.0f - pi);
            double d_dice_dp = (2.0 * y[i] * denom - numer) / denom_sq;
            double d_loss_dz = -d_dice_dp * dp_dz;
            gz[i] = static_cast<float>(factor * d_loss_dz);
        }

        propagate_grad(logits, grad_z);
    }

    void release_variables() override {
        logits = nullptr;
        targets = nullptr;
    }
};

DiceLoss::DiceLoss(float weight, float smooth)
    : BaseLoss(weight), smooth_(smooth) {}

TensorPtr DiceLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    if (logits->shape() != targets->shape()) {
        if (logits->numel() != targets->numel()) {
            throw ShapeError("DiceLoss: shape mismatch between logits and targets");
        }
    }
    TensorPtr loss = Tensor::create({1}, logits->requires_grad());

    if (logits->is_cuda()) {
        if (targets && !targets->is_cuda()) targets->to_cuda();
        float out_inter = 0.0f, out_sum_p = 0.0f, out_sum_y = 0.0f;
        float loss_val = soar::cuda::kernels::dice_loss_forward(
            logits->cuda_data(), targets->cuda_data(), weight_, smooth_,
            out_inter, out_sum_p, out_sum_y, logits->numel());
        loss->item() = loss_val;

        if (logits->requires_grad()) {
            auto node = std::make_shared<DiceNode>();
            node->logits = logits;
            node->targets = targets;
            node->weight = weight_;
            node->smooth = smooth_;
            node->inter = out_inter;
            node->sum_p = out_sum_p;
            node->sum_y = out_sum_y;
            loss->set_grad_fn(node);
        }
        return loss;
    }

    if (targets->is_cuda()) targets->sync_to_host();
    const float* z = logits->data();
    const float* y = targets->data();
    size_t n = logits->numel();

    double inter = 0.0;
    double sum_p = 0.0;
    double sum_y = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float p = sigmoid_f(z[i]);
        inter += p * y[i];
        sum_p += p;
        sum_y += y[i];
    }

    double dice = (2.0 * inter + smooth_) / (sum_p + sum_y + smooth_ + 1e-7);
    loss->item() = static_cast<float>(weight_ * (1.0 - dice));

    if (logits->requires_grad()) {
        auto node = std::make_shared<DiceNode>();
        node->logits = logits;
        node->targets = targets;
        node->weight = weight_;
        node->smooth = smooth_;
        node->inter = static_cast<float>(inter);
        node->sum_p = static_cast<float>(sum_p);
        node->sum_y = static_cast<float>(sum_y);
        loss->set_grad_fn(node);
    }

    return loss;
}

// -------------------------------------------------------------
// Focal Loss
// -------------------------------------------------------------
struct FocalNode : public AutogradNode {
    TensorPtr logits;
    TensorPtr targets;
    float weight;
    float gamma;
    float alpha;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (logits && logits->grad_fn()) return {logits->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!logits || !logits->requires_grad()) return;
        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* y = targets->data();
        float* gz = grad_z->data();
        float go = grad_output->item();
        size_t n = logits->numel();
        float scale = (go * weight) / static_cast<float>(n);

        for (size_t i = 0; i < n; ++i) {
            float p = sigmoid_f(z[i]);
            float target = y[i];
            float pt = target * p + (1.0f - target) * (1.0f - p);
            float alpha_t = target * alpha + (1.0f - target) * (1.0f - alpha);
            float focal_weight = std::pow(std::max(0.0f, 1.0f - pt), gamma);

            // d(BCE)/dz = p - y
            float grad_bce = p - target;
            gz[i] = alpha_t * focal_weight * grad_bce * scale;
        }

        propagate_grad(logits, grad_z);
    }
};

FocalLoss::FocalLoss(float weight, float gamma, float alpha)
    : BaseLoss(weight), gamma_(gamma), alpha_(alpha) {}

TensorPtr FocalLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    if (logits->shape() != targets->shape()) {
        throw ShapeError("FocalLoss: shape mismatch between logits and targets");
    }

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    const float* z = logits->data();
    const float* y = targets->data();
    size_t n = logits->numel();

    double total = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float zi = z[i];
        float yi = y[i];
        float p = sigmoid_f(zi);
        float pt = yi * p + (1.0f - yi) * (1.0f - p);
        float focal_w = std::pow(std::max(0.0f, 1.0f - pt), gamma_);
        float alpha_t = yi * alpha_ + (1.0f - yi) * (1.0f - alpha_);

        float bce = std::max(-zi, 0.0f) + std::log(1.0f + std::exp(-std::abs(zi))) + (1.0f - yi) * zi;
        total += alpha_t * focal_w * bce;
    }

    loss->item() = static_cast<float>(weight_ * (total / static_cast<double>(n)));

    if (logits->requires_grad()) {
        auto node = std::make_shared<FocalNode>();
        node->logits = logits;
        node->targets = targets;
        node->weight = weight_;
        node->gamma = gamma_;
        node->alpha = alpha_;
        loss->set_grad_fn(node);
    }

    return loss;
}

// -------------------------------------------------------------
// Tversky Loss & Focal Tversky Loss
// -------------------------------------------------------------
struct TverskyNode : public AutogradNode {
    TensorPtr logits;
    TensorPtr targets;
    float weight;
    float alpha;
    float beta;
    float smooth;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (logits && logits->grad_fn()) return {logits->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!logits || !logits->requires_grad()) return;
        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* y = targets->data();
        float* gz = grad_z->data();
        float go = grad_output->item();
        size_t n = logits->numel();

        double tp = 0.0, fp = 0.0, fn = 0.0;
        for (size_t i = 0; i < n; ++i) {
            float pi = sigmoid_f(z[i]);
            tp += pi * y[i];
            fp += pi * (1.0f - y[i]);
            fn += (1.0f - pi) * y[i];
        }

        double num = tp + smooth;
        double den = tp + alpha * fp + beta * fn + smooth;
        double den_sq = den * den;
        float factor = go * weight;

        for (size_t i = 0; i < n; ++i) {
            float pi = sigmoid_f(z[i]);
            float dp_dz = pi * (1.0f - pi);
            double d_tp = y[i];
            double d_den = y[i] + alpha * (1.0f - y[i]) - beta * y[i];
            double d_tversky_dp = (d_tp * den - num * d_den) / den_sq;
            double d_loss_dz = -d_tversky_dp * dp_dz;
            gz[i] = static_cast<float>(factor * d_loss_dz);
        }

        propagate_grad(logits, grad_z);
    }
};

TverskyLoss::TverskyLoss(float weight, float alpha, float beta, float smooth)
    : BaseLoss(weight), alpha_(alpha), beta_(beta), smooth_(smooth) {}

TensorPtr TverskyLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    if (logits->shape() != targets->shape()) {
        throw ShapeError("TverskyLoss: shape mismatch");
    }

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    const float* z = logits->data();
    const float* y = targets->data();
    size_t n = logits->numel();

    double tp = 0.0, fp = 0.0, fn = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float p = sigmoid_f(z[i]);
        tp += p * y[i];
        fp += p * (1.0f - y[i]);
        fn += (1.0f - p) * y[i];
    }

    double tversky = (tp + smooth_) / (tp + alpha_ * fp + beta_ * fn + smooth_ + 1e-7);
    loss->item() = static_cast<float>(weight_ * (1.0 - tversky));

    if (logits->requires_grad()) {
        auto node = std::make_shared<TverskyNode>();
        node->logits = logits;
        node->targets = targets;
        node->weight = weight_;
        node->alpha = alpha_;
        node->beta = beta_;
        node->smooth = smooth_;
        loss->set_grad_fn(node);
    }

    return loss;
}

FocalTverskyLoss::FocalTverskyLoss(float weight, float alpha, float beta, float gamma, float smooth)
    : BaseLoss(weight), alpha_(alpha), beta_(beta), gamma_(gamma), smooth_(smooth) {}

TensorPtr FocalTverskyLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    TverskyLoss tv(1.0f, alpha_, beta_, smooth_);
    TensorPtr base_loss = tv.forward(logits, targets);
    float tversky_val = 1.0f - base_loss->item();
    float ft = std::pow(std::max(0.0f, 1.0f - tversky_val), gamma_);

    TensorPtr out = Tensor::create({1}, logits->requires_grad());
    out->item() = weight_ * ft;
    return out;
}

// -------------------------------------------------------------
// DiceBCELoss
// -------------------------------------------------------------
DiceBCELoss::DiceBCELoss(float weight, float dice_weight, float bce_weight, float bce_pos_weight, float smooth)
    : BaseLoss(weight), dice_weight_(dice_weight), bce_weight_(bce_weight),
      bce_(1.0f, bce_pos_weight), dice_(1.0f, smooth) {}

TensorPtr DiceBCELoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    TensorPtr bce_loss = bce_.forward(logits, targets);
    TensorPtr dice_loss = dice_.forward(logits, targets);

    last_bce_ = bce_loss->item();
    last_dice_ = dice_loss->item();

    if (auto dn = std::dynamic_pointer_cast<DiceNode>(dice_loss->grad_fn())) {
        last_inter_ = dn->inter;
        last_sum_p_ = dn->sum_p;
        last_sum_y_ = dn->sum_y;
    }

    TensorPtr total = Tensor::create({1}, logits->requires_grad());
    total->item() = weight_ * (bce_weight_ * last_bce_ + dice_weight_ * last_dice_);

    if (logits->requires_grad()) {
        struct DiceBCENode : public AutogradNode {
            TensorPtr logits;
            TensorPtr targets;
            float w;
            float bw;
            float dw;
            float pos_weight;
            float smooth;
            float inter{0.0f};
            float sum_p{0.0f};
            float sum_y{0.0f};

            std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
                if (logits && logits->grad_fn()) return {logits->grad_fn()};
                return {};
            }

            void backward(const TensorPtr& grad_output) override {
                if (!logits || !logits->requires_grad()) return;
                float go = grad_output->item();

                if (logits->is_cuda()) {
                    if (targets && !targets->is_cuda()) targets->to_cuda();
                    TensorPtr grad_z = Tensor::zeros(logits->shape(), false, true);
                    soar::cuda::kernels::dice_bce_loss_backward(
                        logits->cuda_data(), targets->cuda_data(), grad_z->cuda_data(),
                        go, w, bw, dw, pos_weight, smooth, inter, sum_p, sum_y, logits->numel());
                    propagate_grad(logits, grad_z);
                    return;
                }

                TensorPtr grad_z = Tensor::zeros(logits->shape());
                float* gz = grad_z->data();
                const float* z = logits->data();
                const float* y = targets->data();
                size_t n = logits->numel();

                // 1. BCE gradient
                if (bw > 0.0f) {
                    float scale_bce = (go * w * bw) / static_cast<float>(n);
                    for (size_t i = 0; i < n; ++i) {
                        float sig = sigmoid_f(z[i]);
                        float w_pos = 1.0f + (pos_weight - 1.0f) * y[i];
                        gz[i] += (sig * w_pos - pos_weight * y[i]) * scale_bce;
                    }
                }

                // 2. Dice gradient
                if (dw > 0.0f) {
                    double inter_val = 0.0, sum_p_val = 0.0, sum_y_val = 0.0;
                    for (size_t i = 0; i < n; ++i) {
                        float pi = sigmoid_f(z[i]);
                        inter_val += pi * y[i];
                        sum_p_val += pi;
                        sum_y_val += y[i];
                    }
                    double denom = sum_p_val + sum_y_val + smooth;
                    double denom_sq = denom * denom;
                    double numer = 2.0 * inter_val + smooth;
                    float scale_dice = go * w * dw;
                    for (size_t i = 0; i < n; ++i) {
                        float pi = sigmoid_f(z[i]);
                        float dp_dz = pi * (1.0f - pi);
                        double d_dice_dp = (2.0 * y[i] * denom - numer) / denom_sq;
                        double d_loss_dz = -d_dice_dp * dp_dz;
                        gz[i] += static_cast<float>(scale_dice * d_loss_dz);
                    }
                }

                propagate_grad(logits, grad_z);
            }

            void release_variables() override {
                logits = nullptr;
                targets = nullptr;
            }
        };

        auto node = std::make_shared<DiceBCENode>();
        node->logits = logits;
        node->targets = targets;
        node->w = weight_;
        node->bw = bce_weight_;
        node->dw = dice_weight_;
        node->pos_weight = bce_.pos_weight();
        node->smooth = dice_.smooth();
        node->inter = last_inter_;
        node->sum_p = last_sum_p_;
        node->sum_y = last_sum_y_;
        total->set_grad_fn(node);
    }

    return total;
}

// -------------------------------------------------------------
// Sobel Edge Detection & Boundary Losses
// -------------------------------------------------------------
TensorPtr sobel_edges(const TensorPtr& x) {
    size_t C = x->dim(0);
    size_t H = x->dim(1);
    size_t W = x->dim(2);

    TensorPtr edges = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)});
    const float* in = x->data();
    float* out = edges->data();

    static const float kx[3][3] = {
        {-1.0f, 0.0f, 1.0f},
        {-2.0f, 0.0f, 2.0f},
        {-1.0f, 0.0f, 1.0f}
    };
    static const float ky[3][3] = {
        {-1.0f, -2.0f, -1.0f},
        { 0.0f,  0.0f,  0.0f},
        { 1.0f,  2.0f,  1.0f}
    };

    for (size_t c = 0; c < C; ++c) {
        const float* plane = in + c * (H * W);
        float* edge_plane = out + c * (H * W);

        for (size_t y = 0; y < H; ++y) {
            for (size_t x_idx = 0; x_idx < W; ++x_idx) {
                float gx = 0.0f;
                float gy = 0.0f;

                for (int dy = -1; dy <= 1; ++dy) {
                    int py = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(H) - 1);
                    for (int dx = -1; dx <= 1; ++dx) {
                        int px = std::clamp(static_cast<int>(x_idx) + dx, 0, static_cast<int>(W) - 1);
                        float val = plane[py * W + px];
                        gx += val * kx[dy + 1][dx + 1];
                        gy += val * ky[dy + 1][dx + 1];
                    }
                }
                edge_plane[y * W + x_idx] = std::sqrt(gx * gx + gy * gy + 1e-8f);
            }
        }
    }
    return edges;
}

BoundaryBCELoss::BoundaryBCELoss(float weight) : BaseLoss(weight) {}

TensorPtr BoundaryBCELoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    TensorPtr probs = Tensor::create(logits->shape());
    const float* z = logits->data();
    float* p = probs->data();
    for (size_t i = 0; i < logits->numel(); ++i) {
        p[i] = sigmoid_f(z[i]);
    }

    TensorPtr edge_pred = sobel_edges(probs);
    TensorPtr edge_target = sobel_edges(targets);

    double l1_diff = 0.0;
    size_t n = edge_pred->numel();
    const float* ep = edge_pred->data();
    const float* et = edge_target->data();
    for (size_t i = 0; i < n; ++i) {
        l1_diff += std::abs(ep[i] - et[i]);
    }

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    loss->item() = static_cast<float>(weight_ * (l1_diff / static_cast<double>(n)));
    return loss;
}

BoundaryDiceLoss::BoundaryDiceLoss(float weight, float smooth)
    : BaseLoss(weight), smooth_(smooth) {}

TensorPtr BoundaryDiceLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    TensorPtr probs = Tensor::create(logits->shape());
    const float* z = logits->data();
    float* p = probs->data();
    for (size_t i = 0; i < logits->numel(); ++i) {
        p[i] = sigmoid_f(z[i]);
    }

    TensorPtr edge_pred = sobel_edges(probs);
    TensorPtr edge_target = sobel_edges(targets);

    double inter = 0.0, sum_p = 0.0, sum_t = 0.0;
    size_t n = edge_pred->numel();
    const float* ep = edge_pred->data();
    const float* et = edge_target->data();

    for (size_t i = 0; i < n; ++i) {
        inter += ep[i] * et[i];
        sum_p += ep[i];
        sum_t += et[i];
    }

    double dice = (2.0 * inter + smooth_) / (sum_p + sum_t + smooth_ + 1e-7);
    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    loss->item() = static_cast<float>(weight_ * (1.0 - dice));
    return loss;
}

// -------------------------------------------------------------
// Euclidean Distance Transform (EDT) & Signed Distance Field (SDF)
// Meijster's exact linear-time algorithm O(H * W)
// -------------------------------------------------------------
namespace {

std::vector<float> compute_edt_2d(const std::vector<uint8_t>& binary, size_t H, size_t W) {
    const float INF = 1e9f;
    std::vector<float> dt(H * W, INF);

    // Phase 1: Column-wise forward and backward pass
    for (size_t x = 0; x < W; ++x) {
        // Forward
        if (binary[0 * W + x]) dt[0 * W + x] = 0.0f;
        for (size_t y = 1; y < H; ++y) {
            if (binary[y * W + x]) {
                dt[y * W + x] = 0.0f;
            } else if (dt[(y - 1) * W + x] < INF) {
                dt[y * W + x] = dt[(y - 1) * W + x] + 1.0f;
            }
        }
        // Backward
        for (int y = static_cast<int>(H) - 2; y >= 0; --y) {
            if (dt[(y + 1) * W + x] + 1.0f < dt[y * W + x]) {
                dt[y * W + x] = dt[(y + 1) * W + x] + 1.0f;
            }
        }
    }

    // Phase 2: Row-wise parabolic lower envelope
    std::vector<float> out(H * W, 0.0f);
    std::vector<int> s(W, 0);
    std::vector<float> t(W, 0.0f);

    auto f_parabola = [&](int x_val, int i, float g_val) -> float {
        float dx = static_cast<float>(x_val - i);
        return dx * dx + g_val * g_val;
    };

    auto sep = [&](int i, int u, float g_i, float g_u) -> float {
        return (static_cast<float>(u * u - i * i) + (g_u * g_u - g_i * g_i)) / (2.0f * static_cast<float>(u - i));
    };

    for (size_t y = 0; y < H; ++y) {
        int q = 0;
        s[0] = 0;
        t[0] = 0.0f;

        for (int u = 1; u < static_cast<int>(W); ++u) {
            while (q >= 0 && f_parabola(static_cast<int>(t[q]), s[q], dt[y * W + s[q]]) >
                             f_parabola(static_cast<int>(t[q]), u, dt[y * W + u])) {
                q--;
            }
            if (q < 0) {
                q = 0;
                s[0] = u;
                t[0] = 0.0f;
            } else {
                float w_val = 1.0f + sep(s[q], u, dt[y * W + s[q]], dt[y * W + u]);
                if (w_val < static_cast<float>(W)) {
                    q++;
                    s[q] = u;
                    t[q] = w_val;
                }
            }
        }

        for (int u = static_cast<int>(W) - 1; u >= 0; --u) {
            float dist2 = f_parabola(u, s[q], dt[y * W + s[q]]);
            out[y * W + u] = std::sqrt(dist2);
            if (u == static_cast<int>(t[q]) && q > 0) {
                q--;
            }
        }
    }

    return out;
}

} // anonymous namespace

TensorPtr compute_sdf_2d(const TensorPtr& target) {
    size_t C = target->dim(0);
    size_t H = target->dim(1);
    size_t W = target->dim(2);

    TensorPtr sdf = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)});
    const float* y = target->data();
    float* out = sdf->data();

    for (size_t c = 0; c < C; ++c) {
        const float* plane = y + c * (H * W);
        float* sdf_plane = out + c * (H * W);

        std::vector<uint8_t> pos(H * W, 0);
        std::vector<uint8_t> neg(H * W, 0);
        bool any_pos = false;
        bool any_neg = false;

        for (size_t i = 0; i < H * W; ++i) {
            if (plane[i] > 0.5f) {
                pos[i] = 1;
                any_pos = true;
            } else {
                neg[i] = 1;
                any_neg = true;
            }
        }

        if (!any_pos) {
            std::fill(sdf_plane, sdf_plane + H * W, 1.0f);
            continue;
        }
        if (!any_neg) {
            std::fill(sdf_plane, sdf_plane + H * W, -1.0f);
            continue;
        }

        auto d_out = compute_edt_2d(pos, H, W); // Distance to foreground
        auto d_in = compute_edt_2d(neg, H, W);  // Distance to background

        for (size_t i = 0; i < H * W; ++i) {
            sdf_plane[i] = d_out[i] - d_in[i];
        }
    }

    return sdf;
}

struct BoundaryDistNode : public AutogradNode {
    TensorPtr logits;
    TensorPtr sdf;
    float weight;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (logits && logits->grad_fn()) return {logits->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!logits || !logits->requires_grad()) return;
        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* s = sdf->data();
        float* gz = grad_z->data();
        float go = grad_output->item();
        size_t n = logits->numel();
        float scale = (go * weight) / static_cast<float>(n);

        for (size_t i = 0; i < n; ++i) {
            float p = sigmoid_f(z[i]);
            gz[i] = s[i] * p * (1.0f - p) * scale;
        }

        propagate_grad(logits, grad_z);
    }
};

BoundaryDistLoss::BoundaryDistLoss(float weight) : BaseLoss(weight) {}

TensorPtr BoundaryDistLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    TensorPtr sdf = compute_sdf_2d(targets);
    size_t n = logits->numel();
    const float* z = logits->data();
    const float* s = sdf->data();

    double penalty = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float p = sigmoid_f(z[i]);
        penalty += p * s[i];
    }

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    loss->item() = static_cast<float>(weight_ * (penalty / static_cast<double>(n)));

    if (logits->requires_grad()) {
        auto node = std::make_shared<BoundaryDistNode>();
        node->logits = logits;
        node->sdf = sdf;
        node->weight = weight_;
        loss->set_grad_fn(node);
    }

    return loss;
}

// -------------------------------------------------------------
// Topological Soft Morphological Operations & CLDiceLoss
// -------------------------------------------------------------
TensorPtr soft_erode(const TensorPtr& x) {
    size_t C = x->dim(0);
    size_t H = x->dim(1);
    size_t W = x->dim(2);

    TensorPtr eroded = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)});
    const float* in = x->data();
    float* out = eroded->data();

    for (size_t c = 0; c < C; ++c) {
        const float* plane = in + c * (H * W);
        float* out_plane = out + c * (H * W);

        for (size_t y = 0; y < H; ++y) {
            size_t y_prev = (y > 0) ? (y - 1) : 0;
            size_t y_next = (y + 1 < H) ? (y + 1) : (H - 1);

            for (size_t x_idx = 0; x_idx < W; ++x_idx) {
                size_t x_prev = (x_idx > 0) ? (x_idx - 1) : 0;
                size_t x_next = (x_idx + 1 < W) ? (x_idx + 1) : (W - 1);

                float p1 = std::min({plane[y_prev * W + x_idx], plane[y * W + x_idx], plane[y_next * W + x_idx]});
                float p2 = std::min({plane[y * W + x_prev], plane[y * W + x_idx], plane[y * W + x_next]});
                out_plane[y * W + x_idx] = std::min(p1, p2);
            }
        }
    }
    return eroded;
}

TensorPtr soft_dilate(const TensorPtr& x) {
    size_t C = x->dim(0);
    size_t H = x->dim(1);
    size_t W = x->dim(2);

    TensorPtr dilated = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)});
    const float* in = x->data();
    float* out = dilated->data();

    for (size_t c = 0; c < C; ++c) {
        const float* plane = in + c * (H * W);
        float* out_plane = out + c * (H * W);

        for (size_t y = 0; y < H; ++y) {
            for (size_t x_idx = 0; x_idx < W; ++x_idx) {
                float max_val = plane[y * W + x_idx];
                for (int dy = -1; dy <= 1; ++dy) {
                    int py = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(H) - 1);
                    for (int dx = -1; dx <= 1; ++dx) {
                        int px = std::clamp(static_cast<int>(x_idx) + dx, 0, static_cast<int>(W) - 1);
                        max_val = std::max(max_val, plane[py * W + px]);
                    }
                }
                out_plane[y * W + x_idx] = max_val;
            }
        }
    }
    return dilated;
}

TensorPtr soft_skeletonize(const TensorPtr& x, size_t n_iter) {
    size_t C = x->dim(0);
    size_t H = x->dim(1);
    size_t W = x->dim(2);

    TensorPtr skel = Tensor::zeros({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)});
    TensorPtr curr = x->clone();
    float* s = skel->data();

    for (size_t iter = 0; iter < n_iter; ++iter) {
        TensorPtr eroded = soft_erode(curr);
        TensorPtr opened = soft_dilate(eroded);

        const float* c = curr->data();
        const float* o = opened->data();
        size_t n = curr->numel();

        for (size_t i = 0; i < n; ++i) {
            float delta = std::max(0.0f, c[i] - o[i]);
            s[i] = s[i] + delta * (1.0f - s[i]);
        }
        curr = eroded;
    }
    return skel;
}

struct CLDiceNode : public AutogradNode {
    TensorPtr logits;
    TensorPtr targets;
    float weight;
    float smooth;
    size_t n_iter;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (logits && logits->grad_fn()) return {logits->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!logits || !logits->requires_grad()) return;
        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* y = targets->data();
        float* gz = grad_z->data();
        float go = grad_output->item();
        size_t n = logits->numel();

        TensorPtr probs = Tensor::create(logits->shape());
        float* p = probs->data();
        for (size_t i = 0; i < n; ++i) p[i] = sigmoid_f(z[i]);

        TensorPtr skel_p = soft_skeletonize(probs, n_iter);
        TensorPtr skel_t = soft_skeletonize(targets, n_iter);

        const float* sp = skel_p->data();
        const float* st = skel_t->data();

        double sum_sp_y = 0.0, sum_sp = 0.0;
        double sum_st_p = 0.0, sum_st = 0.0;

        for (size_t i = 0; i < n; ++i) {
            sum_sp_y += sp[i] * y[i];
            sum_sp += sp[i];
            sum_st_p += st[i] * p[i];
            sum_st += st[i];
        }

        double t_prec = (sum_sp_y + smooth) / (sum_sp + smooth + 1e-7);
        double t_sens = (sum_st_p + smooth) / (sum_st + smooth + 1e-7);
        double denom = t_prec + t_sens + 1e-7;

        double d_cldice_d_sens = 2.0 * t_prec * t_prec / (denom * denom);

        for (size_t i = 0; i < n; ++i) {
            double dp_dz = p[i] * (1.0f - p[i]);
            double d_sens_dp = st[i] / (sum_st + smooth + 1e-7);
            double d_loss_dz = -d_cldice_d_sens * d_sens_dp * dp_dz;
            gz[i] = static_cast<float>(go * weight * d_loss_dz);
        }

        propagate_grad(logits, grad_z);
    }
};

CLDiceLoss::CLDiceLoss(float weight, size_t n_iter, float smooth)
    : BaseLoss(weight), n_iter_(n_iter), smooth_(smooth) {}

TensorPtr CLDiceLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    size_t n = logits->numel();
    TensorPtr probs = Tensor::create(logits->shape());
    const float* z = logits->data();
    float* p = probs->data();
    for (size_t i = 0; i < n; ++i) p[i] = sigmoid_f(z[i]);

    TensorPtr skel_p = soft_skeletonize(probs, n_iter_);
    TensorPtr skel_t = soft_skeletonize(targets, n_iter_);

    const float* sp = skel_p->data();
    const float* st = skel_t->data();
    const float* y = targets->data();

    double sum_sp_y = 0.0, sum_sp = 0.0;
    double sum_st_p = 0.0, sum_st = 0.0;

    for (size_t i = 0; i < n; ++i) {
        sum_sp_y += sp[i] * y[i];
        sum_sp += sp[i];
        sum_st_p += st[i] * p[i];
        sum_st += st[i];
    }

    double t_prec = (sum_sp_y + smooth_) / (sum_sp + smooth_ + 1e-7);
    double t_sens = (sum_st_p + smooth_) / (sum_st + smooth_ + 1e-7);
    double cldice = (2.0 * t_prec * t_sens) / (t_prec + t_sens + 1e-7);

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    loss->item() = static_cast<float>(weight_ * (1.0 - cldice));

    if (logits->requires_grad()) {
        auto node = std::make_shared<CLDiceNode>();
        node->logits = logits;
        node->targets = targets;
        node->weight = weight_;
        node->smooth = smooth_;
        node->n_iter = n_iter_;
        loss->set_grad_fn(node);
    }

    return loss;
}

SkeletonLoss::SkeletonLoss(float weight, size_t n_iter)
    : BaseLoss(weight), n_iter_(n_iter) {}

TensorPtr SkeletonLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    size_t n = logits->numel();
    TensorPtr probs = Tensor::create(logits->shape());
    const float* z = logits->data();
    float* p = probs->data();
    for (size_t i = 0; i < n; ++i) p[i] = sigmoid_f(z[i]);

    TensorPtr skel_p = soft_skeletonize(probs, n_iter_);
    TensorPtr skel_t = soft_skeletonize(targets, n_iter_);

    const float* sp = skel_p->data();
    const float* st = skel_t->data();

    double l1 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        l1 += std::abs(sp[i] - st[i]);
    }

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    loss->item() = static_cast<float>(weight_ * (l1 / static_cast<double>(n)));
    return loss;
}

// -------------------------------------------------------------
// SegmentationLoss (Full Composite Scientific Framework)
// -------------------------------------------------------------
SegmentationLoss::SegmentationLoss(float bce_weight, float dice_weight, float pos_weight, float dice_smooth,
                                   float cldice_weight, size_t cldice_warmup_epochs)
    : bce_weight_(bce_weight), dice_weight_(dice_weight), pos_weight_(pos_weight),
      target_structure_weight_(cldice_weight), cldice_warmup_epochs_(cldice_warmup_epochs),
      deep_supervision_(false) {
    region_ = std::make_shared<DiceBCELoss>(1.0f, dice_weight_, bce_weight_, pos_weight_, dice_smooth);
    if (target_structure_weight_ > 0.0f) {
        structure_ = std::make_shared<CLDiceLoss>(target_structure_weight_, 3, 1.0f);
    }
}

SegmentationLoss::SegmentationLoss(std::shared_ptr<BaseLoss> region,
                                   std::shared_ptr<BaseLoss> boundary,
                                   std::shared_ptr<BaseLoss> structure,
                                   float bce_weight,
                                   float dice_weight,
                                   float pos_weight,
                                   float cldice_weight,
                                   size_t cldice_warmup_epochs,
                                   bool deep_supervision,
                                   const std::vector<float>& deep_supervision_weights)
    : region_(region), boundary_(boundary), structure_(structure),
      bce_weight_(bce_weight), dice_weight_(dice_weight), pos_weight_(pos_weight),
      target_structure_weight_(cldice_weight), cldice_warmup_epochs_(cldice_warmup_epochs),
      deep_supervision_(deep_supervision), deep_supervision_weights_(deep_supervision_weights) {

    if (!region_) {
        region_ = std::make_shared<DiceBCELoss>(1.0f, dice_weight_, bce_weight_, pos_weight_, 1.0f);
    }
    if (!structure_ && target_structure_weight_ > 0.0f) {
        structure_ = std::make_shared<CLDiceLoss>(target_structure_weight_, 3, 1.0f);
    }
}

TensorPtr SegmentationLoss::forward(const TensorPtr& logits, const TensorPtr& targets, size_t epoch) {
    LossBreakdown dummy;
    return forward(logits, targets, dummy, epoch);
}

TensorPtr SegmentationLoss::forward(const TensorPtr& logits, const TensorPtr& targets, float& out_bce, float& out_dice) {
    TensorPtr res = forward(logits, targets, 0);
    if (auto dbc = std::dynamic_pointer_cast<DiceBCELoss>(region_)) {
        out_bce = dbc->last_bce();
        out_dice = dbc->last_dice();
    } else {
        out_bce = 0.0f;
        out_dice = 0.0f;
    }
    return res;
}

TensorPtr SegmentationLoss::forward(const TensorPtr& logits, const TensorPtr& targets,
                                    LossBreakdown& breakdown, size_t epoch,
                                    const std::vector<TensorPtr>& auxiliary_preds) {
    TensorPtr reg_loss = region_->forward(logits, targets);
    float total_val = reg_loss->item();
    breakdown.region = reg_loss->item();

    TensorPtr bnd_loss = nullptr;
    if (boundary_) {
        bnd_loss = boundary_->forward(logits, targets);
        total_val += bnd_loss->item();
        breakdown.boundary = bnd_loss->item();
    }

    TensorPtr struct_loss = nullptr;
    if (structure_ && target_structure_weight_ > 0.0f) {
        float scale = 0.0f;
        if (epoch >= cldice_warmup_epochs_) {
            scale = std::min(1.0f, static_cast<float>(epoch - cldice_warmup_epochs_ + 1) /
                                   std::max(1.0f, static_cast<float>(cldice_warmup_epochs_)));
        }
        structure_->set_weight(target_structure_weight_ * scale);
        if (structure_->weight() > 0.0f) {
            struct_loss = structure_->forward(logits, targets);
            total_val += struct_loss->item();
            breakdown.cldice = struct_loss->item();
        }
    }

    float ds_val = 0.0f;
    std::vector<TensorPtr> ds_losses;
    if (deep_supervision_ && !auxiliary_preds.empty()) {
        for (size_t i = 0; i < auxiliary_preds.size() && i < deep_supervision_weights_.size(); ++i) {
            auto l = region_->forward(auxiliary_preds[i], targets);
            ds_val += deep_supervision_weights_[i] * l->item();
            ds_losses.push_back(l);
        }
        total_val += ds_val;
        breakdown.deep_supervision = ds_val;
    }

    breakdown.total = total_val;
    TensorPtr total = Tensor::create({1}, logits->requires_grad());
    total->item() = total_val;

    if (logits->requires_grad()) {
        struct SegLossNode : public AutogradNode {
            TensorPtr reg_t;
            TensorPtr bnd_t;
            TensorPtr struct_t;

            std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
                std::vector<std::shared_ptr<AutogradNode>> res;
                if (reg_t && reg_t->grad_fn()) res.push_back(reg_t->grad_fn());
                if (bnd_t && bnd_t->grad_fn()) res.push_back(bnd_t->grad_fn());
                if (struct_t && struct_t->grad_fn()) res.push_back(struct_t->grad_fn());
                return res;
            }

            void backward(const TensorPtr& grad_output) override {
                float go = grad_output->item();
                TensorPtr unit = Tensor::create({1});
                unit->item() = go;
                if (reg_t) propagate_grad(reg_t, unit);
                if (bnd_t) propagate_grad(bnd_t, unit);
                if (struct_t) propagate_grad(struct_t, unit);
            }
        };
        auto node = std::make_shared<SegLossNode>();
        node->reg_t = reg_loss;
        node->bnd_t = bnd_loss;
        node->struct_t = struct_loss;
        total->set_grad_fn(node);
    }

    return total;
}

} // namespace soar::losses
