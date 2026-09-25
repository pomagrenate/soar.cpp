#include <soar/losses/losses.hpp>
#include <soar/autograd/node.hpp>
#include <soar/nn/blocks.hpp>

#include <cmath>
#include <algorithm>

namespace soar::losses {

struct BCENode : public AutogradNode {
    TensorPtr logits;
    TensorPtr targets;
    float pos_weight;

    void backward(const TensorPtr& grad_output) override {
        if (!logits->requires_grad()) return;
        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* y = targets->data();
        float* gz = grad_z->data();
        float go = grad_output->item();

        size_t n = logits->numel();
        float scale = go / float(n);

        for (size_t i = 0; i < n; ++i) {
            float sig = 1.0f / (1.0f + std::exp(-z[i]));
            gz[i] = (sig - y[i]) * scale;
        }

        accumulate_grad(logits->grad(), grad_z);
        if (logits->grad_fn()) logits->grad_fn()->backward(grad_z);
    }
};

BCEWithLogitsLoss::BCEWithLogitsLoss(float pos_weight) : pos_weight_(pos_weight) {}

TensorPtr BCEWithLogitsLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    if (logits->shape() != targets->shape()) {
        throw ShapeError("BCEWithLogitsLoss: shape mismatch between logits and targets");
    }

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    const float* z = logits->data();
    const float* y = targets->data();
    size_t n = logits->numel();

    double total_loss = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float zi = z[i];
        float yi = y[i];
        float max_val = std::max(zi, 0.0f);
        float loss_i = max_val - zi * yi + std::log(1.0f + std::exp(-std::abs(zi)));
        total_loss += loss_i;
    }

    loss->item() = static_cast<float>(total_loss / double(n));

    if (logits->requires_grad()) {
        auto node = std::make_shared<BCENode>();
        node->logits = logits;
        node->targets = targets;
        node->pos_weight = pos_weight_;
        loss->set_grad_fn(node);
    }

    return loss;
}

struct DiceNode : public AutogradNode {
    TensorPtr logits;
    TensorPtr targets;
    float eps;

    void backward(const TensorPtr& grad_output) override {
        if (!logits->requires_grad()) return;
        TensorPtr grad_z = Tensor::zeros(logits->shape());
        const float* z = logits->data();
        const float* y = targets->data();
        float* gz = grad_z->data();
        float go = grad_output->item();
        size_t n = logits->numel();

        double inter = 0.0;
        double sum_p = 0.0;
        double sum_y = 0.0;

        std::vector<float> p(n);
        for (size_t i = 0; i < n; ++i) {
            float pi = 1.0f / (1.0f + std::exp(-z[i]));
            p[i] = pi;
            inter += pi * y[i];
            sum_p += pi;
            sum_y += y[i];
        }

        double union_sum = sum_p + sum_y + eps;
        double dice_num = 2.0 * inter + eps;

        for (size_t i = 0; i < n; ++i) {
            double d_dice_dp = (2.0 * y[i] * union_sum - dice_num) / (union_sum * union_sum);
            double dp_dz = p[i] * (1.0f - p[i]);
            gz[i] = static_cast<float>(-d_dice_dp * dp_dz * go);
        }

        accumulate_grad(logits->grad(), grad_z);
        if (logits->grad_fn()) logits->grad_fn()->backward(grad_z);
    }
};

DiceLoss::DiceLoss(float eps) : eps_(eps) {}

TensorPtr DiceLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    if (logits->shape() != targets->shape()) {
        throw ShapeError("DiceLoss: shape mismatch between logits and targets");
    }

    TensorPtr loss = Tensor::create({1}, logits->requires_grad());
    const float* z = logits->data();
    const float* y = targets->data();
    size_t n = logits->numel();

    double inter = 0.0;
    double sum_p = 0.0;
    double sum_y = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float p = 1.0f / (1.0f + std::exp(-z[i]));
        inter += p * y[i];
        sum_p += p;
        sum_y += y[i];
    }

    double dice = (2.0 * inter + eps_) / (sum_p + sum_y + eps_);
    loss->item() = static_cast<float>(1.0 - dice);

    if (logits->requires_grad()) {
        auto node = std::make_shared<DiceNode>();
        node->logits = logits;
        node->targets = targets;
        node->eps = eps_;
        loss->set_grad_fn(node);
    }

    return loss;
}

CompositeLoss::CompositeLoss(float bce_weight, float dice_weight)
    : bce_weight_(bce_weight), dice_weight_(dice_weight) {}

TensorPtr CompositeLoss::forward(const TensorPtr& logits, const TensorPtr& targets) {
    TensorPtr l_bce = bce_.forward(logits, targets);
    TensorPtr l_dice = dice_.forward(logits, targets);

    TensorPtr total = Tensor::create({1}, logits->requires_grad());
    total->item() = bce_weight_ * l_bce->item() + dice_weight_ * l_dice->item();

    // Attach combined backward node
    if (logits->requires_grad()) {
        struct CombinedLossNode : public AutogradNode {
            TensorPtr bce_loss;
            TensorPtr dice_loss;
            float w_bce;
            float w_dice;
            void backward(const TensorPtr& grad_output) override {
                float go = grad_output->item();
                TensorPtr go_bce = Tensor::create({1});
                go_bce->item() = go * w_bce;
                TensorPtr go_dice = Tensor::create({1});
                go_dice->item() = go * w_dice;

                if (bce_loss->grad_fn()) bce_loss->grad_fn()->backward(go_bce);
                if (dice_loss->grad_fn()) dice_loss->grad_fn()->backward(go_dice);
            }
        };

        auto node = std::make_shared<CombinedLossNode>();
        node->bce_loss = l_bce;
        node->dice_loss = l_dice;
        node->w_bce = bce_weight_;
        node->w_dice = dice_weight_;
        total->set_grad_fn(node);
    }

    return total;
}

} // namespace soar::losses
