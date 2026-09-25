#include <soar/engine/trainer.hpp>
#include <cmath>
#include <algorithm>

namespace soar::engine {

Trainer::Trainer(std::shared_ptr<nn::SOARModel> model,
                 losses::CompositeLoss loss_fn,
                 optim::AdamW optimizer,
                 std::unique_ptr<optim::CosineAnnealingLR> scheduler)
    : model_(std::move(model)), loss_fn_(loss_fn),
      optimizer_(std::move(optimizer)), scheduler_(std::move(scheduler)) {}

StepMetrics Trainer::compute_metrics(const TensorPtr& logits, const TensorPtr& masks, float loss_val) {
    StepMetrics m;
    m.loss = loss_val;
    m.lr = optimizer_.get_lr();

    const float* z = logits->data();
    const float* y = masks->data();
    size_t n = logits->numel();

    double inter = 0.0;
    double sum_pred = 0.0;
    double sum_target = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float p = (1.0f / (1.0f + std::exp(-z[i]))) >= 0.5f ? 1.0f : 0.0f;
        float t = y[i] >= 0.5f ? 1.0f : 0.0f;
        if (p > 0.5f && t > 0.5f) inter += 1.0;
        if (p > 0.5f) sum_pred += 1.0;
        if (t > 0.5f) sum_target += 1.0;
    }

    double union_val = sum_pred + sum_target - inter;
    m.iou_score = static_cast<float>((inter + 1e-5) / (union_val + 1e-5));
    m.dice_score = static_cast<float>((2.0 * inter + 1e-5) / (sum_pred + sum_target + 1e-5));
    return m;
}

StepMetrics Trainer::train_step(const TensorPtr& images, const TensorPtr& masks) {
    model_->train(true);
    optimizer_.zero_grad();

    images->set_requires_grad(false);
    TensorPtr logits = model_->forward(images);
    TensorPtr loss = loss_fn_.forward(logits, masks);

    loss->backward();
    optimizer_.clip_grad_norm(1.0f);
    optimizer_.step();

    if (scheduler_) {
        scheduler_->step();
    }

    return compute_metrics(logits, masks, loss->item());
}

StepMetrics Trainer::evaluate_step(const TensorPtr& images, const TensorPtr& masks) {
    model_->eval();
    TensorPtr logits = model_->forward(images);
    TensorPtr loss = loss_fn_.forward(logits, masks);
    return compute_metrics(logits, masks, loss->item());
}

} // namespace soar::engine
