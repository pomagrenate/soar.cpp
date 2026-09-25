#include <soar/engine/trainer.hpp>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace soar::engine {

Trainer::Trainer(std::shared_ptr<nn::SOARModel> model,
                 losses::CompositeLoss loss_fn,
                 optim::AdamW optimizer,
                 std::unique_ptr<optim::CosineAnnealingLR> scheduler,
                 size_t accumulate_grad_batches,
                 float grad_clip)
    : model_(std::move(model)), loss_fn_(loss_fn),
      optimizer_(std::move(optimizer)), scheduler_(std::move(scheduler)),
      accumulate_grad_batches_(std::max(size_t(1), accumulate_grad_batches)),
      grad_clip_(grad_clip) {}

StepMetrics Trainer::compute_metrics(const TensorPtr& logits, const TensorPtr& masks, float loss_val, float bce_l, float dice_l) {
    StepMetrics m;
    m.loss = loss_val;
    m.bce_loss = bce_l;
    m.dice_loss = dice_l;
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
    m.iou_score = static_cast<float>((inter + 1e-7) / (union_val + 1e-7));
    m.dice_score = static_cast<float>((2.0 * inter + 1e-7) / (sum_pred + sum_target + 1e-7));
    return m;
}

StepMetrics Trainer::train_step(const TensorPtr& images, const TensorPtr& masks, bool is_accumulating) {
    model_->train(true);

    images->set_requires_grad(false);
    std::cout << "    [train_step] Starting forward pass..." << std::endl;
    TensorPtr logits = model_->forward(images);

    std::cout << "    [train_step] Forward done. Computing loss..." << std::endl;
    float bce_l = 0.0f;
    float dice_l = 0.0f;
    TensorPtr loss = loss_fn_.forward(logits, masks, bce_l, dice_l);

    std::cout << "    [train_step] Loss done (" << loss->item() << "). Starting backward..." << std::endl;
    // Virtual batch scaling via gradient accumulation (loss / accumulate_grad_batches)
    float scale = 1.0f / static_cast<float>(accumulate_grad_batches_);
    TensorPtr grad_out = Tensor::create({1});
    grad_out->item() = scale;
    loss->backward(grad_out);

    std::cout << "    [train_step] Backward done. Optimizer step..." << std::endl;
    if (!is_accumulating) {
        optimizer_.clip_grad_norm(grad_clip_);
        optimizer_.step();
        optimizer_.zero_grad();
    }

    return compute_metrics(logits, masks, loss->item(), bce_l, dice_l);
}

void Trainer::step_scheduler() {
    if (scheduler_) {
        scheduler_->step();
    }
}

StepMetrics Trainer::evaluate_step(const TensorPtr& images, const TensorPtr& masks) {
    model_->eval();
    TensorPtr logits = model_->forward(images);
    float bce_l = 0.0f;
    float dice_l = 0.0f;
    TensorPtr loss = loss_fn_.forward(logits, masks, bce_l, dice_l);
    return compute_metrics(logits, masks, loss->item(), bce_l, dice_l);
}

} // namespace soar::engine
