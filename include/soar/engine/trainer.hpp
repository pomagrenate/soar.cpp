#pragma once

#include <soar/nn/soar_model.hpp>
#include <soar/losses/losses.hpp>
#include <soar/optim/adamw.hpp>

namespace soar::engine {

struct StepMetrics {
    float loss{0.0f};
    float dice_score{0.0f};
    float iou_score{0.0f};
    float lr{0.0f};
};

/**
 * @brief High-performance C++20 Segmentation Trainer for SOAR architectures.
 */
class Trainer {
public:
    Trainer(std::shared_ptr<nn::SOARModel> model,
            losses::CompositeLoss loss_fn,
            optim::AdamW optimizer,
            std::unique_ptr<optim::CosineAnnealingLR> scheduler = nullptr);

    StepMetrics train_step(const TensorPtr& images, const TensorPtr& masks);
    StepMetrics evaluate_step(const TensorPtr& images, const TensorPtr& masks);

    [[nodiscard]] std::shared_ptr<nn::SOARModel> model() const { return model_; }
    [[nodiscard]] optim::AdamW& optimizer() { return optimizer_; }

private:
    StepMetrics compute_metrics(const TensorPtr& logits, const TensorPtr& masks, float loss_val);

    std::shared_ptr<nn::SOARModel> model_;
    losses::CompositeLoss loss_fn_;
    optim::AdamW optimizer_;
    std::unique_ptr<optim::CosineAnnealingLR> scheduler_;
};

} // namespace soar::engine
