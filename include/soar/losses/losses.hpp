#pragma once

#include <soar/tensor/tensor.hpp>
#include <memory>

namespace soar::losses {

/**
 * @brief Stable Binary Cross-Entropy with Logits Loss.
 */
class BCEWithLogitsLoss {
public:
    explicit BCEWithLogitsLoss(float pos_weight = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets);

private:
    float pos_weight_;
};

/**
 * @brief Soft Dice Loss for segmentation boundary and region overlap.
 */
class DiceLoss {
public:
    explicit DiceLoss(float eps = 1e-5f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets);

private:
    float eps_;
};

/**
 * @brief Composite High-Resolution Segmentation Loss (BCE + Dice).
 */
class CompositeLoss {
public:
    CompositeLoss(float bce_weight = 0.5f, float dice_weight = 0.5f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets);

private:
    float bce_weight_;
    float dice_weight_;
    BCEWithLogitsLoss bce_;
    DiceLoss dice_;
};

} // namespace soar::losses
