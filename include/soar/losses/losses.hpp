#pragma once

#include <soar/tensor/tensor.hpp>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace soar::losses {

/**
 * @brief Loss breakdown dictionary returned by composite segmentation losses.
 */
struct LossBreakdown {
    float total{0.0f};
    float region{0.0f};
    float boundary{0.0f};
    float cldice{0.0f};
    float deep_supervision{0.0f};
};

/**
 * @brief Base interface for all segmentation loss functions.
 */
class BaseLoss {
public:
    explicit BaseLoss(float weight = 1.0f) : weight_(weight) {}
    virtual ~BaseLoss() = default;

    virtual TensorPtr forward(const TensorPtr& pred, const TensorPtr& target) = 0;

    [[nodiscard]] float weight() const noexcept { return weight_; }
    void set_weight(float w) noexcept { weight_ = w; }

protected:
    float weight_{1.0f};
};

/**
 * @brief Stable Binary Cross-Entropy with Logits Loss with pos_weight support matching PyTorch aten.
 */
class BCEWithLogitsLoss : public BaseLoss {
public:
    explicit BCEWithLogitsLoss(float weight = 1.0f, float pos_weight = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

    [[nodiscard]] float pos_weight() const noexcept { return pos_weight_; }
    void set_pos_weight(float pw) noexcept { pos_weight_ = pw; }

private:
    float pos_weight_{1.0f};
};

/**
 * @brief Soft Dice Loss for segmentation boundary and region overlap with float32 reduction.
 */
class DiceLoss : public BaseLoss {
public:
    explicit DiceLoss(float weight = 1.0f, float smooth = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

    [[nodiscard]] float smooth() const noexcept { return smooth_; }
    void set_smooth(float s) noexcept { smooth_ = s; }

private:
    float smooth_{1.0f};
};

/**
 * @brief Focal loss for extreme foreground-background class imbalance (Lin et al.).
 * loss = alpha_t * (1 - pt)^gamma * BCE(logits, targets)
 */
class FocalLoss : public BaseLoss {
public:
    explicit FocalLoss(float weight = 1.0f, float gamma = 2.0f, float alpha = 0.25f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

    [[nodiscard]] float gamma() const noexcept { return gamma_; }
    [[nodiscard]] float alpha() const noexcept { return alpha_; }

private:
    float gamma_{2.0f};
    float alpha_{0.25f};
};

/**
 * @brief Tversky Loss controlling false-positive and false-negative penalties.
 */
class TverskyLoss : public BaseLoss {
public:
    explicit TverskyLoss(float weight = 1.0f, float alpha = 0.5f, float beta = 0.5f, float smooth = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

    [[nodiscard]] float alpha() const noexcept { return alpha_; }
    [[nodiscard]] float beta() const noexcept { return beta_; }
    [[nodiscard]] float smooth() const noexcept { return smooth_; }

private:
    float alpha_{0.5f};
    float beta_{0.5f};
    float smooth_{1.0f};
};

/**
 * @brief Focal Tversky Loss focusing on challenging curvilinear structures.
 */
class FocalTverskyLoss : public BaseLoss {
public:
    explicit FocalTverskyLoss(float weight = 1.0f, float alpha = 0.5f, float beta = 0.5f,
                              float gamma = 2.0f, float smooth = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

private:
    float alpha_{0.5f};
    float beta_{0.5f};
    float gamma_{2.0f};
    float smooth_{1.0f};
};

/**
 * @brief Composite Dice and Binary Cross-Entropy baseline loss.
 */
class DiceBCELoss : public BaseLoss {
public:
    explicit DiceBCELoss(float weight = 1.0f,
                         float dice_weight = 1.0f,
                         float bce_weight = 1.0f,
                         float bce_pos_weight = 1.0f,
                         float smooth = 1.0f);

    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

    [[nodiscard]] float bce_weight() const noexcept { return bce_weight_; }
    [[nodiscard]] float dice_weight() const noexcept { return dice_weight_; }
    [[nodiscard]] float last_bce() const noexcept { return last_bce_; }
    [[nodiscard]] float last_dice() const noexcept { return last_dice_; }

private:
    float dice_weight_{1.0f};
    float bce_weight_{1.0f};
    BCEWithLogitsLoss bce_;
    DiceLoss dice_;
    float last_bce_{0.0f};
    float last_dice_{0.0f};
    float last_inter_{0.0f};
    float last_sum_p_{0.0f};
    float last_sum_y_{0.0f};
};

/**
 * @brief Sobel edge detection helper. Computes gradient magnitude of 2D input.
 */
TensorPtr sobel_edges(const TensorPtr& x);

/**
 * @brief Boundary-aware loss using Sobel edge detection and L1 difference.
 */
class BoundaryBCELoss : public BaseLoss {
public:
    explicit BoundaryBCELoss(float weight = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;
};

/**
 * @brief Boundary Dice loss on spatial gradient maps.
 */
class BoundaryDiceLoss : public BaseLoss {
public:
    explicit BoundaryDiceLoss(float weight = 1.0f, float smooth = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

private:
    float smooth_{1.0f};
};

/**
 * @brief Euclidean Distance Transform (EDT) and Signed Distance Field (SDF) calculation.
 * Computes exact signed distance transform where boundary is 0, foreground is negative,
 * background is positive (matching SciPy distance_transform_edt).
 */
TensorPtr compute_sdf_2d(const TensorPtr& target);

/**
 * @brief Continuous Signed Distance Transform Boundary Loss (Kervadec et al.).
 * Integrates predicted probability fields with target SDF.
 */
class BoundaryDistLoss : public BaseLoss {
public:
    explicit BoundaryDistLoss(float weight = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;
};

// Topological soft morphological operations
TensorPtr soft_erode(const TensorPtr& x);
TensorPtr soft_dilate(const TensorPtr& x);
TensorPtr soft_skeletonize(const TensorPtr& x, size_t n_iter = 3);

/**
 * @brief Continuous clDice loss for curvilinear and thin topological structure segmentation.
 */
class CLDiceLoss : public BaseLoss {
public:
    explicit CLDiceLoss(float weight = 1.0f, size_t n_iter = 3, float smooth = 1.0f);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

    [[nodiscard]] size_t n_iter() const noexcept { return n_iter_; }
    [[nodiscard]] float smooth() const noexcept { return smooth_; }

private:
    size_t n_iter_{3};
    float smooth_{1.0f};
};

/**
 * @brief Direct L1 penalty on soft topological skeletons.
 */
class SkeletonLoss : public BaseLoss {
public:
    explicit SkeletonLoss(float weight = 1.0f, size_t n_iter = 3);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets) override;

private:
    size_t n_iter_{3};
};

/**
 * @brief Full Scientific Segmentation Loss framework matching segres.
 * Combines pixel region (DiceBCE), boundary gradients/SDF, and topological CLDice with warmup scheduling.
 */
class SegmentationLoss {
public:
    SegmentationLoss(float bce_weight, float dice_weight, float pos_weight = 3.0f, float dice_smooth = 1.0f,
                     float cldice_weight = 0.0f, size_t cldice_warmup_epochs = 0);

    SegmentationLoss(std::shared_ptr<BaseLoss> region = nullptr,
                     std::shared_ptr<BaseLoss> boundary = nullptr,
                     std::shared_ptr<BaseLoss> structure = nullptr,
                     float bce_weight = 1.0f,
                     float dice_weight = 1.0f,
                     float pos_weight = 3.0f,
                     float cldice_weight = 0.2f,
                     size_t cldice_warmup_epochs = 5,
                     bool deep_supervision = false,
                     const std::vector<float>& deep_supervision_weights = {0.4f, 0.3f, 0.2f, 0.1f});

    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets, size_t epoch = 0);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets, float& out_bce, float& out_dice);
    TensorPtr forward(const TensorPtr& logits, const TensorPtr& targets,
                      LossBreakdown& breakdown, size_t epoch = 0,
                      const std::vector<TensorPtr>& auxiliary_preds = {});

    [[nodiscard]] float bce_weight() const noexcept { return bce_weight_; }
    [[nodiscard]] float dice_weight() const noexcept { return dice_weight_; }
    [[nodiscard]] float pos_weight() const noexcept { return pos_weight_; }
    [[nodiscard]] float cldice_weight() const noexcept { return target_structure_weight_; }

private:
    std::shared_ptr<BaseLoss> region_{nullptr};
    std::shared_ptr<BaseLoss> boundary_{nullptr};
    std::shared_ptr<BaseLoss> structure_{nullptr};

    float bce_weight_{1.0f};
    float dice_weight_{1.0f};
    float pos_weight_{3.0f};
    float target_structure_weight_{0.2f};
    size_t cldice_warmup_epochs_{5};
    bool deep_supervision_{false};
    std::vector<float> deep_supervision_weights_;
};

/// Backwards compatibility alias for CompositeLoss
using CompositeLoss = SegmentationLoss;

} // namespace soar::losses
