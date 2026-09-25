#pragma once

#include <soar/nn/soar_model.hpp>
#include <soar/losses/losses.hpp>
#include <soar/data/coco_dataset.hpp>
#include <soar/data/yolo_dataset.hpp>
#include <string>
#include <vector>
#include <memory>

namespace soar::engine {

struct ValidationMetrics {
    float loss{0.0f};
    float bce_loss{0.0f};
    float dice_loss{0.0f};
    float iou{0.0f};
    float dice{0.0f};
    float accuracy{0.0f};
    size_t samples{0};
};

/**
 * @brief Standalone evaluation & validation engine matching PyTorch / segres BaseValidator.
 *        Computes streaming O(1) memory metrics (Loss, IoU, Dice, Accuracy) and generates visual comparison BMPs.
 */
class Validator {
public:
    explicit Validator(std::shared_ptr<nn::SOARModel> model,
                       losses::CompositeLoss loss_fn = losses::CompositeLoss(1.0f, 1.0f, 3.0f, 1.0f),
                       float threshold = 0.5f);

    /**
     * @brief Validate a single sample without computing autograd gradients.
     */
    ValidationMetrics validate_sample(const TensorPtr& image, const TensorPtr& mask);

    /**
     * @brief Run streaming validation over a dataset (COCO or YOLO).
     */
    template <typename DatasetType>
    ValidationMetrics validate(const DatasetType& dataset,
                               const std::vector<size_t>& sample_indices = {},
                               const std::string& save_vis_dir = "",
                               size_t max_visualizations = 4);

    /**
     * @brief Print clean tabular metric summary matching segres formatting.
     */
    static void print_results(const ValidationMetrics& metrics,
                              size_t epoch = 0,
                              size_t total_epochs = 0,
                              const std::string& stage = "Validation");

    [[nodiscard]] std::shared_ptr<nn::SOARModel> model() const noexcept { return model_; }
    [[nodiscard]] float threshold() const noexcept { return threshold_; }
    void set_threshold(float th) noexcept { threshold_ = th; }

private:
    std::shared_ptr<nn::SOARModel> model_;
    losses::CompositeLoss loss_fn_;
    float threshold_{0.5f};
};

} // namespace soar::engine
