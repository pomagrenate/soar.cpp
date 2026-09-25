#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace soar::utils {

struct PanopticQualityResult {
    float pq{0.0f};
    float sq{0.0f};
    float rq{0.0f};
    size_t tp{0};
    size_t fp{0};
    size_t fn{0};
};

/**
 * @brief Compute Intersection over Union (IoU) for two binary masks.
 */
float compute_iou(const uint8_t* mask1, const uint8_t* mask2, size_t n);
float compute_iou(const std::vector<uint8_t>& mask1, const std::vector<uint8_t>& mask2);

/**
 * @brief Compute Dice coefficient for two binary masks.
 */
float compute_dice(const uint8_t* mask1, const uint8_t* mask2, size_t n);
float compute_dice(const std::vector<uint8_t>& mask1, const std::vector<uint8_t>& mask2);

/**
 * @brief Compute Panoptic Quality (PQ, SQ, RQ) for instance segmentation.
 */
PanopticQualityResult compute_panoptic_quality(
    const std::vector<std::vector<uint8_t>>& pred_masks,
    const std::vector<std::vector<uint8_t>>& gt_masks,
    float iou_threshold = 0.5f);

} // namespace soar::utils
