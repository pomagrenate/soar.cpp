#include <soar/utils/metrics.hpp>
#include <unordered_set>
#include <algorithm>

namespace soar::utils {

float compute_iou(const uint8_t* mask1, const uint8_t* mask2, size_t n) {
    if (n == 0) return 1.0f;
    size_t inter = 0;
    size_t un = 0;
    for (size_t i = 0; i < n; ++i) {
        bool m1 = (mask1[i] > 0);
        bool m2 = (mask2[i] > 0);
        if (m1 && m2) inter++;
        if (m1 || m2) un++;
    }
    return static_cast<float>(inter) / (static_cast<float>(un) + 1e-8f);
}

float compute_iou(const std::vector<uint8_t>& mask1, const std::vector<uint8_t>& mask2) {
    size_t n = std::min(mask1.size(), mask2.size());
    return compute_iou(mask1.data(), mask2.data(), n);
}

float compute_dice(const uint8_t* mask1, const uint8_t* mask2, size_t n) {
    if (n == 0) return 1.0f;
    size_t inter = 0;
    size_t sum1 = 0;
    size_t sum2 = 0;
    for (size_t i = 0; i < n; ++i) {
        bool m1 = (mask1[i] > 0);
        bool m2 = (mask2[i] > 0);
        if (m1 && m2) inter++;
        if (m1) sum1++;
        if (m2) sum2++;
    }
    return (2.0f * static_cast<float>(inter)) / (static_cast<float>(sum1 + sum2) + 1e-8f);
}

float compute_dice(const std::vector<uint8_t>& mask1, const std::vector<uint8_t>& mask2) {
    size_t n = std::min(mask1.size(), mask2.size());
    return compute_dice(mask1.data(), mask2.data(), n);
}

PanopticQualityResult compute_panoptic_quality(
    const std::vector<std::vector<uint8_t>>& pred_masks,
    const std::vector<std::vector<uint8_t>>& gt_masks,
    float iou_threshold) {

    PanopticQualityResult res{};

    if (pred_masks.empty() && gt_masks.empty()) {
        res.pq = 1.0f;
        res.sq = 1.0f;
        res.rq = 1.0f;
        return res;
    }

    if (pred_masks.empty() || gt_masks.empty()) {
        res.pq = 0.0f;
        res.sq = 0.0f;
        res.rq = 0.0f;
        res.fp = pred_masks.size();
        res.fn = gt_masks.size();
        return res;
    }

    std::unordered_set<size_t> matched_gt;
    float iou_sum = 0.0f;
    size_t tp = 0;

    for (const auto& p : pred_masks) {
        float best_iou = 0.0f;
        int64_t best_idx = -1;

        for (size_t j = 0; j < gt_masks.size(); ++j) {
            if (matched_gt.find(j) != matched_gt.end()) continue;
            float curr_iou = compute_iou(p, gt_masks[j]);
            if (curr_iou > best_iou) {
                best_iou = curr_iou;
                best_idx = static_cast<int64_t>(j);
            }
        }

        if (best_iou >= iou_threshold && best_idx >= 0) {
            matched_gt.insert(static_cast<size_t>(best_idx));
            iou_sum += best_iou;
            tp++;
        }
    }

    size_t fp = pred_masks.size() - tp;
    size_t fn = gt_masks.size() - matched_gt.size();

    float sq = (tp > 0) ? (iou_sum / (static_cast<float>(tp) + 1e-8f)) : 0.0f;
    float rq = static_cast<float>(tp) / (static_cast<float>(tp) + 0.5f * static_cast<float>(fp) + 0.5f * static_cast<float>(fn) + 1e-8f);

    res.tp = tp;
    res.fp = fp;
    res.fn = fn;
    res.sq = sq;
    res.rq = rq;
    res.pq = sq * rq;

    return res;
}

} // namespace soar::utils
