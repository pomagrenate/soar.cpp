#pragma once

#include <soar/tensor/tensor.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace soar::data {

/**
 * @brief Normalize tensor to [0, 1] range using percentile clipping.
 */
void normalize_01(TensorPtr& img, float p_low = 1.0f, float p_high = 99.0f);

/**
 * @brief Min-max normalization to [0, 1] range.
 */
void normalize_min_max(TensorPtr& img);

/**
 * @brief Z-score normalization (mean=0, std=1).
 */
void normalize_zscore(TensorPtr& img);

/**
 * @brief Morphological closing (dilation followed by erosion) using an elliptical structuring element.
 */
void morphological_close(const uint8_t* binary, uint8_t* out, size_t H, size_t W, int kernel_size = 3);
std::vector<uint8_t> morphological_close(const std::vector<uint8_t>& binary, size_t H, size_t W, int kernel_size = 3);

/**
 * @brief Remove connected components smaller than min_area using 8-connectivity.
 */
void filter_connected_components(const uint8_t* binary, uint8_t* out, size_t H, size_t W, size_t min_area = 16);
std::vector<uint8_t> filter_connected_components(const std::vector<uint8_t>& binary, size_t H, size_t W, size_t min_area = 16);

/**
 * @brief Extract individual connected components with area >= min_area as separate binary masks.
 */
std::vector<std::vector<uint8_t>> extract_connected_components(const uint8_t* binary, size_t H, size_t W, size_t min_area = 16);
inline std::vector<std::vector<uint8_t>> extract_connected_components(const std::vector<uint8_t>& binary, size_t H, size_t W, size_t min_area = 16) {
    return extract_connected_components(binary.data(), H, W, min_area);
}

} // namespace soar::data
