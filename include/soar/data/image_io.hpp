#pragma once

#include <soar/tensor/tensor.hpp>
#include <string>

namespace soar::data {

class ImageIO {
public:
    /**
     * @brief Load image from disk into a Tensor of shape [C, H, W] normalized to [0.0, 1.0].
     * @param path Path to image file (PNG, JPG, BMP, etc.).
     * @param desired_channels Number of channels (1 for grayscale, 3 for RGB).
     */
    static TensorPtr load(const std::string& path, int desired_channels = 1);

    /**
     * @brief Save a 1-channel or 3-channel Tensor as a standard 24-bit BMP image.
     * @param path Output file path.
     * @param tensor Tensor of shape [C, H, W] or [H, W].
     */
    static bool save_bmp(const std::string& path, const TensorPtr& tensor);

    /**
     * @brief Save a 4-panel visual validation comparison image as a 24-bit BMP:
     *        [Raw Image | Ground Truth (Green) | Prediction (Cyan) | Error Map (TP=Green, FP=Red, FN=Yellow)]
     * @param path Output file path.
     * @param image Input image [C, H, W] or [1, H, W].
     * @param true_mask Ground truth mask [1, H, W] or [H, W].
     * @param pred_mask Predicted mask [1, H, W] or [H, W].
     * @return True if saved successfully.
     */
    static bool save_comparison_bmp(const std::string& path,
                                    const TensorPtr& image,
                                    const TensorPtr& true_mask,
                                    const TensorPtr& pred_mask);
};

} // namespace soar::data
