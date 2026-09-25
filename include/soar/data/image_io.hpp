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
};

} // namespace soar::data
