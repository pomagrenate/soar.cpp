#pragma once

#include <soar/nn/soar_model.hpp>
#include <string>
#include <vector>

namespace soar::engine {

struct PredictionResult {
    TensorPtr probabilities;
    std::vector<uint8_t> binary_mask;
    std::string rle_string;
    size_t foreground_pixels{0};
};

/**
 * @brief High-resolution C++20 Segmentation Predictor with RLE encoding.
 */
class Predictor {
public:
    explicit Predictor(std::shared_ptr<nn::SOARModel> model, float threshold = 0.5f);

    PredictionResult predict(const TensorPtr& input_image);
    static std::string encode_rle(const uint8_t* mask, size_t height, size_t width);

private:
    std::shared_ptr<nn::SOARModel> model_;
    float threshold_;
};

} // namespace soar::engine
