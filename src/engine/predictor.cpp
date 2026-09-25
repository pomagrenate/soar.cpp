#include <soar/engine/predictor.hpp>
#include <cmath>
#include <sstream>

namespace soar::engine {

Predictor::Predictor(std::shared_ptr<nn::SOARModel> model, float threshold)
    : model_(std::move(model)), threshold_(threshold) {}

std::string Predictor::encode_rle(const uint8_t* mask, size_t height, size_t width) {
    // Fortran-order (column-major) RLE
    std::ostringstream ss;
    bool in_run = false;
    size_t run_start = 0;
    size_t run_len = 0;

    for (size_t x = 0; x < width; ++x) {
        for (size_t y = 0; y < height; ++y) {
            size_t row_major_idx = y * width + x;
            size_t col_major_idx = x * height + y + 1; // 1-indexed

            uint8_t pixel = mask[row_major_idx];

            if (pixel > 0) {
                if (!in_run) {
                    in_run = true;
                    run_start = col_major_idx;
                    run_len = 1;
                } else {
                    run_len++;
                }
            } else {
                if (in_run) {
                    if (ss.tellp() > 0) ss << " ";
                    ss << run_start << " " << run_len;
                    in_run = false;
                }
            }
        }
    }

    if (in_run) {
        if (ss.tellp() > 0) ss << " ";
        ss << run_start << " " << run_len;
    }

    return ss.str();
}

PredictionResult Predictor::predict(const TensorPtr& input_image) {
    NoGradGuard guard;
    model_->eval();

    TensorPtr logits = model_->forward(input_image);
    size_t H = logits->dim(1);
    size_t W = logits->dim(2);
    size_t n = H * W;

    PredictionResult result;
    result.probabilities = Tensor::create({1, static_cast<int64_t>(H), static_cast<int64_t>(W)});
    result.binary_mask.resize(n, 0);

    const float* z = logits->data();
    float* p = result.probabilities->data();

    for (size_t i = 0; i < n; ++i) {
        float prob = 1.0f / (1.0f + std::exp(-z[i]));
        p[i] = prob;
        if (prob >= threshold_) {
            result.binary_mask[i] = 1;
            result.foreground_pixels++;
        }
    }

    result.rle_string = encode_rle(result.binary_mask.data(), H, W);
    return result;
}

} // namespace soar::engine
