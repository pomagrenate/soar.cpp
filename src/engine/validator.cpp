#include <soar/engine/validator.hpp>
#include <soar/data/image_io.hpp>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <filesystem>

namespace soar::engine {

Validator::Validator(std::shared_ptr<nn::SOARModel> model,
                     losses::CompositeLoss loss_fn,
                     float threshold)
    : model_(std::move(model)), loss_fn_(loss_fn), threshold_(threshold) {}

ValidationMetrics Validator::validate_sample(const TensorPtr& image, const TensorPtr& mask) {
    NoGradGuard guard;
    model_->eval();

    TensorPtr logits = model_->forward(image);
    float bce_l = 0.0f;
    float dice_l = 0.0f;
    TensorPtr loss = loss_fn_.forward(logits, mask, bce_l, dice_l);

    const float* z = logits->data();
    const float* y = mask->data();
    size_t n = logits->numel();

    double inter = 0.0;
    double union_sum = 0.0;
    double card = 0.0;
    double correct = 0.0;

    for (size_t i = 0; i < n; ++i) {
        float p = (1.0f / (1.0f + std::exp(-z[i]))) >= threshold_ ? 1.0f : 0.0f;
        float t = y[i] >= 0.5f ? 1.0f : 0.0f;

        if (p > 0.5f && t > 0.5f) inter += 1.0;
        if (p > 0.5f || t > 0.5f) union_sum += 1.0;
        if (p > 0.5f) card += 1.0;
        if (t > 0.5f) card += 1.0;
        if (p == t) correct += 1.0;
    }

    ValidationMetrics m;
    m.loss = loss->item();
    m.bce_loss = bce_l;
    m.dice_loss = dice_l;
    m.iou = static_cast<float>((inter + 1e-7) / (union_sum + 1e-7));
    m.dice = static_cast<float>((2.0 * inter + 1e-7) / (card + 1e-7));
    m.accuracy = static_cast<float>(correct / static_cast<double>(n));
    m.samples = 1;
    return m;
}

template <typename DatasetType>
ValidationMetrics Validator::validate(const DatasetType& dataset,
                                      const std::vector<size_t>& sample_indices,
                                      const std::string& save_vis_dir,
                                      size_t max_visualizations) {
    model_->eval();

    const std::vector<size_t>& indices = sample_indices.empty() ? [&]() {
        static std::vector<size_t> all_idx;
        all_idx.resize(dataset.size());
        for (size_t i = 0; i < all_idx.size(); ++i) all_idx[i] = i;
        return all_idx;
    }() : sample_indices;

    if (indices.empty()) {
        return ValidationMetrics{};
    }

    if (!save_vis_dir.empty()) {
        std::filesystem::create_directories(save_vis_dir);
    }

    double total_loss = 0.0;
    double total_bce = 0.0;
    double total_dice_loss = 0.0;
    double total_inter = 0.0;
    double total_union = 0.0;
    double total_card = 0.0;
    double total_correct = 0.0;
    double total_pixels = 0.0;

    size_t vis_count = 0;

    for (size_t idx : indices) {
        auto sample = dataset.get_sample(idx);
        TensorPtr logits = model_->forward(sample.image);

        float bce_l = 0.0f;
        float dice_l = 0.0f;
        TensorPtr loss = loss_fn_.forward(logits, sample.mask, bce_l, dice_l);

        total_loss += loss->item();
        total_bce += bce_l;
        total_dice_loss += dice_l;

        const float* z = logits->data();
        const float* y = sample.mask->data();
        size_t n = logits->numel();
        total_pixels += static_cast<double>(n);

        TensorPtr prob_tensor = nullptr;
        if (!save_vis_dir.empty() && vis_count < max_visualizations) {
            prob_tensor = Tensor::zeros(sample.mask->shape());
        }

        for (size_t i = 0; i < n; ++i) {
            float prob = 1.0f / (1.0f + std::exp(-z[i]));
            if (prob_tensor) {
                prob_tensor->data()[i] = prob;
            }

            float p = prob >= threshold_ ? 1.0f : 0.0f;
            float t = y[i] >= 0.5f ? 1.0f : 0.0f;

            if (p > 0.5f && t > 0.5f) total_inter += 1.0;
            if (p > 0.5f || t > 0.5f) total_union += 1.0;
            if (p > 0.5f) total_card += 1.0;
            if (t > 0.5f) total_card += 1.0;
            if (p == t) total_correct += 1.0;
        }

        if (prob_tensor && !save_vis_dir.empty() && vis_count < max_visualizations) {
            std::string out_path = (std::filesystem::path(save_vis_dir) /
                                   ("val_sample_" + std::to_string(vis_count) + ".bmp")).string();
            data::ImageIO::save_comparison_bmp(out_path, sample.image, sample.mask, prob_tensor);
            vis_count++;
        }
    }

    size_t count = indices.size();
    ValidationMetrics out;
    out.loss = static_cast<float>(total_loss / static_cast<double>(count));
    out.bce_loss = static_cast<float>(total_bce / static_cast<double>(count));
    out.dice_loss = static_cast<float>(total_dice_loss / static_cast<double>(count));
    out.iou = static_cast<float>((total_inter + 1e-7) / (total_union + 1e-7));
    out.dice = static_cast<float>((2.0 * total_inter + 1e-7) / (total_card + 1e-7));
    out.accuracy = static_cast<float>(total_correct / std::max(total_pixels, 1.0));
    out.samples = count;

    return out;
}

// Explicit template instantiations
template ValidationMetrics Validator::validate<data::COCODataset>(
    const data::COCODataset&, const std::vector<size_t>&, const std::string&, size_t);
template ValidationMetrics Validator::validate<data::YOLODataset>(
    const data::YOLODataset&, const std::vector<size_t>&, const std::string&, size_t);

void Validator::print_results(const ValidationMetrics& m,
                              size_t epoch,
                              size_t total_epochs,
                              const std::string& stage) {
    std::string header_tag = stage;
    if (epoch > 0) {
        if (total_epochs > 0) {
            header_tag = "Epoch " + std::to_string(epoch) + "/" + std::to_string(total_epochs);
        } else {
            header_tag = "Epoch " + std::to_string(epoch);
        }
    }

    std::cout << "\n---------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(22) << "Stage / Metric"
              << std::setw(10) << "Samples"
              << std::setw(12) << "Loss"
              << std::setw(12) << "IoU"
              << std::setw(12) << "Dice"
              << std::setw(10) << "Acc" << "\n";
    std::cout << "---------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(22) << header_tag
              << std::setw(10) << m.samples
              << std::fixed << std::setprecision(4)
              << std::setw(12) << m.loss
              << std::setw(12) << m.iou
              << std::setw(12) << m.dice
              << std::setw(10) << m.accuracy << "\n";
    std::cout << "---------------------------------------------------------------------------------\n" << std::endl;
}

} // namespace soar::engine
