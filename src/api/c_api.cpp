#include <soar/api/c_api.h>
#include <soar/nn/soar_model.hpp>
#include <soar/engine/trainer.hpp>
#include <soar/engine/predictor.hpp>
#include <soar/losses/losses.hpp>
#include <soar/optim/adamw.hpp>

#include <cstring>
#include <memory>

struct SoarModelHandle {
    std::shared_ptr<soar::nn::SOARModel> model;
    std::unique_ptr<soar::engine::Trainer> trainer;
    std::unique_ptr<soar::engine::Predictor> predictor;
};

extern "C" {

SOAR_API void* soar_create_model(int in_channels, int num_classes, int variant) {
    try {
        auto h = new SoarModelHandle();
        soar::nn::ModelVariant var = static_cast<soar::nn::ModelVariant>(variant);
        h->model = std::make_shared<soar::nn::SOARModel>(in_channels, num_classes, var);

        soar::losses::CompositeLoss loss_fn(0.5f, 0.5f);
        soar::optim::AdamWOptions opts;
        opts.lr = 1e-3f;
        soar::optim::AdamW optimizer(h->model->parameters(), opts);

        h->trainer = std::make_unique<soar::engine::Trainer>(h->model, loss_fn, optimizer);
        h->predictor = std::make_unique<soar::engine::Predictor>(h->model, 0.5f);
        return h;
    } catch (...) {
        return nullptr;
    }
}

SOAR_API void soar_destroy_model(void* handle) {
    if (handle) {
        delete static_cast<SoarModelHandle*>(handle);
    }
}

SOAR_API size_t soar_get_parameter_count(void* handle) {
    if (!handle) return 0;
    auto* h = static_cast<SoarModelHandle*>(handle);
    return h->model->parameter_count();
}

SOAR_API int soar_save_weights(void* handle, const char* path) {
    if (!handle || !path) return -1;
    try {
        auto* h = static_cast<SoarModelHandle*>(handle);
        h->model->save_weights(path);
        return 0;
    } catch (...) {
        return -1;
    }
}

SOAR_API int soar_load_weights(void* handle, const char* path) {
    if (!handle || !path) return -1;
    try {
        auto* h = static_cast<SoarModelHandle*>(handle);
        h->model->load_weights(path);
        return 0;
    } catch (...) {
        return -1;
    }
}

SOAR_API int soar_train_step(void* handle,
                             const float* image_data,
                             const float* mask_data,
                             int channels, int height, int width,
                             float learning_rate,
                             float* out_loss,
                             float* out_dice,
                             float* out_iou) {
    if (!handle || !image_data || !mask_data) return -1;
    try {
        auto* h = static_cast<SoarModelHandle*>(handle);
        h->trainer->optimizer().set_lr(learning_rate);

        auto img_tensor = soar::Tensor::from_blob({static_cast<int64_t>(channels),
                                                   static_cast<int64_t>(height),
                                                   static_cast<int64_t>(width)},
                                                  image_data);
        auto msk_tensor = soar::Tensor::from_blob({1,
                                                   static_cast<int64_t>(height),
                                                   static_cast<int64_t>(width)},
                                                  mask_data);

        auto metrics = h->trainer->train_step(img_tensor, msk_tensor);
        if (out_loss) *out_loss = metrics.loss;
        if (out_dice) *out_dice = metrics.dice_score;
        if (out_iou) *out_iou = metrics.iou_score;
        return 0;
    } catch (...) {
        return -1;
    }
}

SOAR_API int soar_predict(void* handle,
                          const float* image_data,
                          int channels, int height, int width,
                          float threshold,
                          float* out_probs,
                          uint8_t* out_mask,
                          char* out_rle,
                          size_t max_rle_len) {
    if (!handle || !image_data) return -1;
    try {
        auto* h = static_cast<SoarModelHandle*>(handle);
        auto img_tensor = soar::Tensor::from_blob({static_cast<int64_t>(channels),
                                                   static_cast<int64_t>(height),
                                                   static_cast<int64_t>(width)},
                                                  image_data);

        soar::engine::Predictor predictor(h->model, threshold);
        auto res = predictor.predict(img_tensor);

        if (out_probs) {
            std::memcpy(out_probs, res.probabilities->data(), res.probabilities->bytes());
        }
        if (out_mask) {
            std::memcpy(out_mask, res.binary_mask.data(), res.binary_mask.size());
        }
        if (out_rle && max_rle_len > 0) {
            std::strncpy(out_rle, res.rle_string.c_str(), max_rle_len - 1);
            out_rle[max_rle_len - 1] = '\0';
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

SOAR_API int soar_encode_rle(const uint8_t* mask,
                             size_t height, size_t width,
                             char* out_rle,
                             size_t max_rle_len) {
    if (!mask || !out_rle || max_rle_len == 0) return -1;
    try {
        std::string rle = soar::engine::Predictor::encode_rle(mask, height, width);
        std::strncpy(out_rle, rle.c_str(), max_rle_len - 1);
        out_rle[max_rle_len - 1] = '\0';
        return 0;
    } catch (...) {
        return -1;
    }
}

} // extern "C"
