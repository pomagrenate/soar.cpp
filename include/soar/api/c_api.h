#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SOAR_API __declspec(dllexport)
#else
#define SOAR_API __attribute__((visibility("default")))
#endif

// Model Variant enum: 0=Nano, 1=Small, 2=Medium, 3=Large, 4=XLarge
SOAR_API void* soar_create_model(int in_channels, int num_classes, int variant);
SOAR_API void soar_destroy_model(void* handle);
SOAR_API size_t soar_get_parameter_count(void* handle);

SOAR_API int soar_save_weights(void* handle, const char* path);
SOAR_API int soar_load_weights(void* handle, const char* path);

SOAR_API int soar_train_step(void* handle,
                             const float* image_data,
                             const float* mask_data,
                             int channels, int height, int width,
                             float learning_rate,
                             float* out_loss,
                             float* out_dice,
                             float* out_iou);

SOAR_API int soar_predict(void* handle,
                          const float* image_data,
                          int channels, int height, int width,
                          float threshold,
                          float* out_probs,
                          uint8_t* out_mask,
                          char* out_rle,
                          size_t max_rle_len);

SOAR_API int soar_encode_rle(const uint8_t* mask,
                             size_t height, size_t width,
                             char* out_rle,
                             size_t max_rle_len);

#ifdef __cplusplus
}
#endif
