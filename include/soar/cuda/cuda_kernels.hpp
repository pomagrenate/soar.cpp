#pragma once

#include <cstddef>
#include <cstdint>

namespace soar::cuda::kernels {

bool is_cuda_available();

// Memory initialization
void cuda_zero(float* ptr, size_t n, void* stream = nullptr);
void cuda_fill(float* ptr, float val, size_t n, void* stream = nullptr);

// Elementwise Add: y = a + b
void add_forward(const float* a, const float* b, float* y, size_t n, void* stream = nullptr);
void add_backward(const float* grad_y, float* grad_a, float* grad_b, size_t n, void* stream = nullptr);

// SiLU Activation: y = x * sigmoid(x) = x / (1 + exp(-x))
void silu_forward(const float* x, float* y, size_t n, void* stream = nullptr);
void silu_backward(const float* grad_y, const float* x, float* grad_x, size_t n, void* stream = nullptr);

// Sigmoid Activation: y = 1 / (1 + exp(-x))
void sigmoid_forward(const float* x, float* y, size_t n, void* stream = nullptr);
void sigmoid_backward(const float* grad_y, const float* y, float* grad_x, size_t n, void* stream = nullptr);

// MulBroadcast: y[c, sp] = a[c, sp] * b[c] (for SE channel attention)
void mul_broadcast_forward(const float* a, const float* b, float* y, size_t C, size_t HW, void* stream = nullptr);
void mul_broadcast_backward(const float* grad_y, const float* a, const float* b,
                           float* grad_a, float* grad_b, size_t C, size_t HW, void* stream = nullptr);

// Convex Combination: y = g * a + (1 - g) * b
void convex_combination_forward(const float* g, const float* a, const float* b, float* y, size_t n, void* stream = nullptr);
void convex_combination_backward(const float* grad_y, const float* g, const float* a, const float* b,
                                float* grad_g, float* grad_a, float* grad_b, size_t n, void* stream = nullptr);

// Global Average Pooling 2D: [C, H, W] -> [C, 1, 1]
void global_avg_pool_forward(const float* in, float* out, size_t C, size_t HW, void* stream = nullptr);
void global_avg_pool_backward(const float* grad_out, float* grad_in, size_t C, size_t HW, void* stream = nullptr);

// Upsample Bilinear 2D
void upsample_bilinear_2x_forward(const float* in, float* out, size_t C, size_t H_in, size_t W_in, void* stream = nullptr);
void upsample_bilinear_2x_backward(const float* grad_out, float* grad_in, size_t C, size_t H_in, size_t W_in, void* stream = nullptr);
void upsample_bilinear_forward(const float* in, float* out, size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out, void* stream = nullptr);
void upsample_bilinear_backward(const float* grad_out, float* grad_in, size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out, void* stream = nullptr);

// PixelShuffle
void pixel_shuffle_forward(const float* in, float* out, size_t C_in, size_t H_in, size_t W_in, size_t r, void* stream = nullptr);
void pixel_shuffle_backward(const float* grad_out, float* grad_in, size_t C_in, size_t H_in, size_t W_in, size_t r, void* stream = nullptr);

// MaxPool2D
void max_pool2d_forward(const float* in, float* out, int64_t* argmax, size_t C, size_t H_in, size_t W_in,
                        size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, void* stream = nullptr);
void max_pool2d_backward(const float* grad_out, const int64_t* argmax, float* grad_in,
                         size_t total_out, void* stream = nullptr);

// Conv2D 1x1 Pointwise
void conv2d_1x1_forward(const float* in, const float* weight, const float* bias, float* out,
                        size_t C_in, size_t C_out, size_t HW, void* stream = nullptr);
void conv2d_1x1_backward(const float* grad_out, const float* in, const float* weight,
                         float* grad_in, float* grad_weight, float* grad_bias,
                         size_t C_in, size_t C_out, size_t HW, void* stream = nullptr);

// Conv2D Depthwise
void conv2d_dw_forward(const float* in, const float* weight, const float* bias, float* out,
                       size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out,
                       size_t K, size_t pad, size_t stride, size_t dil, void* stream = nullptr);
void conv2d_dw_backward(const float* grad_out, const float* in, const float* weight,
                        float* grad_in, float* grad_weight, float* grad_bias,
                        size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out,
                        size_t K, size_t pad, size_t stride, size_t dil, void* stream = nullptr);

// General Grouped Conv2D
void conv2d_forward(const float* in, const float* weight, const float* bias, float* out,
                    size_t C_in, size_t C_out, size_t H_in, size_t W_in,
                    size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, size_t dil, size_t groups, void* stream = nullptr);
void conv2d_backward(const float* grad_out, const float* in, const float* weight,
                     float* grad_in, float* grad_weight, float* grad_bias,
                     size_t C_in, size_t C_out, size_t H_in, size_t W_in,
                     size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, size_t dil, size_t groups, void* stream = nullptr);

// Group Normalization with Affine
void group_norm_forward(const float* in, const float* gamma, const float* beta, float* out,
                        float* saved_mean, float* saved_rstd,
                        size_t num_groups, size_t C, size_t HW, float eps, void* stream = nullptr);
void group_norm_backward(const float* grad_out, const float* in, const float* gamma,
                         const float* saved_mean, const float* saved_rstd,
                         float* grad_in, float* grad_gamma, float* grad_beta,
                         size_t num_groups, size_t C, size_t HW, void* stream = nullptr);

// Loss Functions
float bce_with_logits_forward(const float* logits, const float* targets,
                              float weight, float pos_weight, size_t n, void* stream = nullptr);
void bce_with_logits_backward(const float* logits, const float* targets, float* grad_logits,
                             float grad_out, float weight, float pos_weight, size_t n, void* stream = nullptr);

float dice_loss_forward(const float* logits, const float* targets,
                        float weight, float smooth, float& out_inter, float& out_sum_p, float& out_sum_y,
                        size_t n, void* stream = nullptr);
void dice_loss_backward(const float* logits, const float* targets, float* grad_logits,
                        float grad_out, float weight, float smooth,
                        float inter, float sum_p, float sum_y, size_t n, void* stream = nullptr);

void dice_bce_loss_backward(const float* logits, const float* targets, float* grad_logits,
                            float grad_out, float w, float bw, float dw, float pos_weight, float smooth,
                            float inter, float sum_p, float sum_y, size_t n, void* stream = nullptr);

// AdamW Optimizer Step
void adamw_step(float* theta, const float* g, float* m, float* v,
                float lr, float beta1, float beta2, float eps, float wd,
                float step_size, float sqrt_bc2, size_t n, void* stream = nullptr);

} // namespace soar::cuda::kernels

