#include <soar/cuda/cuda_kernels.hpp>
#include <soar/core/error.hpp>
#include <stdexcept>

namespace soar::cuda::kernels {

#if !defined(SOAR_HAS_CUDA)

bool is_cuda_available() {
    return false;
}

void cuda_zero(float* /*ptr*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void cuda_fill(float* /*ptr*/, float /*val*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void add_forward(const float* /*a*/, const float* /*b*/, float* /*y*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void add_backward(const float* /*grad_y*/, float* /*grad_a*/, float* /*grad_b*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void silu_forward(const float* /*x*/, float* /*y*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void silu_backward(const float* /*grad_y*/, const float* /*x*/, float* /*grad_x*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void mul_broadcast_forward(const float* /*a*/, const float* /*b*/, float* /*y*/, size_t /*C*/, size_t /*HW*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void mul_broadcast_backward(const float* /*grad_y*/, const float* /*a*/, const float* /*b*/,
                           float* /*grad_a*/, float* /*grad_b*/, size_t /*C*/, size_t /*HW*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void global_avg_pool_forward(const float* /*in*/, float* /*out*/, size_t /*C*/, size_t /*HW*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void global_avg_pool_backward(const float* /*grad_out*/, float* /*grad_in*/, size_t /*C*/, size_t /*HW*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void upsample_bilinear_2x_forward(const float* /*in*/, float* /*out*/, size_t /*C*/, size_t /*H_in*/, size_t /*W_in*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void upsample_bilinear_2x_backward(const float* /*grad_out*/, float* /*grad_in*/, size_t /*C*/, size_t /*H_in*/, size_t /*W_in*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void conv2d_1x1_forward(const float* /*in*/, const float* /*weight*/, const float* /*bias*/, float* /*out*/,
                        size_t /*C_in*/, size_t /*C_out*/, size_t /*HW*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void conv2d_1x1_backward(const float* /*grad_out*/, const float* /*in*/, const float* /*weight*/,
                         float* /*grad_in*/, float* /*grad_weight*/, float* /*grad_bias*/,
                         size_t /*C_in*/, size_t /*C_out*/, size_t /*HW*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void conv2d_dw_forward(const float* /*in*/, const float* /*weight*/, const float* /*bias*/, float* /*out*/,
                       size_t /*C*/, size_t /*H_in*/, size_t /*W_in*/, size_t /*H_out*/, size_t /*W_out*/,
                       size_t /*K*/, size_t /*pad*/, size_t /*stride*/, size_t /*dil*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void conv2d_dw_backward(const float* /*grad_out*/, const float* /*in*/, const float* /*weight*/,
                        float* /*grad_in*/, float* /*grad_weight*/, float* /*grad_bias*/,
                        size_t /*C*/, size_t /*H_in*/, size_t /*W_in*/, size_t /*H_out*/, size_t /*W_out*/,
                        size_t /*K*/, size_t /*pad*/, size_t /*stride*/, size_t /*dil*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void conv2d_forward(const float* /*in*/, const float* /*weight*/, const float* /*bias*/, float* /*out*/,
                    size_t /*C_in*/, size_t /*C_out*/, size_t /*H_in*/, size_t /*W_in*/,
                    size_t /*H_out*/, size_t /*W_out*/, size_t /*K*/, size_t /*pad*/, size_t /*stride*/, size_t /*dil*/, size_t /*groups*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void conv2d_backward(const float* /*grad_out*/, const float* /*in*/, const float* /*weight*/,
                     float* /*grad_in*/, float* /*grad_weight*/, float* /*grad_bias*/,
                     size_t /*C_in*/, size_t /*C_out*/, size_t /*H_in*/, size_t /*W_in*/,
                     size_t /*H_out*/, size_t /*W_out*/, size_t /*K*/, size_t /*pad*/, size_t /*stride*/, size_t /*dil*/, size_t /*groups*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void group_norm_forward(const float* /*in*/, const float* /*gamma*/, const float* /*beta*/, float* /*out*/,
                        float* /*saved_mean*/, float* /*saved_rstd*/,
                        size_t /*num_groups*/, size_t /*C*/, size_t /*HW*/, float /*eps*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void group_norm_backward(const float* /*grad_out*/, const float* /*in*/, const float* /*gamma*/,
                         const float* /*saved_mean*/, const float* /*saved_rstd*/,
                         float* /*grad_in*/, float* /*grad_gamma*/, float* /*grad_beta*/,
                         size_t /*num_groups*/, size_t /*C*/, size_t /*HW*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void sigmoid_forward(const float* /*x*/, float* /*y*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void sigmoid_backward(const float* /*grad_y*/, const float* /*y*/, float* /*grad_x*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void convex_combination_forward(const float* /*g*/, const float* /*a*/, const float* /*b*/, float* /*y*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void convex_combination_backward(const float* /*grad_y*/, const float* /*g*/, const float* /*a*/, const float* /*b*/,
                                float* /*grad_g*/, float* /*grad_a*/, float* /*grad_b*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void upsample_bilinear_forward(const float* /*in*/, float* /*out*/, size_t /*C*/, size_t /*H_in*/, size_t /*W_in*/, size_t /*H_out*/, size_t /*W_out*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void upsample_bilinear_backward(const float* /*grad_out*/, float* /*grad_in*/, size_t /*C*/, size_t /*H_in*/, size_t /*W_in*/, size_t /*H_out*/, size_t /*W_out*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void pixel_shuffle_forward(const float* /*in*/, float* /*out*/, size_t /*C_in*/, size_t /*H_in*/, size_t /*W_in*/, size_t /*r*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void pixel_shuffle_backward(const float* /*grad_out*/, float* /*grad_in*/, size_t /*C_in*/, size_t /*H_in*/, size_t /*W_in*/, size_t /*r*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void max_pool2d_forward(const float* /*in*/, float* /*out*/, int64_t* /*argmax*/, size_t /*C*/, size_t /*H_in*/, size_t /*W_in*/,
                        size_t /*H_out*/, size_t /*W_out*/, size_t /*K*/, size_t /*pad*/, size_t /*stride*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void max_pool2d_backward(const float* /*grad_out*/, const int64_t* /*argmax*/, float* /*grad_in*/,
                         size_t /*total_out*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

float bce_with_logits_forward(const float* /*logits*/, const float* /*targets*/,
                              float /*weight*/, float /*pos_weight*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void bce_with_logits_backward(const float* /*logits*/, const float* /*targets*/, float* /*grad_logits*/,
                             float /*grad_out*/, float /*weight*/, float /*pos_weight*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

float dice_loss_forward(const float* /*logits*/, const float* /*targets*/,
                        float /*weight*/, float /*smooth*/, float& /*out_inter*/, float& /*out_sum_p*/, float& /*out_sum_y*/,
                        size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void dice_loss_backward(const float* /*logits*/, const float* /*targets*/, float* /*grad_logits*/,
                        float /*grad_out*/, float /*weight*/, float /*smooth*/,
                        float /*inter*/, float /*sum_p*/, float /*sum_y*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void dice_bce_loss_backward(const float* /*logits*/, const float* /*targets*/, float* /*grad_logits*/,
                            float /*grad_out*/, float /*w*/, float /*bw*/, float /*dw*/, float /*pos_weight*/, float /*smooth*/,
                            float /*inter*/, float /*sum_p*/, float /*sum_y*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

void adamw_step(float* /*theta*/, const float* /*g*/, float* /*m*/, float* /*v*/,
                float /*lr*/, float /*beta1*/, float /*beta2*/, float /*eps*/, float /*wd*/,
                float /*step_size*/, float /*sqrt_bc2*/, size_t /*n*/, void* /*stream*/) {
    throw DeviceError("CUDA kernel called but soar.cpp was compiled without CUDA compiler (nvcc).");
}

#endif

} // namespace soar::cuda::kernels
