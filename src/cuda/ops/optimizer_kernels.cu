#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cmath>

namespace soar::cuda::kernels {

/**
 * @brief Fused AdamW parameter step kernel mirroring PyTorch fused AdamW.
 *
 * m_t = beta1 * m_{t-1} + (1 - beta1) * g
 * v_t = beta2 * v_{t-1} + (1 - beta2) * g^2
 * step_size = lr / (1 - beta1^t)
 * denom = sqrt(v_t) / sqrt(1 - beta2^t) + eps
 * theta_t = theta_{t-1} - lr * wd * theta_{t-1} - step_size * (m_t / denom)
 */
__global__ void k_adamw_step(
    float* __restrict__ theta,
    const float* __restrict__ g,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float step_size,
    float sqrt_bc2,
    int64_t n) {
    CUDA_KERNEL_LOOP(idx, n) {
        float p = theta[idx];
        float grad = g[idx];

        // Decoupled weight decay
        if (wd != 0.0f) {
            p -= lr * wd * p;
        }

        // Biased first and second moment updates
        float m_val = beta1 * m[idx] + (1.0f - beta1) * grad;
        float v_val = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        m[idx] = m_val;
        v[idx] = v_val;

        // Bias-corrected denominator
        float denom = (sqrtf(v_val) / sqrt_bc2) + eps;
        p -= step_size * (m_val / denom);

        theta[idx] = p;
    }
}

void adamw_step(float* theta, const float* g, float* m, float* v,
                float lr, float beta1, float beta2, float eps, float wd,
                float step_size, float sqrt_bc2, size_t n, void* stream) {
    if (n == 0 || !theta || !g || !m || !v) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_adamw_step<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        theta, g, m, v, lr, beta1, beta2, eps, wd, step_size, sqrt_bc2, static_cast<int64_t>(n));
}

} // namespace soar::cuda::kernels
