#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cmath>

namespace soar::cuda::kernels {

/**
 * @brief Fused BCE with logits backward kernel mirroring PyTorch at::native::binary_cross_entropy_with_logits_backward.
 *
 * grad_logit = grad_out * (sigmoid(logit) * (1 + (pos_weight - 1) * target) - target * pos_weight) * weight / n
 * For pos_weight = 1.0, this simplifies to: grad_out * (sigmoid(logit) - target) * weight / n.
 */
__global__ void k_bce_with_logits_backward(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ grad_logits,
    float grad_out,
    float weight,
    float pos_weight,
    int64_t n) {
    float inv_n = 1.0f / static_cast<float>(n);
    CUDA_KERNEL_LOOP(idx, n) {
        float z = logits[idx];
        float y = targets[idx];
        float sig = 1.0f / (1.0f + __expf(-z));
        
        float grad;
        if (pos_weight != 1.0f) {
            grad = sig * (1.0f + (pos_weight - 1.0f) * y) - y * pos_weight;
        } else {
            grad = sig - y;
        }
        
        grad_logits[idx] += grad_out * grad * weight * inv_n;
    }
}

void bce_with_logits_backward(const float* logits, const float* targets, float* grad_logits,
                             float grad_out, float weight, float pos_weight, size_t n, void* stream) {
    if (n == 0 || !logits || !targets || !grad_logits) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_bce_with_logits_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, targets, grad_logits, grad_out, weight, pos_weight, static_cast<int64_t>(n));
}

} // namespace soar::cuda::kernels
