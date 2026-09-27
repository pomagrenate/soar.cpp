#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cmath>

namespace soar::cuda::kernels {

// ============================================================
//  Multi-tensor fused AdamW optimizer
//  Inspired by PyTorch's MultiTensorApply and fused_adam_utils.cuh
//  Processes all parameter tensors in a single kernel launch
// ============================================================

constexpr int kAdamILP = 4;  // Process 4 elements per thread (vectorized)
constexpr int kAdamChunkSize = 65536;  // Chunk size for multi-tensor processing
constexpr int kAdamBlockSize = 512;
constexpr int kMaxTensorsPerLaunch = 64;  // Conservative limit to stay under 4KB argument limit

/**
 * @brief Metadata for multi-tensor AdamW kernel.
 * 
 * Contains pointers and sizes for up to kMaxTensorsPerLaunch parameter tensors.
 * Designed to fit within CUDA's 4KB kernel argument limit.
 */
struct AdamTensorMeta {
    float* params[kMaxTensorsPerLaunch];
    const float* grads[kMaxTensorsPerLaunch];
    float* m[kMaxTensorsPerLaunch];
    float* v[kMaxTensorsPerLaunch];
    int64_t numel[kMaxTensorsPerLaunch];
    int n_tensors;
    
    // Maps block index to tensor index and chunk offset
    uint8_t block_to_tensor[320];  // Max blocks = 320 (64 tensors * 5 chunks each)
    int32_t block_to_chunk[320];
};

/**
 * @brief Multi-tensor fused AdamW kernel.
 * 
 * Processes all parameter tensors in a single launch, eliminating
 * per-tensor kernel launch overhead. Each block works on a chunk
 * of one tensor, allowing efficient parallelism across all parameters.
 */
__global__ void k_adamw_multi_tensor(
    AdamTensorMeta meta,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float step_size,
    float sqrt_bc2) {
    
    // Determine which tensor and chunk this block processes
    int tensor_idx = meta.block_to_tensor[blockIdx.x];
    int chunk_idx = meta.block_to_chunk[blockIdx.x];
    
    if (tensor_idx >= meta.n_tensors) return;
    
    float* theta = meta.params[tensor_idx];
    const float* g = meta.grads[tensor_idx];
    float* m = meta.m[tensor_idx];
    float* v = meta.v[tensor_idx];
    int64_t n = meta.numel[tensor_idx];
    
    // Compute this block's range in the tensor
    int64_t chunk_start = chunk_idx * kAdamChunkSize;
    int64_t chunk_end = min(chunk_start + kAdamChunkSize, n);
    int64_t chunk_size = chunk_end - chunk_start;
    
    if (chunk_start >= n) return;
    
    // Process elements with ILP (Instruction Level Parallelism)
    int64_t tid = threadIdx.x;
    int64_t stride = blockDim.x * kAdamILP;
    
    for (int64_t base = tid; base < chunk_size; base += stride) {
        #pragma unroll
        for (int k = 0; k < kAdamILP; k++) {
            int64_t idx = chunk_start + base + k * blockDim.x;
            if (idx < chunk_end) {
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
    }
}

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
    SOAR_CUDA_KERNEL_LAUNCH_CHECK_DEBUG();
}

/**
 * @brief Multi-tensor fused AdamW step (host-side dispatch).
 * 
 * Processes multiple parameter tensors in a single kernel launch,
 * reducing launch overhead from ~5-10μs per tensor to a single launch.
 * 
 * @param params Array of parameter tensor pointers
 * @param grads Array of gradient tensor pointers
 * @param m Array of first moment tensor pointers
 * @param v Array of second moment tensor pointers
 * @param numel Array of element counts for each tensor
 * @param n_tensors Number of tensors (must be <= kMaxTensorsPerLaunch)
 * @param lr Learning rate
 * @param beta1 Adam beta1
 * @param beta2 Adam beta2
 * @param eps Adam epsilon
 * @param wd Weight decay
 * @param step_size Bias-corrected step size = lr / (1 - beta1^t)
 * @param sqrt_bc2 sqrt(1 - beta2^t)
 * @param stream CUDA stream
 */
void adamw_step_multi_tensor(
    float** params,
    const float** grads,
    float** m,
    float** v,
    const int64_t* numel,
    int n_tensors,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float wd,
    float step_size,
    float sqrt_bc2,
    void* stream) {
    
    if (n_tensors == 0) return;
    if (n_tensors > kMaxTensorsPerLaunch) {
        // Fall back to per-tensor launches if too many tensors
        for (int i = 0; i < n_tensors; i++) {
            adamw_step(params[i], grads[i], m[i], v[i], lr, beta1, beta2, eps, wd,
                       step_size, sqrt_bc2, static_cast<size_t>(numel[i]), stream);
        }
        return;
    }
    
    // Build metadata
    AdamTensorMeta meta;
    meta.n_tensors = n_tensors;
    
    int block_idx = 0;
    for (int i = 0; i < n_tensors; i++) {
        meta.params[i] = params[i];
        meta.grads[i] = grads[i];
        meta.m[i] = m[i];
        meta.v[i] = v[i];
        meta.numel[i] = numel[i];
        
        // Calculate number of chunks for this tensor
        int64_t n = numel[i];
        int n_chunks = static_cast<int>((n + kAdamChunkSize - 1) / kAdamChunkSize);
        
        for (int c = 0; c < n_chunks; c++) {
            if (block_idx < 320) {
                meta.block_to_tensor[block_idx] = static_cast<uint8_t>(i);
                meta.block_to_chunk[block_idx] = c;
                block_idx++;
            }
        }
    }
    
    // Launch multi-tensor kernel
    k_adamw_multi_tensor<<<block_idx, kAdamBlockSize, 0, static_cast<cudaStream_t>(stream)>>>(
        meta, lr, beta1, beta2, eps, wd, step_size, sqrt_bc2);
    SOAR_CUDA_KERNEL_LAUNCH_CHECK_DEBUG();
}

} // namespace soar::cuda::kernels
