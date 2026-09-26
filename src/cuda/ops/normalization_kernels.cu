#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cmath>

namespace soar::cuda::kernels {

/**
 * @brief GroupNorm moments kernel mirroring PyTorch RowwiseMomentsCUDAKernel.
 * Computes mean and rstd (1.0 / sqrt(var + eps)) per group using shared memory and warp reduction.
 */
__global__ void k_group_norm_moments(
    const float* __restrict__ in,
    float* __restrict__ mean,
    float* __restrict__ rstd,
    int64_t num_groups,
    int64_t C,
    int64_t HW,
    float eps) {
    int64_t g = blockIdx.x; // One block per group
    if (g >= num_groups) return;

    int64_t channels_per_group = C / num_groups;
    int64_t M = channels_per_group * HW; // Total elements in this group

    __shared__ float s_sum[32];
    __shared__ float s_sq_sum[32];

    float thread_sum = 0.0f;
    float thread_sq_sum = 0.0f;

    for (int64_t idx = threadIdx.x; idx < M; idx += blockDim.x) {
        int64_t c_offset = idx / HW;
        int64_t hw = idx % HW;
        int64_t c = g * channels_per_group + c_offset;
        float val = in[c * HW + hw];
        thread_sum += val;
        thread_sq_sum += val * val;
    }

    // Warp reduce
    thread_sum = warp_reduce_sum(thread_sum);
    thread_sq_sum = warp_reduce_sum(thread_sq_sum);

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        s_sum[wid] = thread_sum;
        s_sq_sum[wid] = thread_sq_sum;
    }
    __syncthreads();

    // Block reduce across warps
    if (wid == 0) {
        float sum = (lane < blockDim.x / 32) ? s_sum[lane] : 0.0f;
        float sq_sum = (lane < blockDim.x / 32) ? s_sq_sum[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        sq_sum = warp_reduce_sum(sq_sum);

        if (lane == 0) {
            float m = sum / static_cast<float>(M);
            float var = (sq_sum / static_cast<float>(M)) - (m * m);
            if (var < 0.0f) var = 0.0f;
            mean[g] = m;
            rstd[g] = 1.0f / sqrtf(var + eps);
        }
    }
}

/**
 * @brief GroupNorm forward normalization kernel:
 * y = (x - mean) * rstd * gamma + beta
 */
__global__ void k_group_norm_forward(
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ out,
    int64_t total,
    int64_t channels_per_group,
    int64_t HW) {
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t c = idx / HW;
        int64_t g = c / channels_per_group;
        float x = in[idx];
        float m = mean[g];
        float r = rstd[g];
        float norm = (x - m) * r;
        float g_val = gamma ? gamma[c] : 1.0f;
        float b_val = beta ? beta[c] : 0.0f;
        out[idx] = norm * g_val + b_val;
    }
}

void group_norm_forward(const float* in, const float* gamma, const float* beta, float* out,
                        float* saved_mean, float* saved_rstd,
                        size_t num_groups, size_t C, size_t HW, float eps, void* stream) {
    if (C == 0 || HW == 0 || !in || !out) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    
    // Launch moments kernel: 1 block of 256 threads per group
    k_group_norm_moments<<<static_cast<unsigned int>(num_groups), 256, 0, s>>>(
        in, saved_mean, saved_rstd, num_groups, C, HW, eps);

    // Launch elementwise normalization
    int64_t total = static_cast<int64_t>(C * HW);
    int blocks = GET_BLOCKS(total);
    int64_t cpg = static_cast<int64_t>(C / num_groups);
    k_group_norm_forward<<<blocks, CUDA_NUM_THREADS, 0, s>>>(
        in, gamma, beta, saved_mean, saved_rstd, out, total, cpg, static_cast<int64_t>(HW));
}

/**
 * @brief GroupNorm backward reduction for dgamma and dbeta
 */
__global__ void k_group_norm_backward_params(
    const float* __restrict__ grad_out,
    const float* __restrict__ in,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    int64_t C,
    int64_t channels_per_group,
    int64_t HW) {
    int64_t c = blockIdx.x;
    if (c >= C) return;

    int64_t g = c / channels_per_group;
    float m = mean[g];
    float r = rstd[g];

    const float* go = grad_out + c * HW;
    const float* inp = in + c * HW;

    float sum_gamma = 0.0f;
    float sum_beta = 0.0f;

    for (int64_t hw = threadIdx.x; hw < HW; hw += blockDim.x) {
        float dy = go[hw];
        float x = inp[hw];
        float norm = (x - m) * r;
        if (grad_gamma) sum_gamma += dy * norm;
        if (grad_beta) sum_beta += dy;
    }

    if (grad_gamma) sum_gamma = block_reduce_sum(sum_gamma);
    if (grad_beta) sum_beta = block_reduce_sum(sum_beta);

    if (threadIdx.x == 0) {
        if (grad_gamma) grad_gamma[c] += sum_gamma;
        if (grad_beta) grad_beta[c] += sum_beta;
    }
}

/**
 * @brief GroupNorm backward moments kernel to calculate group sums of dy and dy*(x - mean)
 */
__global__ void k_group_norm_backward_moments(
    const float* __restrict__ grad_out,
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    float* __restrict__ sum_dy_gamma,
    float* __restrict__ sum_dy_gamma_diff,
    int64_t num_groups,
    int64_t C,
    int64_t HW) {
    int64_t g = blockIdx.x;
    if (g >= num_groups) return;

    int64_t channels_per_group = C / num_groups;
    int64_t M = channels_per_group * HW;

    __shared__ float s_dy[32];
    __shared__ float s_diff[32];

    float thread_dy = 0.0f;
    float thread_diff = 0.0f;

    for (int64_t idx = threadIdx.x; idx < M; idx += blockDim.x) {
        int64_t c_offset = idx / HW;
        int64_t hw = idx % HW;
        int64_t c = g * channels_per_group + c_offset;
        float dy = grad_out[c * HW + hw];
        float g_val = gamma ? gamma[c] : 1.0f;
        float diff = in[c * HW + hw] - mean[g];
        thread_dy += dy * g_val;
        thread_diff += dy * g_val * diff;
    }

    thread_dy = warp_reduce_sum(thread_dy);
    thread_diff = warp_reduce_sum(thread_diff);

    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) {
        s_dy[wid] = thread_dy;
        s_diff[wid] = thread_diff;
    }
    __syncthreads();

    if (wid == 0) {
        float sum_dy = (lane < blockDim.x / 32) ? s_dy[lane] : 0.0f;
        float sum_diff = (lane < blockDim.x / 32) ? s_diff[lane] : 0.0f;
        sum_dy = warp_reduce_sum(sum_dy);
        sum_diff = warp_reduce_sum(sum_diff);
        if (lane == 0) {
            sum_dy_gamma[g] = sum_dy;
            sum_dy_gamma_diff[g] = sum_diff;
        }
    }
}

/**
 * @brief GroupNorm backward input gradient kernel using exact PyTorch formulation:
 * dx = rstd * [ dy * gamma - 1/M * (sum_dy_gamma + diff * rstd^2 * sum_dy_gamma_diff) ]
 */
__global__ void k_group_norm_backward_input(
    const float* __restrict__ grad_out,
    const float* __restrict__ in,
    const float* __restrict__ gamma,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    const float* __restrict__ sum_dy_gamma,
    const float* __restrict__ sum_dy_gamma_diff,
    float* __restrict__ grad_in,
    int64_t total,
    int64_t channels_per_group,
    int64_t HW) {
    int64_t M = channels_per_group * HW;
    float inv_M = 1.0f / static_cast<float>(M);

    CUDA_KERNEL_LOOP(idx, total) {
        int64_t c = idx / HW;
        int64_t g = c / channels_per_group;
        float dy = grad_out[idx];
        float g_val = gamma ? gamma[c] : 1.0f;
        float x = in[idx];
        float diff = x - mean[g];
        float r = rstd[g];

        float sum_dy = sum_dy_gamma[g];
        float sum_diff = sum_dy_gamma_diff[g];

        float dx = r * (dy * g_val - inv_M * (sum_dy + diff * (r * r) * sum_diff));
        grad_in[idx] += dx;
    }
}

void group_norm_backward(const float* grad_out, const float* in, const float* gamma,
                         const float* saved_mean, const float* saved_rstd,
                         float* grad_in, float* grad_gamma, float* grad_beta,
                         size_t num_groups, size_t C, size_t HW, void* stream) {
    if (C == 0 || HW == 0 || !grad_out) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int64_t total = static_cast<int64_t>(C * HW);
    int64_t cpg = static_cast<int64_t>(C / num_groups);

    // 1. Accumulate parameter gradients
    if (grad_gamma || grad_beta) {
        k_group_norm_backward_params<<<static_cast<unsigned int>(C), 256, 0, s>>>(
            grad_out, in, saved_mean, saved_rstd, grad_gamma, grad_beta, static_cast<int64_t>(C), cpg, static_cast<int64_t>(HW));
    }

    // 2. Compute input gradients if needed
    if (grad_in) {
        // Allocate temporary scratch for group reductions
        float* sum_dy_gamma = nullptr;
        float* sum_dy_gamma_diff = nullptr;
        cudaMallocAsync(&sum_dy_gamma, num_groups * sizeof(float), s);
        cudaMallocAsync(&sum_dy_gamma_diff, num_groups * sizeof(float), s);

        k_group_norm_backward_moments<<<static_cast<unsigned int>(num_groups), 256, 0, s>>>(
            grad_out, in, gamma, saved_mean, sum_dy_gamma, sum_dy_gamma_diff,
            num_groups, C, HW);

        int blocks = GET_BLOCKS(total);
        k_group_norm_backward_input<<<blocks, CUDA_NUM_THREADS, 0, s>>>(
            grad_out, in, gamma, saved_mean, saved_rstd, sum_dy_gamma, sum_dy_gamma_diff,
            grad_in, total, cpg, static_cast<int64_t>(HW));

        cudaFreeAsync(sum_dy_gamma, s);
        cudaFreeAsync(sum_dy_gamma_diff, s);
    }
}

} // namespace soar::cuda::kernels
