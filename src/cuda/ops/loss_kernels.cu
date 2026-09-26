#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cmath>

namespace soar::cuda::kernels {

// =============================================================================
// BCE with Logits Forward & Backward
// =============================================================================

__global__ void k_bce_with_logits_forward(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ out_loss,
    float pos_weight,
    int64_t n) {
    float sum = 0.0f;
    CUDA_KERNEL_LOOP(idx, n) {
        float z = logits[idx];
        float y = targets[idx];
        float w = 1.0f + (pos_weight - 1.0f) * y;
        float term = fmaxf(-z, 0.0f) + log1pf(expf(-fabsf(z)));
        float loss_i = (1.0f - y) * z + w * term;
        sum += loss_i;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_loss, sum);
    }
}

float bce_with_logits_forward(const float* logits, const float* targets,
                              float weight, float pos_weight, size_t n, void* stream) {
    if (n == 0 || !logits || !targets) return 0.0f;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    float* d_loss = nullptr;
    cudaMallocAsync(&d_loss, sizeof(float), s);
    cudaMemsetAsync(d_loss, 0, sizeof(float), s);

    int blocks = GET_BLOCKS(static_cast<int64_t>(n), 256);
    k_bce_with_logits_forward<<<blocks, 256, 0, s>>>(
        logits, targets, d_loss, pos_weight, static_cast<int64_t>(n));

    float h_loss = 0.0f;
    cudaMemcpyAsync(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaFreeAsync(d_loss, s);
    cudaStreamSynchronize(s);

    return static_cast<float>(weight * (static_cast<double>(h_loss) / static_cast<double>(n)));
}

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

// =============================================================================
// Dice Loss Forward & Backward
// =============================================================================

__global__ void k_dice_loss_forward(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ out_accum, // [3]: inter, sum_p, sum_y
    int64_t n) {
    float thread_inter = 0.0f;
    float thread_sum_p = 0.0f;
    float thread_sum_y = 0.0f;

    CUDA_KERNEL_LOOP(idx, n) {
        float z = logits[idx];
        float y = targets[idx];
        float p = 1.0f / (1.0f + __expf(-z));
        thread_inter += p * y;
        thread_sum_p += p;
        thread_sum_y += y;
    }

    float b_inter = block_reduce_sum(thread_inter);
    float b_sum_p = block_reduce_sum(thread_sum_p);
    float b_sum_y = block_reduce_sum(thread_sum_y);

    if (threadIdx.x == 0) {
        atomicAdd(&out_accum[0], b_inter);
        atomicAdd(&out_accum[1], b_sum_p);
        atomicAdd(&out_accum[2], b_sum_y);
    }
}

float dice_loss_forward(const float* logits, const float* targets,
                        float weight, float smooth, float& out_inter, float& out_sum_p, float& out_sum_y,
                        size_t n, void* stream) {
    if (n == 0 || !logits || !targets) {
        out_inter = out_sum_p = out_sum_y = 0.0f;
        return 0.0f;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    float* d_accum = nullptr;
    cudaMallocAsync(&d_accum, 3 * sizeof(float), s);
    cudaMemsetAsync(d_accum, 0, 3 * sizeof(float), s);

    int blocks = GET_BLOCKS(static_cast<int64_t>(n), 256);
    k_dice_loss_forward<<<blocks, 256, 0, s>>>(
        logits, targets, d_accum, static_cast<int64_t>(n));

    float h_accum[3] = {0.0f, 0.0f, 0.0f};
    cudaMemcpyAsync(h_accum, d_accum, 3 * sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaFreeAsync(d_accum, s);
    cudaStreamSynchronize(s);

    out_inter = h_accum[0];
    out_sum_p = h_accum[1];
    out_sum_y = h_accum[2];

    double denom = static_cast<double>(out_sum_p) + static_cast<double>(out_sum_y) + static_cast<double>(smooth);
    double numer = 2.0 * static_cast<double>(out_inter) + static_cast<double>(smooth);
    double dice = numer / (denom + 1e-7);
    return static_cast<float>(weight * (1.0 - dice));
}

__global__ void k_dice_loss_backward(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ grad_logits,
    float factor,
    float denom,
    float numer,
    int64_t n) {
    float denom_sq = denom * denom;
    CUDA_KERNEL_LOOP(idx, n) {
        float z = logits[idx];
        float y = targets[idx];
        float p = 1.0f / (1.0f + __expf(-z));
        float dp_dz = p * (1.0f - p);
        float d_dice_dp = (2.0f * y * denom - numer) / denom_sq;
        float d_loss_dz = -d_dice_dp * dp_dz;
        grad_logits[idx] += factor * d_loss_dz;
    }
}

void dice_loss_backward(const float* logits, const float* targets, float* grad_logits,
                        float grad_out, float weight, float smooth,
                        float inter, float sum_p, float sum_y, size_t n, void* stream) {
    if (n == 0 || !logits || !targets || !grad_logits) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    float denom = sum_p + sum_y + smooth;
    float numer = 2.0f * inter + smooth;
    float factor = grad_out * weight;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_dice_loss_backward<<<blocks, CUDA_NUM_THREADS, 0, s>>>(
        logits, targets, grad_logits, factor, denom, numer, static_cast<int64_t>(n));
}

// =============================================================================
// Fused Dice + BCE Loss Backward
// =============================================================================

__global__ void k_dice_bce_loss_backward(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ grad_logits,
    float go,
    float w,
    float bw,
    float dw,
    float pos_weight,
    float smooth,
    float inter,
    float sum_p,
    float sum_y,
    int64_t n) {
    float scale_bce = (bw > 0.0f) ? ((go * w * bw) / static_cast<float>(n)) : 0.0f;
    float scale_dice = (dw > 0.0f) ? (go * w * dw) : 0.0f;
    float denom = sum_p + sum_y + smooth;
    float denom_sq = denom * denom;
    float numer = 2.0f * inter + smooth;

    CUDA_KERNEL_LOOP(idx, n) {
        float z = logits[idx];
        float y = targets[idx];
        float sig = 1.0f / (1.0f + __expf(-z));
        float g = 0.0f;

        if (bw > 0.0f) {
            float w_pos = 1.0f + (pos_weight - 1.0f) * y;
            g += (sig * w_pos - pos_weight * y) * scale_bce;
        }

        if (dw > 0.0f) {
            float dp_dz = sig * (1.0f - sig);
            float d_dice_dp = (2.0f * y * denom - numer) / denom_sq;
            float d_loss_dz = -d_dice_dp * dp_dz;
            g += scale_dice * d_loss_dz;
        }

        grad_logits[idx] += g;
    }
}

void dice_bce_loss_backward(const float* logits, const float* targets, float* grad_logits,
                            float grad_out, float w, float bw, float dw, float pos_weight, float smooth,
                            float inter, float sum_p, float sum_y, size_t n, void* stream) {
    if (n == 0 || !logits || !targets || !grad_logits) return;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_dice_bce_loss_backward<<<blocks, CUDA_NUM_THREADS, 0, s>>>(
        logits, targets, grad_logits, grad_out, w, bw, dw, pos_weight, smooth,
        inter, sum_p, sum_y, static_cast<int64_t>(n));
}

} // namespace soar::cuda::kernels
