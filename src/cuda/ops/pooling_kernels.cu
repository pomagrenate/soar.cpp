#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cfloat>

namespace soar::cuda::kernels {

__global__ void k_global_avg_pool_forward(const float* __restrict__ in, float* __restrict__ out,
                                         int64_t C, int64_t HW) {
    int64_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        const float* in_c = in + c * HW;
        float sum = 0.0f;
        for (int64_t i = 0; i < HW; ++i) {
            sum += in_c[i];
        }
        out[c] = sum / static_cast<float>(HW);
    }
}

void global_avg_pool_forward(const float* in, float* out, size_t C, size_t HW, void* stream) {
    if (C == 0 || HW == 0 || !in || !out) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(C), 256);
    k_global_avg_pool_forward<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        in, out, static_cast<int64_t>(C), static_cast<int64_t>(HW));
}

__global__ void k_global_avg_pool_backward(const float* __restrict__ grad_out, float* __restrict__ grad_in,
                                          int64_t total, int64_t HW) {
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t c = idx / HW;
        grad_in[idx] += grad_out[c] / static_cast<float>(HW);
    }
}

void global_avg_pool_backward(const float* grad_out, float* grad_in, size_t C, size_t HW, void* stream) {
    int64_t total = static_cast<int64_t>(C * HW);
    if (total == 0 || !grad_out || !grad_in) return;
    int blocks = GET_BLOCKS(total);
    k_global_avg_pool_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        grad_out, grad_in, total, static_cast<int64_t>(HW));
}

__global__ void k_max_pool2d_forward(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t* __restrict__ argmax,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t K,
    int64_t pad,
    int64_t stride) {
    int64_t total = C * H_out * W_out;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t wo = idx % W_out;
        int64_t temp = idx / W_out;
        int64_t ho = temp % H_out;
        int64_t c = temp / H_out;

        const float* in_c = in + c * (H_in * W_in);

        int64_t hstart = ho * stride - pad;
        int64_t wstart = wo * stride - pad;
        int64_t hend = min(hstart + K, H_in);
        int64_t wend = min(wstart + K, W_in);
        hstart = max(hstart, (int64_t)0);
        wstart = max(wstart, (int64_t)0);

        float max_val = -FLT_MAX;
        int64_t max_idx = -1;

        for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
                int64_t in_pos = h * W_in + w;
                float val = in_c[in_pos];
                if (val > max_val) {
                    max_val = val;
                    max_idx = in_pos;
                }
            }
        }
        out[idx] = max_val;
        if (argmax) argmax[idx] = max_idx;
    }
}

void max_pool2d_forward(const float* in, float* out, int64_t* argmax, size_t C, size_t H_in, size_t W_in,
                        size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, void* stream) {
    int64_t total = static_cast<int64_t>(C * H_out * W_out);
    if (total == 0 || !in || !out) return;
    int blocks = GET_BLOCKS(total);
    k_max_pool2d_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        in, out, argmax,
        static_cast<int64_t>(C), static_cast<int64_t>(H_in), static_cast<int64_t>(W_in),
        static_cast<int64_t>(H_out), static_cast<int64_t>(W_out),
        static_cast<int64_t>(K), static_cast<int64_t>(pad), static_cast<int64_t>(stride));
}

__global__ void k_max_pool2d_backward(
    const float* __restrict__ grad_out,
    const int64_t* __restrict__ argmax,
    float* __restrict__ grad_in,
    int64_t total_out,
    int64_t spatial_out,
    int64_t spatial_in) {
    CUDA_KERNEL_LOOP(idx, total_out) {
        int64_t c = idx / spatial_out;
        int64_t in_pos = argmax[idx];
        if (in_pos >= 0) {
            atomicAdd(&grad_in[c * spatial_in + in_pos], grad_out[idx]);
        }
    }
}

void max_pool2d_backward(const float* grad_out, const int64_t* argmax, float* grad_in,
                         size_t total_out, void* stream) {
    if (total_out == 0 || !grad_out || !argmax || !grad_in) return;
    // Assuming 2D pooling; spatial out and in are handled accordingly
    int blocks = GET_BLOCKS(static_cast<int64_t>(total_out));
    k_max_pool2d_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        grad_out, argmax, grad_in, static_cast<int64_t>(total_out), 1, 1);
}

} // namespace soar::cuda::kernels
