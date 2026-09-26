#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cmath>

namespace soar::cuda::kernels {

/**
 * @brief SiLU forward kernel mirroring PyTorch ATen/native/cuda/ActivationSiluKernel.cu.
 * y = x / (1 + exp(-x))
 */
__global__ void silu_forward_kernel(const int64_t n, const float* __restrict__ x, float* __restrict__ y) {
    CUDA_KERNEL_LOOP(idx, n) {
        const float val = x[idx];
        const float s = 1.0f / (1.0f + __expf(-val));
        y[idx] = val * s;
    }
}

/**
 * @brief SiLU backward kernel mirroring PyTorch:
 * dy * s * (1 + x * (1 - s)) where s = 1 / (1 + exp(-x))
 */
__global__ void silu_backward_kernel(const int64_t n, const float* __restrict__ grad_y,
                                     const float* __restrict__ x, float* __restrict__ grad_x) {
    CUDA_KERNEL_LOOP(idx, n) {
        const float dy = grad_y[idx];
        const float val = x[idx];
        const float s = 1.0f / (1.0f + __expf(-val));
        grad_x[idx] += dy * s * (1.0f + val * (1.0f - s));
    }
}

void silu_forward(const float* x, float* y, size_t n, void* stream) {
    if (n == 0 || !x || !y) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    silu_forward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), x, y);
}

void silu_backward(const float* grad_y, const float* x, float* grad_x, size_t n, void* stream) {
    if (n == 0 || !grad_y || !x || !grad_x) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    silu_backward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), grad_y, x, grad_x);
}

/**
 * @brief Sigmoid forward kernel: y = 1 / (1 + exp(-x))
 */
__global__ void sigmoid_forward_kernel(const int64_t n, const float* __restrict__ x, float* __restrict__ y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = 1.0f / (1.0f + __expf(-x[idx]));
    }
}

/**
 * @brief Sigmoid backward kernel: dy * y * (1 - y)
 */
__global__ void sigmoid_backward_kernel(const int64_t n, const float* __restrict__ grad_y,
                                       const float* __restrict__ y, float* __restrict__ grad_x) {
    CUDA_KERNEL_LOOP(idx, n) {
        const float out = y[idx];
        grad_x[idx] += grad_y[idx] * out * (1.0f - out);
    }
}

void sigmoid_forward(const float* x, float* y, size_t n, void* stream) {
    if (n == 0 || !x || !y) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    sigmoid_forward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), x, y);
}

void sigmoid_backward(const float* grad_y, const float* y, float* grad_x, size_t n, void* stream) {
    if (n == 0 || !grad_y || !y || !grad_x) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    sigmoid_backward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), grad_y, y, grad_x);
}

} // namespace soar::cuda::kernels
