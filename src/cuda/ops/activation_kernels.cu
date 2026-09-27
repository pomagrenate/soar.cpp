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
 * @brief Vectorized SiLU forward kernel (float4) for 2-4× bandwidth improvement.
 * Processes 4 elements per thread using aligned vector loads.
 */
__global__ void silu_forward_kernel_vec4(const int64_t n, const float* __restrict__ x, float* __restrict__ y) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    // Vectorized path: process 4 floats at once
    const int64_t n_vec = n / 4;
    const auto* x_vec = reinterpret_cast<const VecLoad<float,4>*>(x);
    auto* y_vec = reinterpret_cast<VecLoad<float,4>*>(y);
    
    for (int64_t i = tid; i < n_vec; i += stride) {
        VecLoad<float,4> v = x_vec[i];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float val = v.val[k];
            float s = 1.0f / (1.0f + __expf(-val));
            v.val[k] = val * s;
        }
        y_vec[i] = v;
    }
    
    // Scalar tail for elements not divisible by 4
    for (int64_t i = n_vec * 4 + tid; i < n; i += stride) {
        const float val = x[i];
        const float s = 1.0f / (1.0f + __expf(-val));
        y[i] = val * s;
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

/**
 * @brief Vectorized SiLU backward kernel (float4).
 */
__global__ void silu_backward_kernel_vec4(const int64_t n, const float* __restrict__ grad_y,
                                         const float* __restrict__ x, float* __restrict__ grad_x) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    const int64_t n_vec = n / 4;
    const auto* grad_y_vec = reinterpret_cast<const VecLoad<float,4>*>(grad_y);
    const auto* x_vec = reinterpret_cast<const VecLoad<float,4>*>(x);
    auto* grad_x_vec = reinterpret_cast<VecLoad<float,4>*>(grad_x);
    
    for (int64_t i = tid; i < n_vec; i += stride) {
        VecLoad<float,4> dy = grad_y_vec[i];
        VecLoad<float,4> val = x_vec[i];
        VecLoad<float,4> gx;
        
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float s = 1.0f / (1.0f + __expf(-val.val[k]));
            gx.val[k] = dy.val[k] * s * (1.0f + val.val[k] * (1.0f - s));
        }
        grad_x_vec[i] = gx;
    }
    
    for (int64_t i = n_vec * 4 + tid; i < n; i += stride) {
        const float dy = grad_y[i];
        const float val = x[i];
        const float s = 1.0f / (1.0f + __expf(-val));
        grad_x[i] += dy * s * (1.0f + val * (1.0f - s));
    }
}

void silu_forward(const float* x, float* y, size_t n, void* stream) {
    if (n == 0 || !x || !y) return;
    
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    
    // Use vectorized kernel for large aligned buffers
    if (n >= 1024 && is_vec_aligned<float,4>(x) && is_vec_aligned<float,4>(y)) {
        silu_forward_kernel_vec4<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<int64_t>(n), x, y);
    } else {
        silu_forward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<int64_t>(n), x, y);
    }
    
    SOAR_CUDA_KERNEL_LAUNCH_CHECK_DEBUG();
}

void silu_backward(const float* grad_y, const float* x, float* grad_x, size_t n, void* stream) {
    if (n == 0 || !grad_y || !x || !grad_x) return;
    
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    
    if (n >= 1024 && is_vec_aligned<float,4>(grad_y) && is_vec_aligned<float,4>(x) && is_vec_aligned<float,4>(grad_x)) {
        silu_backward_kernel_vec4<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<int64_t>(n), grad_y, x, grad_x);
    } else {
        silu_backward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<int64_t>(n), grad_y, x, grad_x);
    }
    
    SOAR_CUDA_KERNEL_LAUNCH_CHECK_DEBUG();
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
 * @brief Vectorized Sigmoid forward kernel (float4).
 */
__global__ void sigmoid_forward_kernel_vec4(const int64_t n, const float* __restrict__ x, float* __restrict__ y) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    
    const int64_t n_vec = n / 4;
    const auto* x_vec = reinterpret_cast<const VecLoad<float,4>*>(x);
    auto* y_vec = reinterpret_cast<VecLoad<float,4>*>(y);
    
    for (int64_t i = tid; i < n_vec; i += stride) {
        VecLoad<float,4> v = x_vec[i];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            v.val[k] = 1.0f / (1.0f + __expf(-v.val[k]));
        }
        y_vec[i] = v;
    }
    
    for (int64_t i = n_vec * 4 + tid; i < n; i += stride) {
        y[i] = 1.0f / (1.0f + __expf(-x[i]));
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
    
    if (n >= 1024 && is_vec_aligned<float,4>(x) && is_vec_aligned<float,4>(y)) {
        sigmoid_forward_kernel_vec4<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<int64_t>(n), x, y);
    } else {
        sigmoid_forward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<int64_t>(n), x, y);
    }
    
    SOAR_CUDA_KERNEL_LAUNCH_CHECK_DEBUG();
}

void sigmoid_backward(const float* grad_y, const float* y, float* grad_x, size_t n, void* stream) {
    if (n == 0 || !grad_y || !y || !grad_x) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    sigmoid_backward_kernel<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), grad_y, y, grad_x);
}

} // namespace soar::cuda::kernels
