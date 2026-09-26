#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>

namespace soar::cuda::kernels {

bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

__global__ void k_cuda_zero(const int64_t n, float* __restrict__ ptr) {
    CUDA_KERNEL_LOOP(idx, n) {
        ptr[idx] = 0.0f;
    }
}

void cuda_zero(float* ptr, size_t n, void* stream) {
    if (n == 0 || !ptr) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_cuda_zero<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), ptr);
}

__global__ void k_cuda_fill(const int64_t n, float* __restrict__ ptr, float val) {
    CUDA_KERNEL_LOOP(idx, n) {
        ptr[idx] = val;
    }
}

void cuda_fill(float* ptr, float val, size_t n, void* stream) {
    if (n == 0 || !ptr) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_cuda_fill<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), ptr, val);
}

__global__ void k_add_forward(const int64_t n, const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ y) {
    CUDA_KERNEL_LOOP(idx, n) {
        y[idx] = a[idx] + b[idx];
    }
}

void add_forward(const float* a, const float* b, float* y, size_t n, void* stream) {
    if (n == 0 || !a || !b || !y) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_add_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), a, b, y);
}

__global__ void k_add_backward(const int64_t n, const float* __restrict__ grad_y,
                               float* __restrict__ grad_a, float* __restrict__ grad_b) {
    CUDA_KERNEL_LOOP(idx, n) {
        const float gy = grad_y[idx];
        if (grad_a) grad_a[idx] += gy;
        if (grad_b) grad_b[idx] += gy;
    }
}

void add_backward(const float* grad_y, float* grad_a, float* grad_b, size_t n, void* stream) {
    if (n == 0 || !grad_y) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_add_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), grad_y, grad_a, grad_b);
}

__global__ void k_mul_broadcast_forward(const int64_t total, const float* __restrict__ a,
                                       const float* __restrict__ b, float* __restrict__ y,
                                       int64_t HW) {
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t c = idx / HW;
        y[idx] = a[idx] * b[c];
    }
}

void mul_broadcast_forward(const float* a, const float* b, float* y, size_t C, size_t HW, void* stream) {
    int64_t total = static_cast<int64_t>(C * HW);
    if (total == 0 || !a || !b || !y) return;
    int blocks = GET_BLOCKS(total);
    k_mul_broadcast_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        total, a, b, y, static_cast<int64_t>(HW));
}

__global__ void k_mul_broadcast_backward(const int64_t total, const float* __restrict__ grad_y,
                                        const float* __restrict__ a, const float* __restrict__ b,
                                        float* __restrict__ grad_a, float* __restrict__ grad_b,
                                        int64_t HW) {
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t c = idx / HW;
        const float gy = grad_y[idx];
        if (grad_a) grad_a[idx] += gy * b[c];
        if (grad_b) atomicAdd(&grad_b[c], gy * a[idx]);
    }
}

void mul_broadcast_backward(const float* grad_y, const float* a, const float* b,
                           float* grad_a, float* grad_b, size_t C, size_t HW, void* stream) {
    int64_t total = static_cast<int64_t>(C * HW);
    if (total == 0 || !grad_y) return;
    int blocks = GET_BLOCKS(total);
    k_mul_broadcast_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        total, grad_y, a, b, grad_a, grad_b, static_cast<int64_t>(HW));
}

__global__ void k_convex_combination_forward(const int64_t n, const float* __restrict__ g,
                                            const float* __restrict__ a, const float* __restrict__ b,
                                            float* __restrict__ y) {
    CUDA_KERNEL_LOOP(idx, n) {
        float gate = g[idx];
        y[idx] = gate * a[idx] + (1.0f - gate) * b[idx];
    }
}

void convex_combination_forward(const float* g, const float* a, const float* b, float* y, size_t n, void* stream) {
    if (n == 0 || !g || !a || !b || !y) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_convex_combination_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), g, a, b, y);
}

__global__ void k_convex_combination_backward(const int64_t n, const float* __restrict__ grad_y,
                                             const float* __restrict__ g, const float* __restrict__ a,
                                             const float* __restrict__ b, float* __restrict__ grad_g,
                                             float* __restrict__ grad_a, float* __restrict__ grad_b) {
    CUDA_KERNEL_LOOP(idx, n) {
        float gy = grad_y[idx];
        float gate = g[idx];
        float aval = a[idx];
        float bval = b[idx];
        if (grad_g) grad_g[idx] += gy * (aval - bval);
        if (grad_a) grad_a[idx] += gy * gate;
        if (grad_b) grad_b[idx] += gy * (1.0f - gate);
    }
}

void convex_combination_backward(const float* grad_y, const float* g, const float* a, const float* b,
                                float* grad_g, float* grad_a, float* grad_b, size_t n, void* stream) {
    if (n == 0 || !grad_y) return;
    int blocks = GET_BLOCKS(static_cast<int64_t>(n));
    k_convex_combination_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<int64_t>(n), grad_y, g, a, b, grad_g, grad_a, grad_b);
}

} // namespace soar::cuda::kernels
