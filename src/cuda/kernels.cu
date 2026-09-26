#include <soar/cuda/cuda_kernels.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>

namespace soar::cuda::kernels {

bool is_cuda_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
}

// -----------------------------------------------------------------------------
// In-place Memset / Fill Kernels
// -----------------------------------------------------------------------------
__global__ void k_cuda_zero(float* ptr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = 0.0f;
}

void cuda_zero(float* ptr, size_t n, void* stream) {
    if (n == 0 || !ptr) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_cuda_zero<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(ptr, n);
}

__global__ void k_cuda_fill(float* ptr, float val, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) ptr[idx] = val;
}

void cuda_fill(float* ptr, float val, size_t n, void* stream) {
    if (n == 0 || !ptr) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_cuda_fill<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(ptr, val, n);
}

// -----------------------------------------------------------------------------
// Elementwise Add Kernels
// -----------------------------------------------------------------------------
__global__ void k_add_forward(const float* a, const float* b, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) y[idx] = a[idx] + b[idx];
}

void add_forward(const float* a, const float* b, float* y, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_add_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(a, b, y, n);
}

__global__ void k_add_backward(const float* grad_y, float* grad_a, float* grad_b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grad_y[idx];
        if (grad_a) grad_a[idx] += g;
        if (grad_b) grad_b[idx] += g;
    }
}

void add_backward(const float* grad_y, float* grad_a, float* grad_b, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_add_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_y, grad_a, grad_b, n);
}

// -----------------------------------------------------------------------------
// SiLU Activation Kernels: y = x * sigmoid(x) = x / (1 + exp(-x))
// -----------------------------------------------------------------------------
__global__ void k_silu_forward(const float* x, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + __expf(-val));
        y[idx] = val * sig;
    }
}

void silu_forward(const float* x, float* y, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_silu_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(x, y, n);
}

__global__ void k_silu_backward(const float* grad_y, const float* x, float* grad_x, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + __expf(-val));
        float d_silu = sig * (1.0f + val * (1.0f - sig));
        grad_x[idx] += grad_y[idx] * d_silu;
    }
}

void silu_backward(const float* grad_y, const float* x, float* grad_x, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_silu_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_y, x, grad_x, n);
}

// -----------------------------------------------------------------------------
// Sigmoid Activation Kernels: y = 1 / (1 + exp(-x))
// -----------------------------------------------------------------------------
__global__ void k_sigmoid_forward(const float* x, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = 1.0f / (1.0f + __expf(-x[idx]));
    }
}

void sigmoid_forward(const float* x, float* y, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_sigmoid_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(x, y, n);
}

__global__ void k_sigmoid_backward(const float* grad_y, const float* y, float* grad_x, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = y[idx];
        grad_x[idx] += grad_y[idx] * val * (1.0f - val);
    }
}

void sigmoid_backward(const float* grad_y, const float* y, float* grad_x, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_sigmoid_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_y, y, grad_x, n);
}

// -----------------------------------------------------------------------------
// MulBroadcast Kernels (SE Channel Attention: y[c, sp] = a[c, sp] * b[c])
// -----------------------------------------------------------------------------
__global__ void k_mul_broadcast_forward(const float* a, const float* b, float* y, size_t C, size_t HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * HW;
    if (idx < total) {
        size_t c = idx / HW;
        y[idx] = a[idx] * b[c];
    }
}

void mul_broadcast_forward(const float* a, const float* b, float* y, size_t C, size_t HW, void* stream) {
    size_t total = C * HW;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_mul_broadcast_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(a, b, y, C, HW);
}

__global__ void k_mul_broadcast_backward(const float* grad_y, const float* a, const float* b,
                                        float* grad_a, float* grad_b, size_t C, size_t HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * HW;
    if (idx < total) {
        size_t c = idx / HW;
        float gy = grad_y[idx];
        if (grad_a) grad_a[idx] += gy * b[c];
        if (grad_b) atomicAdd(&grad_b[c], gy * a[idx]);
    }
}

void mul_broadcast_backward(const float* grad_y, const float* a, const float* b,
                           float* grad_a, float* grad_b, size_t C, size_t HW, void* stream) {
    size_t total = C * HW;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_mul_broadcast_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_y, a, b, grad_a, grad_b, C, HW);
}

// -----------------------------------------------------------------------------
// Convex Combination Kernels: y = g * a + (1 - g) * b
// -----------------------------------------------------------------------------
__global__ void k_convex_combination_forward(const float* g, const float* a, const float* b, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g_val = g[idx];
        y[idx] = g_val * a[idx] + (1.0f - g_val) * b[idx];
    }
}

void convex_combination_forward(const float* g, const float* a, const float* b, float* y, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_convex_combination_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(g, a, b, y, n);
}

__global__ void k_convex_combination_backward(const float* grad_y, const float* g, const float* a, const float* b,
                                             float* grad_g, float* grad_a, float* grad_b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gy = grad_y[idx];
        float g_val = g[idx];
        if (grad_a) grad_a[idx] += gy * g_val;
        if (grad_b) grad_b[idx] += gy * (1.0f - g_val);
        if (grad_g) grad_g[idx] += gy * (a[idx] - b[idx]);
    }
}

void convex_combination_backward(const float* grad_y, const float* g, const float* a, const float* b,
                                float* grad_g, float* grad_a, float* grad_b, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_convex_combination_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_y, g, a, b, grad_g, grad_a, grad_b, n);
}

// -----------------------------------------------------------------------------
// Global Average Pooling 2D Kernels: [C, H, W] -> [C, 1, 1]
// -----------------------------------------------------------------------------
__global__ void k_global_avg_pool_forward(const float* in, float* out, size_t C, size_t HW) {
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) {
        const float* p_in = in + c * HW;
        float sum = 0.0f;
        for (size_t i = 0; i < HW; ++i) {
            sum += p_in[i];
        }
        out[c] = sum / static_cast<float>(HW);
    }
}

void global_avg_pool_forward(const float* in, float* out, size_t C, size_t HW, void* stream) {
    if (C == 0 || HW == 0) return;
    size_t block = 128;
    size_t grid = (C + block - 1) / block;
    k_global_avg_pool_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(in, out, C, HW);
}

__global__ void k_global_avg_pool_backward(const float* grad_out, float* grad_in, size_t C, size_t HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * HW;
    if (idx < total) {
        size_t c = idx / HW;
        grad_in[idx] += grad_out[c] / static_cast<float>(HW);
    }
}

void global_avg_pool_backward(const float* grad_out, float* grad_in, size_t C, size_t HW, void* stream) {
    size_t total = C * HW;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_global_avg_pool_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_out, grad_in, C, HW);
}

// -----------------------------------------------------------------------------
// Upsample Bilinear 2D Kernels (2x scale optimized)
// -----------------------------------------------------------------------------
__global__ void k_upsample_bilinear_2x_forward(const float* in, float* out, size_t C, size_t H_in, size_t W_in) {
    size_t H_out = H_in * 2;
    size_t W_out = W_in * 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * H_out * W_out;

    if (idx < total) {
        size_t w_out = idx % W_out;
        size_t h_out = (idx / W_out) % H_out;
        size_t c = idx / (H_out * W_out);

        float h_in_f = (static_cast<float>(h_out) + 0.5f) * 0.5f - 0.5f;
        float w_in_f = (static_cast<float>(w_out) + 0.5f) * 0.5f - 0.5f;

        h_in_f = fmaxf(0.0f, fminf(static_cast<float>(H_in - 1), h_in_f));
        w_in_f = fmaxf(0.0f, fminf(static_cast<float>(W_in - 1), w_in_f));

        size_t h0 = static_cast<size_t>(h_in_f);
        size_t w0 = static_cast<size_t>(w_in_f);
        size_t h1 = min(h0 + 1, H_in - 1);
        size_t w1 = min(w0 + 1, W_in - 1);

        float h_diff = h_in_f - static_cast<float>(h0);
        float w_diff = w_in_f - static_cast<float>(w0);

        const float* in_c = in + c * (H_in * W_in);
        float val00 = in_c[h0 * W_in + w0];
        float val01 = in_c[h0 * W_in + w1];
        float val10 = in_c[h1 * W_in + w0];
        float val11 = in_c[h1 * W_in + w1];

        float top = val00 + (val01 - val00) * w_diff;
        float bot = val10 + (val11 - val10) * w_diff;
        out[idx] = top + (bot - top) * h_diff;
    }
}

void upsample_bilinear_2x_forward(const float* in, float* out, size_t C, size_t H_in, size_t W_in, void* stream) {
    size_t total = C * (H_in * 2) * (W_in * 2);
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_upsample_bilinear_2x_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(in, out, C, H_in, W_in);
}

__global__ void k_upsample_bilinear_2x_backward(const float* grad_out, float* grad_in, size_t C, size_t H_in, size_t W_in) {
    size_t H_out = H_in * 2;
    size_t W_out = W_in * 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * H_out * W_out;

    if (idx < total) {
        size_t w_out = idx % W_out;
        size_t h_out = (idx / W_out) % H_out;
        size_t c = idx / (H_out * W_out);

        float h_in_f = (static_cast<float>(h_out) + 0.5f) * 0.5f - 0.5f;
        float w_in_f = (static_cast<float>(w_out) + 0.5f) * 0.5f - 0.5f;

        h_in_f = fmaxf(0.0f, fminf(static_cast<float>(H_in - 1), h_in_f));
        w_in_f = fmaxf(0.0f, fminf(static_cast<float>(W_in - 1), w_in_f));

        size_t h0 = static_cast<size_t>(h_in_f);
        size_t w0 = static_cast<size_t>(w_in_f);
        size_t h1 = min(h0 + 1, H_in - 1);
        size_t w1 = min(w0 + 1, W_in - 1);

        float h_diff = h_in_f - static_cast<float>(h0);
        float w_diff = w_in_f - static_cast<float>(w0);

        float go = grad_out[idx];
        float* gin_c = grad_in + c * (H_in * W_in);

        atomicAdd(&gin_c[h0 * W_in + w0], go * (1.0f - h_diff) * (1.0f - w_diff));
        atomicAdd(&gin_c[h0 * W_in + w1], go * (1.0f - h_diff) * w_diff);
        atomicAdd(&gin_c[h1 * W_in + w0], go * h_diff * (1.0f - w_diff));
        atomicAdd(&gin_c[h1 * W_in + w1], go * h_diff * w_diff);
    }
}

void upsample_bilinear_2x_backward(const float* grad_out, float* grad_in, size_t C, size_t H_in, size_t W_in, void* stream) {
    size_t total = C * (H_in * 2) * (W_in * 2);
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_upsample_bilinear_2x_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_out, grad_in, C, H_in, W_in);
}

// -----------------------------------------------------------------------------
// Upsample Bilinear 2D Kernels (Arbitrary Dimensions)
// -----------------------------------------------------------------------------
__global__ void k_upsample_bilinear_forward(const float* in, float* out, size_t C,
                                           size_t H_in, size_t W_in, size_t H_out, size_t W_out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * H_out * W_out;
    if (idx < total) {
        size_t xo = idx % W_out;
        size_t yo = (idx / W_out) % H_out;
        size_t c = idx / (H_out * W_out);

        float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
        float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

        float src_y = (static_cast<float>(yo) + 0.5f) * scale_y - 0.5f;
        float src_x = (static_cast<float>(xo) + 0.5f) * scale_x - 0.5f;
        src_y = fmaxf(0.0f, fminf(static_cast<float>(H_in - 1), src_y));
        src_x = fmaxf(0.0f, fminf(static_cast<float>(W_in - 1), src_x));

        size_t y0 = static_cast<size_t>(src_y);
        size_t x0 = static_cast<size_t>(src_x);
        size_t y1 = min(y0 + 1, H_in - 1);
        size_t x1 = min(x0 + 1, W_in - 1);

        float ly = src_y - static_cast<float>(y0);
        float lx = src_x - static_cast<float>(x0);
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;

        const float* in_c = in + c * (H_in * W_in);
        float v00 = in_c[y0 * W_in + x0];
        float v01 = in_c[y0 * W_in + x1];
        float v10 = in_c[y1 * W_in + x0];
        float v11 = in_c[y1 * W_in + x1];

        out[idx] = hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
    }
}

void upsample_bilinear_forward(const float* in, float* out, size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out, void* stream) {
    size_t total = C * H_out * W_out;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_upsample_bilinear_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        in, out, C, H_in, W_in, H_out, W_out);
}

__global__ void k_upsample_bilinear_backward(const float* grad_out, float* grad_in, size_t C,
                                            size_t H_in, size_t W_in, size_t H_out, size_t W_out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * H_out * W_out;
    if (idx < total) {
        size_t xo = idx % W_out;
        size_t yo = (idx / W_out) % H_out;
        size_t c = idx / (H_out * W_out);

        float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
        float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

        float src_y = (static_cast<float>(yo) + 0.5f) * scale_y - 0.5f;
        float src_x = (static_cast<float>(xo) + 0.5f) * scale_x - 0.5f;
        src_y = fmaxf(0.0f, fminf(static_cast<float>(H_in - 1), src_y));
        src_x = fmaxf(0.0f, fminf(static_cast<float>(W_in - 1), src_x));

        size_t y0 = static_cast<size_t>(src_y);
        size_t x0 = static_cast<size_t>(src_x);
        size_t y1 = min(y0 + 1, H_in - 1);
        size_t x1 = min(x0 + 1, W_in - 1);

        float ly = src_y - static_cast<float>(y0);
        float lx = src_x - static_cast<float>(x0);
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;

        float go = grad_out[idx];
        float* gin_c = grad_in + c * (H_in * W_in);

        atomicAdd(&gin_c[y0 * W_in + x0], hy * hx * go);
        atomicAdd(&gin_c[y0 * W_in + x1], hy * lx * go);
        atomicAdd(&gin_c[y1 * W_in + x0], ly * hx * go);
        atomicAdd(&gin_c[y1 * W_in + x1], ly * lx * go);
    }
}

void upsample_bilinear_backward(const float* grad_out, float* grad_in, size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out, void* stream) {
    size_t total = C * H_out * W_out;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_upsample_bilinear_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        grad_out, grad_in, C, H_in, W_in, H_out, W_out);
}

// -----------------------------------------------------------------------------
// PixelShuffle Kernels
// -----------------------------------------------------------------------------
__global__ void k_pixel_shuffle_forward(const float* in, float* out, size_t C_in, size_t H_in, size_t W_in, size_t r) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r2 = r * r;
    size_t C_out = C_in / r2;
    size_t H_out = H_in * r;
    size_t W_out = W_in * r;
    size_t total = C_out * H_out * W_out;

    if (idx < total) {
        size_t x = idx % W_out;
        size_t y = (idx / W_out) % H_out;
        size_t c = idx / (H_out * W_out);

        size_t in_y = y / r;
        size_t in_x = x / r;
        size_t sub_y = y % r;
        size_t sub_x = x % r;
        size_t c_in = c * r2 + sub_y * r + sub_x;

        out[idx] = in[c_in * (H_in * W_in) + in_y * W_in + in_x];
    }
}

void pixel_shuffle_forward(const float* in, float* out, size_t C_in, size_t H_in, size_t W_in, size_t r, void* stream) {
    size_t r2 = r * r;
    size_t C_out = C_in / r2;
    size_t total = C_out * (H_in * r) * (W_in * r);
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_pixel_shuffle_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(in, out, C_in, H_in, W_in, r);
}

__global__ void k_pixel_shuffle_backward(const float* grad_out, float* grad_in, size_t C_in, size_t H_in, size_t W_in, size_t r) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C_in * H_in * W_in;

    if (idx < total) {
        size_t x_in = idx % W_in;
        size_t y_in = (idx / W_in) % H_in;
        size_t c_in = idx / (H_in * W_in);

        size_t r2 = r * r;
        size_t c = c_in / r2;
        size_t rem = c_in % r2;
        size_t sub_y = rem / r;
        size_t sub_x = rem % r;

        size_t y_out = y_in * r + sub_y;
        size_t x_out = x_in * r + sub_x;
        size_t H_out = H_in * r;
        size_t W_out = W_in * r;

        grad_in[idx] += grad_out[c * (H_out * W_out) + y_out * W_out + x_out];
    }
}

void pixel_shuffle_backward(const float* grad_out, float* grad_in, size_t C_in, size_t H_in, size_t W_in, size_t r, void* stream) {
    size_t total = C_in * H_in * W_in;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_pixel_shuffle_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_out, grad_in, C_in, H_in, W_in, r);
}

// -----------------------------------------------------------------------------
// MaxPool2D Kernels
// -----------------------------------------------------------------------------
__global__ void k_max_pool2d_forward(const float* in, float* out, int64_t* argmax,
                                    size_t C, size_t H_in, size_t W_in,
                                    size_t H_out, size_t W_out,
                                    size_t K, size_t pad, size_t stride) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * H_out * W_out;
    if (idx < total) {
        size_t xo = idx % W_out;
        size_t yo = (idx / W_out) % H_out;
        size_t c = idx / (H_out * W_out);

        int base_yi = static_cast<int>(yo * stride) - static_cast<int>(pad);
        int base_xi = static_cast<int>(xo * stride) - static_cast<int>(pad);
        size_t in_c_offset = c * (H_in * W_in);

        float max_val = -1e30f;
        int64_t max_idx = -1;

        for (size_t ky = 0; ky < K; ++ky) {
            int yi = base_yi + static_cast<int>(ky);
            if (yi >= 0 && yi < static_cast<int>(H_in)) {
                for (size_t kx = 0; kx < K; ++kx) {
                    int xi = base_xi + static_cast<int>(kx);
                    if (xi >= 0 && xi < static_cast<int>(W_in)) {
                        int64_t cur_idx = static_cast<int64_t>(in_c_offset + yi * W_in + xi);
                        float val = in[cur_idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = cur_idx;
                        }
                    }
                }
            }
        }
        out[idx] = (max_idx >= 0) ? max_val : 0.0f;
        if (argmax) argmax[idx] = max_idx;
    }
}

void max_pool2d_forward(const float* in, float* out, int64_t* argmax, size_t C, size_t H_in, size_t W_in,
                        size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, void* stream) {
    size_t total = C * H_out * W_out;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_max_pool2d_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        in, out, argmax, C, H_in, W_in, H_out, W_out, K, pad, stride);
}

__global__ void k_max_pool2d_backward(const float* grad_out, const int64_t* argmax, float* grad_in, size_t total_out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_out) {
        int64_t in_idx = argmax[idx];
        if (in_idx >= 0) {
            atomicAdd(&grad_in[in_idx], grad_out[idx]);
        }
    }
}

void max_pool2d_backward(const float* grad_out, const int64_t* argmax, float* grad_in, size_t total_out, void* stream) {
    if (total_out == 0) return;
    size_t block = 256;
    size_t grid = (total_out + block - 1) / block;
    k_max_pool2d_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(grad_out, argmax, grad_in, total_out);
}

// -----------------------------------------------------------------------------
// Conv2D 1x1 Pointwise Kernels
// -----------------------------------------------------------------------------
__global__ void k_conv2d_1x1_forward(const float* __restrict__ in,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ bias,
                                     float* __restrict__ out,
                                     size_t C_in, size_t C_out, size_t HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C_out * HW;
    if (idx < total) {
        size_t sp = idx % HW;
        size_t co = idx / HW;

        float sum = bias ? bias[co] : 0.0f;
        const float* w_row = weight + co * C_in;
        for (size_t ci = 0; ci < C_in; ++ci) {
            sum += in[ci * HW + sp] * w_row[ci];
        }
        out[idx] = sum;
    }
}

void conv2d_1x1_forward(const float* in, const float* weight, const float* bias, float* out,
                        size_t C_in, size_t C_out, size_t HW, void* stream) {
    size_t total = C_out * HW;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_conv2d_1x1_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(in, weight, bias, out, C_in, C_out, HW);
}

__global__ void k_conv2d_1x1_backward_in(const float* __restrict__ grad_out,
                                        const float* __restrict__ weight,
                                        float* __restrict__ grad_in,
                                        size_t C_in, size_t C_out, size_t HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C_in * HW;
    if (idx < total) {
        size_t sp = idx % HW;
        size_t ci = idx / HW;

        float sum = 0.0f;
        for (size_t co = 0; co < C_out; ++co) {
            sum += grad_out[co * HW + sp] * weight[co * C_in + ci];
        }
        grad_in[idx] += sum;
    }
}

__global__ void k_conv2d_1x1_backward_weight(const float* __restrict__ grad_out,
                                             const float* __restrict__ in,
                                             float* __restrict__ grad_weight,
                                             size_t C_in, size_t C_out, size_t HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C_out * C_in;
    if (idx < total) {
        size_t ci = idx % C_in;
        size_t co = idx / C_in;

        const float* go = grad_out + co * HW;
        const float* xi = in + ci * HW;
        float sum = 0.0f;
        for (size_t sp = 0; sp < HW; ++sp) {
            sum += go[sp] * xi[sp];
        }
        grad_weight[idx] += sum;
    }
}

__global__ void k_conv2d_backward_bias(const float* __restrict__ grad_out,
                                       float* __restrict__ grad_bias,
                                       size_t C_out, size_t HW) {
    size_t co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co < C_out) {
        const float* go = grad_out + co * HW;
        float sum = 0.0f;
        for (size_t sp = 0; sp < HW; ++sp) {
            sum += go[sp];
        }
        grad_bias[co] += sum;
    }
}

void conv2d_1x1_backward(const float* grad_out, const float* in, const float* weight,
                         float* grad_in, float* grad_weight, float* grad_bias,
                         size_t C_in, size_t C_out, size_t HW, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (grad_in) {
        size_t total_in = C_in * HW;
        size_t block = 256;
        size_t grid = (total_in + block - 1) / block;
        k_conv2d_1x1_backward_in<<<grid, block, 0, s>>>(grad_out, weight, grad_in, C_in, C_out, HW);
    }
    if (grad_weight) {
        size_t total_w = C_out * C_in;
        size_t block = 128;
        size_t grid = (total_w + block - 1) / block;
        k_conv2d_1x1_backward_weight<<<grid, block, 0, s>>>(grad_out, in, grad_weight, C_in, C_out, HW);
    }
    if (grad_bias) {
        size_t block = 128;
        size_t grid = (C_out + block - 1) / block;
        k_conv2d_backward_bias<<<grid, block, 0, s>>>(grad_out, grad_bias, C_out, HW);
    }
}

// -----------------------------------------------------------------------------
// Conv2D Depthwise Kernels
// -----------------------------------------------------------------------------
__global__ void k_conv2d_dw_forward(const float* __restrict__ in,
                                    const float* __restrict__ weight,
                                    const float* __restrict__ bias,
                                    float* __restrict__ out,
                                    size_t C, size_t H_in, size_t W_in,
                                    size_t H_out, size_t W_out,
                                    size_t K, size_t pad, size_t stride, size_t dil) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * H_out * W_out;
    if (idx < total) {
        size_t xo = idx % W_out;
        size_t yo = (idx / W_out) % H_out;
        size_t c = idx / (H_out * W_out);

        float sum = bias ? bias[c] : 0.0f;
        const float* w_c = weight + c * (K * K);
        const float* in_c = in + c * (H_in * W_in);

        int pad_i = static_cast<int>(pad);
        int dil_i = static_cast<int>(dil);
        int str_i = static_cast<int>(stride);

        for (size_t ky = 0; ky < K; ++ky) {
            int yi = static_cast<int>(yo * str_i) - pad_i + static_cast<int>(ky * dil_i);
            if (yi >= 0 && yi < static_cast<int>(H_in)) {
                for (size_t kx = 0; kx < K; ++kx) {
                    int xi = static_cast<int>(xo * str_i) - pad_i + static_cast<int>(kx * dil_i);
                    if (xi >= 0 && xi < static_cast<int>(W_in)) {
                        sum += in_c[yi * W_in + xi] * w_c[ky * K + kx];
                    }
                }
            }
        }
        out[idx] = sum;
    }
}

void conv2d_dw_forward(const float* in, const float* weight, const float* bias, float* out,
                       size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out,
                       size_t K, size_t pad, size_t stride, size_t dil, void* stream) {
    size_t total = C * H_out * W_out;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_conv2d_dw_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        in, weight, bias, out, C, H_in, W_in, H_out, W_out, K, pad, stride, dil);
}

__global__ void k_conv2d_dw_backward_in(const float* __restrict__ grad_out,
                                       const float* __restrict__ weight,
                                       float* __restrict__ grad_in,
                                       size_t C, size_t H_in, size_t W_in,
                                       size_t H_out, size_t W_out,
                                       size_t K, size_t pad, size_t stride, size_t dil) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * H_in * W_in;
    if (idx < total) {
        size_t xi = idx % W_in;
        size_t yi = (idx / W_in) % H_in;
        size_t c = idx / (H_in * W_in);

        float sum = 0.0f;
        const float* go_c = grad_out + c * (H_out * W_out);
        const float* w_c = weight + c * (K * K);

        int pad_i = static_cast<int>(pad);
        int dil_i = static_cast<int>(dil);
        int str_i = static_cast<int>(stride);

        for (size_t ky = 0; ky < K; ++ky) {
            int y_diff = static_cast<int>(yi) + pad_i - static_cast<int>(ky * dil_i);
            if (y_diff >= 0 && y_diff % str_i == 0) {
                size_t yo = y_diff / str_i;
                if (yo < H_out) {
                    for (size_t kx = 0; kx < K; ++kx) {
                        int x_diff = static_cast<int>(xi) + pad_i - static_cast<int>(kx * dil_i);
                        if (x_diff >= 0 && x_diff % str_i == 0) {
                            size_t xo = x_diff / str_i;
                            if (xo < W_out) {
                                sum += go_c[yo * W_out + xo] * w_c[ky * K + kx];
                            }
                        }
                    }
                }
            }
        }
        grad_in[idx] += sum;
    }
}

__global__ void k_conv2d_dw_backward_w(const float* __restrict__ grad_out,
                                      const float* __restrict__ in,
                                      float* __restrict__ grad_weight,
                                      size_t C, size_t H_in, size_t W_in,
                                      size_t H_out, size_t W_out,
                                      size_t K, size_t pad, size_t stride, size_t dil) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * (K * K);
    if (idx < total) {
        size_t kx = idx % K;
        size_t ky = (idx / K) % K;
        size_t c = idx / (K * K);

        const float* in_c = in + c * (H_in * W_in);
        const float* go_c = grad_out + c * (H_out * W_out);

        int pad_i = static_cast<int>(pad);
        int dil_i = static_cast<int>(dil);
        int str_i = static_cast<int>(stride);

        float sum = 0.0f;
        for (size_t yo = 0; yo < H_out; ++yo) {
            int yi = static_cast<int>(yo * str_i) - pad_i + static_cast<int>(ky * dil_i);
            if (yi >= 0 && yi < static_cast<int>(H_in)) {
                for (size_t xo = 0; xo < W_out; ++xo) {
                    int xi = static_cast<int>(xo * str_i) - pad_i + static_cast<int>(kx * dil_i);
                    if (xi >= 0 && xi < static_cast<int>(W_in)) {
                        sum += in_c[yi * W_in + xi] * go_c[yo * W_out + xo];
                    }
                }
            }
        }
        grad_weight[idx] += sum;
    }
}

void conv2d_dw_backward(const float* grad_out, const float* in, const float* weight,
                        float* grad_in, float* grad_weight, float* grad_bias,
                        size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out,
                        size_t K, size_t pad, size_t stride, size_t dil, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (grad_in) {
        size_t total_in = C * H_in * W_in;
        size_t block = 256;
        size_t grid = (total_in + block - 1) / block;
        k_conv2d_dw_backward_in<<<grid, block, 0, s>>>(grad_out, weight, grad_in, C, H_in, W_in, H_out, W_out, K, pad, stride, dil);
    }
    if (grad_weight) {
        size_t total_w = C * (K * K);
        size_t block = 64;
        size_t grid = (total_w + block - 1) / block;
        k_conv2d_dw_backward_w<<<grid, block, 0, s>>>(grad_out, in, grad_weight, C, H_in, W_in, H_out, W_out, K, pad, stride, dil);
    }
    if (grad_bias) {
        size_t block = 128;
        size_t grid = (C + block - 1) / block;
        k_conv2d_backward_bias<<<grid, block, 0, s>>>(grad_out, grad_bias, C, H_out * W_out);
    }
}

// -----------------------------------------------------------------------------
// General Grouped Conv2D
// -----------------------------------------------------------------------------
__global__ void k_conv2d_forward(const float* __restrict__ in,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ bias,
                                 float* __restrict__ out,
                                 size_t C_in, size_t C_out,
                                 size_t H_in, size_t W_in,
                                 size_t H_out, size_t W_out,
                                 size_t K, size_t pad, size_t stride, size_t dil, size_t groups) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C_out * H_out * W_out;
    if (idx < total) {
        size_t xo = idx % W_out;
        size_t yo = (idx / W_out) % H_out;
        size_t co = idx / (H_out * W_out);

        size_t c_out_per_g = C_out / groups;
        size_t c_in_per_g = C_in / groups;
        size_t g = co / c_out_per_g;
        size_t ci_base = g * c_in_per_g;

        float sum = bias ? bias[co] : 0.0f;
        int pad_i = static_cast<int>(pad);
        int dil_i = static_cast<int>(dil);
        int str_i = static_cast<int>(stride);

        for (size_t ci_rel = 0; ci_rel < c_in_per_g; ++ci_rel) {
            size_t ci = ci_base + ci_rel;
            const float* in_ci = in + ci * (H_in * W_in);
            const float* w_ci = weight + co * (c_in_per_g * K * K) + ci_rel * (K * K);

            for (size_t ky = 0; ky < K; ++ky) {
                int yi = static_cast<int>(yo * str_i) - pad_i + static_cast<int>(ky * dil_i);
                if (yi >= 0 && yi < static_cast<int>(H_in)) {
                    for (size_t kx = 0; kx < K; ++kx) {
                        int xi = static_cast<int>(xo * str_i) - pad_i + static_cast<int>(kx * dil_i);
                        if (xi >= 0 && xi < static_cast<int>(W_in)) {
                            sum += in_ci[yi * W_in + xi] * w_ci[ky * K + kx];
                        }
                    }
                }
            }
        }
        out[idx] = sum;
    }
}

__global__ void k_conv2d_backward_in(const float* __restrict__ grad_out,
                                    const float* __restrict__ weight,
                                    float* __restrict__ grad_in,
                                    size_t C_in, size_t C_out,
                                    size_t H_in, size_t W_in,
                                    size_t H_out, size_t W_out,
                                    size_t K, size_t pad, size_t stride, size_t dil, size_t groups) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C_in * H_in * W_in;
    if (idx < total) {
        size_t xi = idx % W_in;
        size_t yi = (idx / W_in) % H_in;
        size_t ci = idx / (H_in * W_in);

        size_t c_in_per_g = C_in / groups;
        size_t c_out_per_g = C_out / groups;
        size_t g = ci / c_in_per_g;
        size_t ci_rel = ci % c_in_per_g;

        float sum = 0.0f;
        int pad_i = static_cast<int>(pad);
        int dil_i = static_cast<int>(dil);
        int str_i = static_cast<int>(stride);

        for (size_t co_rel = 0; co_rel < c_out_per_g; ++co_rel) {
            size_t co = g * c_out_per_g + co_rel;
            const float* go_co = grad_out + co * (H_out * W_out);
            const float* w_ci = weight + co * (c_in_per_g * K * K) + ci_rel * (K * K);

            for (size_t ky = 0; ky < K; ++ky) {
                int y_diff = static_cast<int>(yi) + pad_i - static_cast<int>(ky * dil_i);
                if (y_diff >= 0 && y_diff % str_i == 0) {
                    size_t yo = y_diff / str_i;
                    if (yo < H_out) {
                        for (size_t kx = 0; kx < K; ++kx) {
                            int x_diff = static_cast<int>(xi) + pad_i - static_cast<int>(kx * dil_i);
                            if (x_diff >= 0 && x_diff % str_i == 0) {
                                size_t xo = x_diff / str_i;
                                if (xo < W_out) {
                                    sum += go_co[yo * W_out + xo] * w_ci[ky * K + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
        grad_in[idx] += sum;
    }
}

__global__ void k_conv2d_backward_w(const float* __restrict__ grad_out,
                                   const float* __restrict__ in,
                                   float* __restrict__ grad_weight,
                                   size_t C_in, size_t C_out,
                                   size_t H_in, size_t W_in,
                                   size_t H_out, size_t W_out,
                                   size_t K, size_t pad, size_t stride, size_t dil, size_t groups) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t c_in_per_g = C_in / groups;
    size_t c_out_per_g = C_out / groups;
    size_t total = C_out * c_in_per_g * K * K;

    if (idx < total) {
        size_t kx = idx % K;
        size_t ky = (idx / K) % K;
        size_t ci_rel = (idx / (K * K)) % c_in_per_g;
        size_t co = idx / (c_in_per_g * K * K);

        size_t g = co / c_out_per_g;
        size_t ci = g * c_in_per_g + ci_rel;

        const float* in_ci = in + ci * (H_in * W_in);
        const float* go_co = grad_out + co * (H_out * W_out);

        int pad_i = static_cast<int>(pad);
        int dil_i = static_cast<int>(dil);
        int str_i = static_cast<int>(stride);

        float sum = 0.0f;
        for (size_t yo = 0; yo < H_out; ++yo) {
            int yi = static_cast<int>(yo * str_i) - pad_i + static_cast<int>(ky * dil_i);
            if (yi >= 0 && yi < static_cast<int>(H_in)) {
                for (size_t xo = 0; xo < W_out; ++xo) {
                    int xi = static_cast<int>(xo * str_i) - pad_i + static_cast<int>(kx * dil_i);
                    if (xi >= 0 && xi < static_cast<int>(W_in)) {
                        sum += in_ci[yi * W_in + xi] * go_co[yo * W_out + xo];
                    }
                }
            }
        }
        grad_weight[idx] += sum;
    }
}

void conv2d_forward(const float* in, const float* weight, const float* bias, float* out,
                    size_t C_in, size_t C_out, size_t H_in, size_t W_in,
                    size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, size_t dil, size_t groups, void* stream) {
    if (groups == 1 && K == 1 && pad == 0 && stride == 1 && H_in == H_out && W_in == W_out) {
        conv2d_1x1_forward(in, weight, bias, out, C_in, C_out, H_in * W_in, stream);
        return;
    }
    if (groups == C_in && C_in == C_out) {
        conv2d_dw_forward(in, weight, bias, out, C_in, H_in, W_in, H_out, W_out, K, pad, stride, dil, stream);
        return;
    }
    size_t total = C_out * H_out * W_out;
    if (total == 0) return;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_conv2d_forward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        in, weight, bias, out, C_in, C_out, H_in, W_in, H_out, W_out, K, pad, stride, dil, groups);
}

void conv2d_backward(const float* grad_out, const float* in, const float* weight,
                     float* grad_in, float* grad_weight, float* grad_bias,
                     size_t C_in, size_t C_out, size_t H_in, size_t W_in,
                     size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, size_t dil, size_t groups, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (groups == 1 && K == 1 && pad == 0 && stride == 1 && H_in == H_out && W_in == W_out) {
        conv2d_1x1_backward(grad_out, in, weight, grad_in, grad_weight, grad_bias, C_in, C_out, H_in * W_in, stream);
        return;
    }
    if (groups == C_in && C_in == C_out) {
        conv2d_dw_backward(grad_out, in, weight, grad_in, grad_weight, grad_bias, C_in, H_in, W_in, H_out, W_out, K, pad, stride, dil, stream);
        return;
    }
    if (grad_in) {
        size_t total_in = C_in * H_in * W_in;
        size_t block = 256;
        size_t grid = (total_in + block - 1) / block;
        k_conv2d_backward_in<<<grid, block, 0, s>>>(
            grad_out, weight, grad_in, C_in, C_out, H_in, W_in, H_out, W_out, K, pad, stride, dil, groups);
    }
    if (grad_weight) {
        size_t c_in_per_g = C_in / groups;
        size_t total_w = C_out * c_in_per_g * K * K;
        size_t block = 128;
        size_t grid = (total_w + block - 1) / block;
        k_conv2d_backward_w<<<grid, block, 0, s>>>(
            grad_out, in, grad_weight, C_in, C_out, H_in, W_in, H_out, W_out, K, pad, stride, dil, groups);
    }
    if (grad_bias) {
        size_t block = 128;
        size_t grid = (C_out + block - 1) / block;
        k_conv2d_backward_bias<<<grid, block, 0, s>>>(grad_out, grad_bias, C_out, H_out * W_out);
    }
}

// -----------------------------------------------------------------------------
// Group Normalization Kernels
// -----------------------------------------------------------------------------
__global__ void k_group_norm_stats(const float* __restrict__ in,
                                   float* __restrict__ saved_mean,
                                   float* __restrict__ saved_rstd,
                                   size_t num_groups, size_t C, size_t HW, float eps) {
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < num_groups) {
        size_t c_per_group = C / num_groups;
        size_t group_size = c_per_group * HW;

        double sum = 0.0;
        double sum_sq = 0.0;

        for (size_t c_rel = 0; c_rel < c_per_group; ++c_rel) {
            size_t c = g * c_per_group + c_rel;
            const float* p_in = in + c * HW;
            for (size_t sp = 0; sp < HW; ++sp) {
                float v = p_in[sp];
                sum += v;
                sum_sq += v * v;
            }
        }

        double mean = sum / static_cast<double>(group_size);
        double var = (sum_sq / static_cast<double>(group_size)) - (mean * mean);
        if (var < 0.0) var = 0.0;
        float rstd = 1.0f / sqrtf(static_cast<float>(var) + eps);

        saved_mean[g] = static_cast<float>(mean);
        saved_rstd[g] = rstd;
    }
}

__global__ void k_group_norm_affine(const float* __restrict__ in,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ saved_mean,
                                    const float* __restrict__ saved_rstd,
                                    float* __restrict__ out,
                                    size_t num_groups, size_t C, size_t HW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = C * HW;
    if (idx < total) {
        size_t sp = idx % HW;
        size_t c = idx / HW;
        size_t c_per_group = C / num_groups;
        size_t g = c / c_per_group;

        float mean = saved_mean[g];
        float rstd = saved_rstd[g];
        float x_hat = (in[idx] - mean) * rstd;

        float g_val = gamma ? gamma[c] : 1.0f;
        float b_val = beta ? beta[c] : 0.0f;

        out[idx] = x_hat * g_val + b_val;
    }
}

void group_norm_forward(const float* in, const float* gamma, const float* beta, float* out,
                        float* saved_mean, float* saved_rstd,
                        size_t num_groups, size_t C, size_t HW, float eps, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    size_t g_block = 32;
    size_t g_grid = (num_groups + g_block - 1) / g_block;
    k_group_norm_stats<<<g_grid, g_block, 0, s>>>(in, saved_mean, saved_rstd, num_groups, C, HW, eps);

    size_t total = C * HW;
    size_t block = 256;
    size_t grid = (total + block - 1) / block;
    k_group_norm_affine<<<grid, block, 0, s>>>(in, gamma, beta, saved_mean, saved_rstd, out, num_groups, C, HW);
}

__global__ void k_group_norm_backward(const float* __restrict__ grad_out,
                                      const float* __restrict__ in,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ saved_mean,
                                      const float* __restrict__ saved_rstd,
                                      float* __restrict__ grad_in,
                                      float* __restrict__ grad_gamma,
                                      float* __restrict__ grad_beta,
                                      size_t num_groups, size_t C, size_t HW) {
    size_t g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < num_groups) {
        size_t c_per_group = C / num_groups;
        size_t M = c_per_group * HW;
        float mean = saved_mean[g];
        float rstd = saved_rstd[g];

        // 1. Compute ds and db for group
        double ds = 0.0;
        double db = 0.0;

        for (size_t c_rel = 0; c_rel < c_per_group; ++c_rel) {
            size_t c = g * c_per_group + c_rel;
            float g_val = gamma ? gamma[c] : 1.0f;
            const float* go = grad_out + c * HW;
            const float* xi = in + c * HW;

            for (size_t sp = 0; sp < HW; ++sp) {
                float dy = go[sp] * g_val;
                float x_hat = (xi[sp] - mean) * rstd;
                ds += dy * x_hat;
                db += dy;

                if (grad_gamma) atomicAdd(&grad_gamma[c], go[sp] * x_hat);
                if (grad_beta) atomicAdd(&grad_beta[c], go[sp]);
            }
        }

        // 2. Propagate to grad_in
        if (grad_in) {
            double inv_M = 1.0 / static_cast<double>(M);
            for (size_t c_rel = 0; c_rel < c_per_group; ++c_rel) {
                size_t c = g * c_per_group + c_rel;
                float g_val = gamma ? gamma[c] : 1.0f;
                const float* go = grad_out + c * HW;
                const float* xi = in + c * HW;
                float* gi = grad_in + c * HW;

                for (size_t sp = 0; sp < HW; ++sp) {
                    float dy = go[sp] * g_val;
                    float x_hat = (xi[sp] - mean) * rstd;
                    float dx = static_cast<float>(rstd * (dy - (x_hat * ds + db) * inv_M));
                    gi[sp] += dx;
                }
            }
        }
    }
}

void group_norm_backward(const float* grad_out, const float* in, const float* gamma,
                         const float* saved_mean, const float* saved_rstd,
                         float* grad_in, float* grad_gamma, float* grad_beta,
                         size_t num_groups, size_t C, size_t HW, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    size_t g_block = 32;
    size_t g_grid = (num_groups + g_block - 1) / g_block;
    k_group_norm_backward<<<g_grid, g_block, 0, s>>>(grad_out, in, gamma, saved_mean, saved_rstd,
                                                    grad_in, grad_gamma, grad_beta,
                                                    num_groups, C, HW);
}

// -----------------------------------------------------------------------------
// BCE with Logits Loss Backward
// -----------------------------------------------------------------------------
__global__ void k_bce_with_logits_backward(const float* logits, const float* targets, float* grad_logits,
                                          float scale, float pos_weight, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float z = logits[i];
        float y = targets[i];
        float sig = 1.0f / (1.0f + __expf(-z));
        float w = 1.0f + (pos_weight - 1.0f) * y;
        grad_logits[i] = (sig * w - pos_weight * y) * scale;
    }
}

void bce_with_logits_backward(const float* logits, const float* targets, float* grad_logits,
                             float grad_out, float weight, float pos_weight, size_t n, void* stream) {
    if (n == 0) return;
    float scale = (grad_out * weight) / static_cast<float>(n);
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_bce_with_logits_backward<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, targets, grad_logits, scale, pos_weight, n);
}

// -----------------------------------------------------------------------------
// AdamW Optimizer Step
// -----------------------------------------------------------------------------
__global__ void k_adamw_step(float* theta, const float* g, float* m, float* v,
                             float lr, float beta1, float beta2, float eps, float wd,
                             float step_size, float sqrt_bc2, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float th = theta[i];
        if (wd != 0.0f) {
            th -= lr * wd * th;
        }
        float g_val = g[i];
        float m_val = beta1 * m[i] + (1.0f - beta1) * g_val;
        float v_val = beta2 * v[i] + (1.0f - beta2) * g_val * g_val;
        m[i] = m_val;
        v[i] = v_val;

        float denom = (sqrtf(v_val) / sqrt_bc2) + eps;
        th -= step_size * (m_val / denom);
        theta[i] = th;
    }
}

void adamw_step(float* theta, const float* g, float* m, float* v,
                float lr, float beta1, float beta2, float eps, float wd,
                float step_size, float sqrt_bc2, size_t n, void* stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    k_adamw_step<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        theta, g, m, v, lr, beta1, beta2, eps, wd, step_size, sqrt_bc2, n);
}

} // namespace soar::cuda::kernels
