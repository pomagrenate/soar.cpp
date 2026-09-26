#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <cmath>

namespace soar::cuda::kernels {

/**
 * @brief PyTorch exact area_pixel_compute_source_index function (align_corners = false)
 * From ATen/native/cuda/UpSample.cuh
 */
__device__ __forceinline__ float area_pixel_compute_source_index(
    float scale, int dst_index, bool align_corners) {
    if (align_corners) {
        return scale * dst_index;
    } else {
        float src_idx = scale * (dst_index + 0.5f) - 0.5f;
        return (src_idx < 0.0f) ? 0.0f : src_idx;
    }
}

/**
 * @brief PyTorch exact Bilinear Upsample 2D forward kernel (from UpSampleBilinear2d.cu)
 */
__global__ void k_upsample_bilinear_forward(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    float rheight,
    float rwidth) {
    int64_t total = C * H_out * W_out;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t w2 = idx % W_out;
        int64_t temp = idx / W_out;
        int64_t h2 = temp % H_out;
        int64_t c = temp / H_out;

        float h1r = area_pixel_compute_source_index(rheight, h2, false);
        int64_t h1 = static_cast<int64_t>(h1r);
        int64_t h1p = (h1 < H_in - 1) ? 1 : 0;
        float h1lambda = h1r - h1;
        float h0lambda = 1.0f - h1lambda;

        float w1r = area_pixel_compute_source_index(rwidth, w2, false);
        int64_t w1 = static_cast<int64_t>(w1r);
        int64_t w1p = (w1 < W_in - 1) ? 1 : 0;
        float w1lambda = w1r - w1;
        float w0lambda = 1.0f - w1lambda;

        const float* in_c = in + c * (H_in * W_in);
        float p00 = in_c[h1 * W_in + w1];
        float p01 = in_c[h1 * W_in + (w1 + w1p)];
        float p10 = in_c[(h1 + h1p) * W_in + w1];
        float p11 = in_c[(h1 + h1p) * W_in + (w1 + w1p)];

        float val = h0lambda * (w0lambda * p00 + w1lambda * p01) +
                    h1lambda * (w0lambda * p10 + w1lambda * p11);
        out[idx] = val;
    }
}

/**
 * @brief PyTorch exact Bilinear Upsample 2D backward kernel (from UpSampleBilinear2d.cu)
 */
__global__ void k_upsample_bilinear_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    float rheight,
    float rwidth) {
    int64_t total = C * H_out * W_out;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t w2 = idx % W_out;
        int64_t temp = idx / W_out;
        int64_t h2 = temp % H_out;
        int64_t c = temp / H_out;

        float h1r = area_pixel_compute_source_index(rheight, h2, false);
        int64_t h1 = static_cast<int64_t>(h1r);
        int64_t h1p = (h1 < H_in - 1) ? 1 : 0;
        float h1lambda = h1r - h1;
        float h0lambda = 1.0f - h1lambda;

        float w1r = area_pixel_compute_source_index(rwidth, w2, false);
        int64_t w1 = static_cast<int64_t>(w1r);
        int64_t w1p = (w1 < W_in - 1) ? 1 : 0;
        float w1lambda = w1r - w1;
        float w0lambda = 1.0f - w1lambda;

        float dy = grad_out[idx];
        float* grad_in_c = grad_in + c * (H_in * W_in);

        atomicAdd(&grad_in_c[h1 * W_in + w1], dy * h0lambda * w0lambda);
        atomicAdd(&grad_in_c[h1 * W_in + (w1 + w1p)], dy * h0lambda * w1lambda);
        atomicAdd(&grad_in_c[(h1 + h1p) * W_in + w1], dy * h1lambda * w0lambda);
        atomicAdd(&grad_in_c[(h1 + h1p) * W_in + (w1 + w1p)], dy * h1lambda * w1lambda);
    }
}

void upsample_bilinear_forward(const float* in, float* out, size_t C, size_t H_in, size_t W_in,
                               size_t H_out, size_t W_out, void* stream) {
    if (C == 0 || H_in == 0 || W_in == 0 || H_out == 0 || W_out == 0 || !in || !out) return;
    float rheight = static_cast<float>(H_in) / static_cast<float>(H_out);
    float rwidth = static_cast<float>(W_in) / static_cast<float>(W_out);
    int64_t total = static_cast<int64_t>(C * H_out * W_out);
    int blocks = GET_BLOCKS(total);
    k_upsample_bilinear_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        in, out, C, H_in, W_in, H_out, W_out, rheight, rwidth);
}

void upsample_bilinear_backward(const float* grad_out, float* grad_in, size_t C, size_t H_in, size_t W_in,
                                size_t H_out, size_t W_out, void* stream) {
    if (C == 0 || H_in == 0 || W_in == 0 || H_out == 0 || W_out == 0 || !grad_out || !grad_in) return;
    float rheight = static_cast<float>(H_in) / static_cast<float>(H_out);
    float rwidth = static_cast<float>(W_in) / static_cast<float>(W_out);
    int64_t total = static_cast<int64_t>(C * H_out * W_out);
    int blocks = GET_BLOCKS(total);
    k_upsample_bilinear_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        grad_out, grad_in, C, H_in, W_in, H_out, W_out, rheight, rwidth);
}

void upsample_bilinear_2x_forward(const float* in, float* out, size_t C, size_t H_in, size_t W_in, void* stream) {
    upsample_bilinear_forward(in, out, C, H_in, W_in, H_in * 2, W_in * 2, stream);
}

void upsample_bilinear_2x_backward(const float* grad_out, float* grad_in, size_t C, size_t H_in, size_t W_in, void* stream) {
    upsample_bilinear_backward(grad_out, grad_in, C, H_in, W_in, H_in * 2, W_in * 2, stream);
}

/**
 * @brief PixelShuffle forward kernel:
 * in: [C_out * r * r, H_in, W_in] -> out: [C_out, H_in * r, W_in * r]
 */
__global__ void k_pixel_shuffle_forward(
    const float* __restrict__ in,
    float* __restrict__ out,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t r) {
    int64_t total = C_out * H_out * W_out;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t w_out = idx % W_out;
        int64_t temp = idx / W_out;
        int64_t h_out = temp % H_out;
        int64_t c_out = temp / H_out;

        int64_t h_in = h_out / r;
        int64_t r_h = h_out % r;
        int64_t w_in = w_out / r;
        int64_t r_w = w_out % r;

        int64_t c_in = (c_out * r + r_h) * r + r_w;
        int64_t in_idx = (c_in * H_in + h_in) * W_in + w_in;
        out[idx] = in[in_idx];
    }
}

/**
 * @brief PixelShuffle backward kernel:
 * grad_out: [C_out, H_in * r, W_in * r] -> grad_in: [C_out * r * r, H_in, W_in]
 */
__global__ void k_pixel_shuffle_backward(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t r) {
    int64_t total = C_out * H_out * W_out;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t w_out = idx % W_out;
        int64_t temp = idx / W_out;
        int64_t h_out = temp % H_out;
        int64_t c_out = temp / H_out;

        int64_t h_in = h_out / r;
        int64_t r_h = h_out % r;
        int64_t w_in = w_out / r;
        int64_t r_w = w_out % r;

        int64_t c_in = (c_out * r + r_h) * r + r_w;
        int64_t in_idx = (c_in * H_in + h_in) * W_in + w_in;
        grad_in[in_idx] += grad_out[idx];
    }
}

void pixel_shuffle_forward(const float* in, float* out, size_t C_in, size_t H_in, size_t W_in, size_t r, void* stream) {
    if (C_in == 0 || H_in == 0 || W_in == 0 || r == 0 || !in || !out) return;
    size_t C_out = C_in / (r * r);
    size_t H_out = H_in * r;
    size_t W_out = W_in * r;
    int64_t total = static_cast<int64_t>(C_out * H_out * W_out);
    int blocks = GET_BLOCKS(total);
    k_pixel_shuffle_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        in, out, C_out, H_in, W_in, H_out, W_out, r);
}

void pixel_shuffle_backward(const float* grad_out, float* grad_in, size_t C_in, size_t H_in, size_t W_in, size_t r, void* stream) {
    if (C_in == 0 || H_in == 0 || W_in == 0 || r == 0 || !grad_out || !grad_in) return;
    size_t C_out = C_in / (r * r);
    size_t H_out = H_in * r;
    size_t W_out = W_in * r;
    int64_t total = static_cast<int64_t>(C_out * H_out * W_out);
    int blocks = GET_BLOCKS(total);
    k_pixel_shuffle_backward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        grad_out, grad_in, C_out, H_in, W_in, H_out, W_out, r);
}

} // namespace soar::cuda::kernels
