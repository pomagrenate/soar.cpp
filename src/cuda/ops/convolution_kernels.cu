#include <soar/cuda/cuda_kernels.hpp>
#include <soar/cuda/cuda_common.cuh>
#include <soar/cuda/im2col.cuh>

namespace soar::cuda::kernels {

// =============================================================================
// 1. Pointwise Conv2D (1x1) Fast Path
// =============================================================================
__global__ void k_conv2d_1x1_forward(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t C_in,
    int64_t C_out,
    int64_t HW) {
    int64_t total = C_out * HW;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t co = idx / HW;
        int64_t hw = idx % HW;

        float sum = bias ? bias[co] : 0.0f;
        const float* w_row = weight + co * C_in;

        for (int64_t ci = 0; ci < C_in; ++ci) {
            sum += w_row[ci] * in[ci * HW + hw];
        }
        out[idx] = sum;
    }
}

void conv2d_1x1_forward(const float* in, const float* weight, const float* bias, float* out,
                        size_t C_in, size_t C_out, size_t HW, void* stream) {
    int64_t total = static_cast<int64_t>(C_out * HW);
    if (total == 0 || !in || !weight || !out) return;
    int blocks = GET_BLOCKS(total);
    k_conv2d_1x1_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        in, weight, bias, out, static_cast<int64_t>(C_in), static_cast<int64_t>(C_out), static_cast<int64_t>(HW));
}

__global__ void k_conv2d_1x1_backward_input(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_in,
    int64_t C_in,
    int64_t C_out,
    int64_t HW) {
    int64_t total = C_in * HW;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t ci = idx / HW;
        int64_t hw = idx % HW;

        float sum = 0.0f;
        for (int64_t co = 0; co < C_out; ++co) {
            sum += weight[co * C_in + ci] * grad_out[co * HW + hw];
        }
        grad_in[idx] += sum;
    }
}

__global__ void k_conv2d_1x1_backward_weight(
    const float* __restrict__ grad_out,
    const float* __restrict__ in,
    float* __restrict__ grad_weight,
    int64_t C_in,
    int64_t C_out,
    int64_t HW) {
    int64_t pair_idx = blockIdx.x;
    if (pair_idx >= C_out * C_in) return;

    int64_t co = pair_idx / C_in;
    int64_t ci = pair_idx % C_in;

    const float* go_row = grad_out + co * HW;
    const float* in_row = in + ci * HW;

    float sum_w = 0.0f;
    for (int64_t hw = threadIdx.x; hw < HW; hw += blockDim.x) {
        sum_w += go_row[hw] * in_row[hw];
    }
    sum_w = block_reduce_sum(sum_w);
    if (threadIdx.x == 0) {
        grad_weight[pair_idx] += sum_w;
    }
}

__global__ void k_conv2d_1x1_backward_bias(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    int64_t C_out,
    int64_t HW) {
    int64_t co = blockIdx.x;
    if (co >= C_out) return;

    const float* go_row = grad_out + co * HW;
    float sum_b = 0.0f;
    for (int64_t hw = threadIdx.x; hw < HW; hw += blockDim.x) {
        sum_b += go_row[hw];
    }
    sum_b = block_reduce_sum(sum_b);
    if (threadIdx.x == 0) {
        grad_bias[co] += sum_b;
    }
}

void conv2d_1x1_backward(const float* grad_out, const float* in, const float* weight,
                         float* grad_in, float* grad_weight, float* grad_bias,
                         size_t C_in, size_t C_out, size_t HW, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int64_t cin = static_cast<int64_t>(C_in);
    int64_t cout = static_cast<int64_t>(C_out);
    int64_t hw = static_cast<int64_t>(HW);

    if (grad_in && weight) {
        int64_t total_in = cin * hw;
        int blocks = GET_BLOCKS(total_in);
        k_conv2d_1x1_backward_input<<<blocks, CUDA_NUM_THREADS, 0, s>>>(
            grad_out, weight, grad_in, cin, cout, hw);
    }
    if (grad_weight && in) {
        int64_t total_w = cout * cin;
        if (total_w > 0) {
            k_conv2d_1x1_backward_weight<<<static_cast<unsigned int>(total_w), 256, 0, s>>>(
                grad_out, in, grad_weight, cin, cout, hw);
        }
    }
    if (grad_bias && grad_out) {
        if (cout > 0) {
            k_conv2d_1x1_backward_bias<<<static_cast<unsigned int>(cout), 256, 0, s>>>(
                grad_out, grad_bias, cout, hw);
        }
    }
}

// =============================================================================
// 2. Depthwise Separable Conv2D
// =============================================================================
__global__ void k_conv2d_dw_forward(
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t K,
    int64_t pad,
    int64_t stride,
    int64_t dil) {
    int64_t total = C * H_out * W_out;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t w_out = idx % W_out;
        int64_t temp = idx / W_out;
        int64_t h_out = temp % H_out;
        int64_t c = temp / H_out;

        float sum = bias ? bias[c] : 0.0f;
        const float* w = weight + c * (K * K);
        const float* in_c = in + c * (H_in * W_in);

        int64_t h_in_base = h_out * stride - pad;
        int64_t w_in_base = w_out * stride - pad;

        for (int64_t kh = 0; kh < K; ++kh) {
            int64_t hi = h_in_base + kh * dil;
            if (hi >= 0 && hi < H_in) {
                for (int64_t kw = 0; kw < K; ++kw) {
                    int64_t wi = w_in_base + kw * dil;
                    if (wi >= 0 && wi < W_in) {
                        sum += in_c[hi * W_in + wi] * w[kh * K + kw];
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
    int64_t total = static_cast<int64_t>(C * H_out * W_out);
    if (total == 0 || !in || !weight || !out) return;
    int blocks = GET_BLOCKS(total);
    k_conv2d_dw_forward<<<blocks, CUDA_NUM_THREADS, 0, static_cast<cudaStream_t>(stream)>>>(
        in, weight, bias, out,
        static_cast<int64_t>(C), static_cast<int64_t>(H_in), static_cast<int64_t>(W_in),
        static_cast<int64_t>(H_out), static_cast<int64_t>(W_out),
        static_cast<int64_t>(K), static_cast<int64_t>(pad), static_cast<int64_t>(stride), static_cast<int64_t>(dil));
}

__global__ void k_conv2d_dw_backward_input(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_in,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t K,
    int64_t pad,
    int64_t stride,
    int64_t dil) {
    int64_t total = C * H_in * W_in;
    CUDA_KERNEL_LOOP(idx, total) {
        int64_t wi = idx % W_in;
        int64_t temp = idx / W_in;
        int64_t hi = temp % H_in;
        int64_t c = temp / H_in;

        float sum = 0.0f;
        const float* w = weight + c * (K * K);
        const float* go = grad_out + c * (H_out * W_out);

        for (int64_t kh = 0; kh < K; ++kh) {
            int64_t num_h = hi + pad - kh * dil;
            if (num_h >= 0 && (num_h % stride == 0)) {
                int64_t ho = num_h / stride;
                if (ho < H_out) {
                    for (int64_t kw = 0; kw < K; ++kw) {
                        int64_t num_w = wi + pad - kw * dil;
                        if (num_w >= 0 && (num_w % stride == 0)) {
                            int64_t wo = num_w / stride;
                            if (wo < W_out) {
                                sum += go[ho * W_out + wo] * w[kh * K + kw];
                            }
                        }
                    }
                }
            }
        }
        grad_in[idx] += sum;
    }
}

__global__ void k_conv2d_dw_backward_weight(
    const float* __restrict__ grad_out,
    const float* __restrict__ in,
    float* __restrict__ grad_weight,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t K,
    int64_t pad,
    int64_t stride,
    int64_t dil) {
    int64_t pair_idx = blockIdx.x;
    if (pair_idx >= C * K * K) return;

    int64_t kw = pair_idx % K;
    int64_t temp = pair_idx / K;
    int64_t kh = temp % K;
    int64_t c = temp / K;

    int64_t HW_out = H_out * W_out;
    const float* go = grad_out + c * HW_out;
    const float* inp = in + c * (H_in * W_in);

    float sum_w = 0.0f;
    for (int64_t idx = threadIdx.x; idx < HW_out; idx += blockDim.x) {
        int64_t ho = idx / W_out;
        int64_t wo = idx % W_out;
        int64_t hi = ho * stride - pad + kh * dil;
        int64_t wi = wo * stride - pad + kw * dil;
        if (hi >= 0 && hi < H_in && wi >= 0 && wi < W_in) {
            sum_w += go[idx] * inp[hi * W_in + wi];
        }
    }
    sum_w = block_reduce_sum(sum_w);
    if (threadIdx.x == 0) {
        grad_weight[pair_idx] += sum_w;
    }
}

__global__ void k_conv2d_dw_backward_bias(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    int64_t C,
    int64_t HW_out) {
    int64_t c = blockIdx.x;
    if (c >= C) return;

    const float* go = grad_out + c * HW_out;
    float sum_b = 0.0f;
    for (int64_t idx = threadIdx.x; idx < HW_out; idx += blockDim.x) {
        sum_b += go[idx];
    }
    sum_b = block_reduce_sum(sum_b);
    if (threadIdx.x == 0) {
        grad_bias[c] += sum_b;
    }
}

void conv2d_dw_backward(const float* grad_out, const float* in, const float* weight,
                        float* grad_in, float* grad_weight, float* grad_bias,
                        size_t C, size_t H_in, size_t W_in, size_t H_out, size_t W_out,
                        size_t K, size_t pad, size_t stride, size_t dil, void* stream) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int64_t c = static_cast<int64_t>(C);
    int64_t hin = static_cast<int64_t>(H_in);
    int64_t win = static_cast<int64_t>(W_in);
    int64_t hout = static_cast<int64_t>(H_out);
    int64_t wout = static_cast<int64_t>(W_out);
    int64_t k = static_cast<int64_t>(K);
    int64_t p = static_cast<int64_t>(pad);
    int64_t st = static_cast<int64_t>(stride);
    int64_t d = static_cast<int64_t>(dil);

    if (grad_in && weight) {
        int64_t total_in = c * hin * win;
        int blocks = GET_BLOCKS(total_in);
        k_conv2d_dw_backward_input<<<blocks, CUDA_NUM_THREADS, 0, s>>>(
            grad_out, weight, grad_in, c, hin, win, hout, wout, k, p, st, d);
    }
    if (grad_weight && in) {
        int64_t total_w = c * k * k;
        if (total_w > 0) {
            k_conv2d_dw_backward_weight<<<static_cast<unsigned int>(total_w), 256, 0, s>>>(
                grad_out, in, grad_weight, c, hin, win, hout, wout, k, p, st, d);
        }
    }
    if (grad_bias && grad_out) {
        if (c > 0) {
            k_conv2d_dw_backward_bias<<<static_cast<unsigned int>(c), 256, 0, s>>>(
                grad_out, grad_bias, c, hout * wout);
        }
    }
}

// =============================================================================
// 3. General Grouped Conv2D
// =============================================================================
void conv2d_forward(const float* in, const float* weight, const float* bias, float* out,
                    size_t C_in, size_t C_out, size_t H_in, size_t W_in,
                    size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, size_t dil, size_t groups, void* stream) {
    if (groups == 1 && K == 1 && pad == 0 && stride == 1 && dil == 1) {
        conv2d_1x1_forward(in, weight, bias, out, C_in, C_out, H_in * W_in, stream);
        return;
    }
    if (groups == C_in && groups == C_out) {
        conv2d_dw_forward(in, weight, bias, out, C_in, H_in, W_in, H_out, W_out, K, pad, stride, dil, stream);
        return;
    }

    // PyTorch-style im2col for grouped general conv
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    size_t in_group_c = C_in / groups;
    size_t out_group_c = C_out / groups;
    size_t HW_out = H_out * W_out;
    size_t col_size = in_group_c * K * K * HW_out;

    float* d_columns = nullptr;
    cudaMallocAsync(&d_columns, col_size * sizeof(float), s);

    for (size_t g = 0; g < groups; ++g) {
        const float* in_g = in + g * in_group_c * H_in * W_in;
        const float* w_g = weight + g * out_group_c * in_group_c * K * K;
        const float* b_g = bias ? (bias + g * out_group_c) : nullptr;
        float* out_g = out + g * out_group_c * HW_out;

        im2col<float>(s, in_g, in_group_c, H_in, W_in, H_out, W_out, K, K, pad, pad, stride, stride, dil, dil, d_columns);
        conv2d_1x1_forward(d_columns, w_g, b_g, out_g, in_group_c * K * K, out_group_c, HW_out, s);
    }
    cudaFreeAsync(d_columns, s);
}

void conv2d_backward(const float* grad_out, const float* in, const float* weight,
                     float* grad_in, float* grad_weight, float* grad_bias,
                     size_t C_in, size_t C_out, size_t H_in, size_t W_in,
                     size_t H_out, size_t W_out, size_t K, size_t pad, size_t stride, size_t dil, size_t groups, void* stream) {
    if (groups == 1 && K == 1 && pad == 0 && stride == 1 && dil == 1) {
        conv2d_1x1_backward(grad_out, in, weight, grad_in, grad_weight, grad_bias, C_in, C_out, H_in * W_in, stream);
        return;
    }
    if (groups == C_in && groups == C_out) {
        conv2d_dw_backward(grad_out, in, weight, grad_in, grad_weight, grad_bias,
                           C_in, H_in, W_in, H_out, W_out, K, pad, stride, dil, stream);
        return;
    }

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    size_t in_group_c = C_in / groups;
    size_t out_group_c = C_out / groups;
    size_t HW_out = H_out * W_out;
    size_t col_size = in_group_c * K * K * HW_out;

    float* d_columns = nullptr;
    float* d_grad_columns = nullptr;
    cudaMallocAsync(&d_columns, col_size * sizeof(float), s);
    if (grad_in) {
        cudaMallocAsync(&d_grad_columns, col_size * sizeof(float), s);
    }

    for (size_t g = 0; g < groups; ++g) {
        const float* go_g = grad_out + g * out_group_c * HW_out;
        const float* in_g = in + g * in_group_c * H_in * W_in;
        const float* w_g = weight + g * out_group_c * in_group_c * K * K;
        float* gi_g = grad_in ? (grad_in + g * in_group_c * H_in * W_in) : nullptr;
        float* gw_g = grad_weight ? (grad_weight + g * out_group_c * in_group_c * K * K) : nullptr;
        float* gb_g = grad_bias ? (grad_bias + g * out_group_c) : nullptr;

        im2col<float>(s, in_g, in_group_c, H_in, W_in, H_out, W_out, K, K, pad, pad, stride, stride, dil, dil, d_columns);

        if (d_grad_columns) cuda_zero(d_grad_columns, col_size, s);

        conv2d_1x1_backward(go_g, d_columns, w_g, d_grad_columns, gw_g, gb_g,
                            in_group_c * K * K, out_group_c, HW_out, s);

        if (gi_g) {
            col2im<float, float>(s, d_grad_columns, in_group_c, H_in, W_in, H_out, W_out,
                                 K, K, pad, pad, stride, stride, dil, dil, gi_g);
        }
    }

    cudaFreeAsync(d_columns, s);
    if (d_grad_columns) cudaFreeAsync(d_grad_columns, s);
}

} // namespace soar::cuda::kernels
