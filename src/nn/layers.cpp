#include <soar/nn/layers.hpp>
#include <soar/autograd/node.hpp>
#include <soar/core/logging.hpp>

#include <random>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace soar::nn {

// -------------------------------------------------------------
// Conv2d Autograd Node
// -------------------------------------------------------------
struct Conv2dNode : public AutogradNode {
    TensorPtr input;
    TensorPtr weight;
    TensorPtr bias;
    size_t stride;
    size_t padding;
    size_t dilation;
    size_t groups;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        std::vector<std::shared_ptr<AutogradNode>> res;
        if (input && input->grad_fn()) res.push_back(input->grad_fn());
        if (weight && weight->grad_fn()) res.push_back(weight->grad_fn());
        if (bias && bias->grad_fn()) res.push_back(bias->grad_fn());
        return res;
    }

    void backward(const TensorPtr& grad_output) override {
        const float* go = grad_output->data();
        const float* x = input->data();
        const float* w = weight->data();

        [[maybe_unused]] size_t B = 1;
        size_t C_in = input->dim(0);
        size_t H_in = input->dim(1);
        size_t W_in = input->dim(2);

        size_t C_out = weight->dim(0);
        size_t K = weight->dim(2);
        size_t H_out = grad_output->dim(1);
        size_t W_out = grad_output->dim(2);

        // 1. Bias gradient: sum over spatial output
        if (bias && bias->requires_grad()) {
            TensorPtr grad_b = Tensor::zeros(bias->shape());
            float* gb = grad_b->data();
            for (size_t co = 0; co < C_out; ++co) {
                float sum = 0.0f;
                for (size_t sp = 0; sp < H_out * W_out; ++sp) {
                    sum += go[co * (H_out * W_out) + sp];
                }
                gb[co] = sum;
            }
            propagate_grad(bias, grad_b);
        }

        // 2. Weight gradient: dW = dY * X^T
        if (weight->requires_grad()) {
            TensorPtr grad_w = Tensor::zeros(weight->shape());
            float* gw = grad_w->data();

            if (groups == 1 && K == 1 && stride == 1 && padding == 0) {
                // Pointwise 1x1
                #pragma omp parallel for
                for (size_t co = 0; co < C_out; ++co) {
                    for (size_t ci = 0; ci < C_in; ++ci) {
                        float sum = 0.0f;
                        for (size_t sp = 0; sp < H_out * W_out; ++sp) {
                            sum += go[co * (H_out * W_out) + sp] * x[ci * (H_out * W_out) + sp];
                        }
                        gw[co * C_in + ci] = sum;
                    }
                }
            } else if (groups == C_in && C_in == C_out) {
                // Depthwise
                int pad = static_cast<int>(padding);
                int dil = static_cast<int>(dilation);
                int str = static_cast<int>(stride);

                #pragma omp parallel for
                for (size_t c = 0; c < C_in; ++c) {
                    for (size_t ky = 0; ky < K; ++ky) {
                        for (size_t kx = 0; kx < K; ++kx) {
                            float sum = 0.0f;
                            for (size_t yo = 0; yo < H_out; ++yo) {
                                int yi = static_cast<int>(yo * str) - pad + static_cast<int>(ky * dil);
                                if (yi < 0 || yi >= static_cast<int>(H_in)) continue;
                                for (size_t xo = 0; xo < W_out; ++xo) {
                                    int xi = static_cast<int>(xo * str) - pad + static_cast<int>(kx * dil);
                                    if (xi < 0 || xi >= static_cast<int>(W_in)) continue;

                                    float val_x = x[c * (H_in * W_in) + yi * W_in + xi];
                                    float val_go = go[c * (H_out * W_out) + yo * W_out + xo];
                                    sum += val_x * val_go;
                                }
                            }
                            gw[c * (K * K) + ky * K + kx] = sum;
                        }
                    }
                }
            } else {
                // General Convolution
                int pad = static_cast<int>(padding);
                int dil = static_cast<int>(dilation);
                int str = static_cast<int>(stride);
                size_t c_in_per_group = C_in / groups;
                size_t c_out_per_group = C_out / groups;

                #pragma omp parallel for collapse(2)
                for (size_t g = 0; g < groups; ++g) {
                    for (size_t co_rel = 0; co_rel < c_out_per_group; ++co_rel) {
                        size_t co = g * c_out_per_group + co_rel;
                        for (size_t ci_rel = 0; ci_rel < c_in_per_group; ++ci_rel) {
                            size_t ci = g * c_in_per_group + ci_rel;
                            for (size_t ky = 0; ky < K; ++ky) {
                                for (size_t kx = 0; kx < K; ++kx) {
                                    float sum = 0.0f;
                                    for (size_t yo = 0; yo < H_out; ++yo) {
                                        int yi = static_cast<int>(yo * str) - pad + static_cast<int>(ky * dil);
                                        if (yi < 0 || yi >= static_cast<int>(H_in)) continue;
                                        for (size_t xo = 0; xo < W_out; ++xo) {
                                            int xi = static_cast<int>(xo * str) - pad + static_cast<int>(kx * dil);
                                            if (xi < 0 || xi >= static_cast<int>(W_in)) continue;
                                            float val_x = x[ci * (H_in * W_in) + yi * W_in + xi];
                                            float val_go = go[co * (H_out * W_out) + yo * W_out + xo];
                                            sum += val_x * val_go;
                                        }
                                    }
                                    size_t w_idx = co * (c_in_per_group * K * K) + ci_rel * (K * K) + ky * K + kx;
                                    gw[w_idx] = sum;
                                }
                            }
                        }
                    }
                }
            }
            propagate_grad(weight, grad_w);
        }

        // 3. Input gradient: dX = dY * W^T
        if (input->requires_grad()) {
            TensorPtr grad_x = Tensor::zeros(input->shape());
            float* gx = grad_x->data();

            if (groups == 1 && K == 1 && stride == 1 && padding == 0) {
                #pragma omp parallel for
                for (size_t ci = 0; ci < C_in; ++ci) {
                    for (size_t co = 0; co < C_out; ++co) {
                        float w_val = w[co * C_in + ci];
                        for (size_t sp = 0; sp < H_out * W_out; ++sp) {
                            gx[ci * (H_out * W_out) + sp] += go[co * (H_out * W_out) + sp] * w_val;
                        }
                    }
                }
            } else if (groups == C_in && C_in == C_out) {
                int pad = static_cast<int>(padding);
                int dil = static_cast<int>(dilation);
                int str = static_cast<int>(stride);

                #pragma omp parallel for
                for (size_t c = 0; c < C_in; ++c) {
                    for (size_t yo = 0; yo < H_out; ++yo) {
                        for (size_t xo = 0; xo < W_out; ++xo) {
                            float go_val = go[c * (H_out * W_out) + yo * W_out + xo];
                            for (size_t ky = 0; ky < K; ++ky) {
                                int yi = static_cast<int>(yo * str) - pad + static_cast<int>(ky * dil);
                                if (yi < 0 || yi >= static_cast<int>(H_in)) continue;
                                for (size_t kx = 0; kx < K; ++kx) {
                                    int xi = static_cast<int>(xo * str) - pad + static_cast<int>(kx * dil);
                                    if (xi < 0 || xi >= static_cast<int>(W_in)) continue;

                                    float w_val = w[c * (K * K) + ky * K + kx];
                                    gx[c * (H_in * W_in) + yi * W_in + xi] += go_val * w_val;
                                }
                            }
                        }
                    }
                }
            } else {
                int pad = static_cast<int>(padding);
                int dil = static_cast<int>(dilation);
                int str = static_cast<int>(stride);
                size_t c_in_per_group = C_in / groups;
                size_t c_out_per_group = C_out / groups;

                #pragma omp parallel for collapse(2)
                for (size_t g = 0; g < groups; ++g) {
                    for (size_t ci_rel = 0; ci_rel < c_in_per_group; ++ci_rel) {
                        size_t ci = g * c_in_per_group + ci_rel;
                        for (size_t co_rel = 0; co_rel < c_out_per_group; ++co_rel) {
                            size_t co = g * c_out_per_group + co_rel;
                            for (size_t ky = 0; ky < K; ++ky) {
                                for (size_t kx = 0; kx < K; ++kx) {
                                    size_t w_idx = co * (c_in_per_group * K * K) + ci_rel * (K * K) + ky * K + kx;
                                    float w_val = w[w_idx];

                                    for (size_t yo = 0; yo < H_out; ++yo) {
                                        int yi = static_cast<int>(yo * str) - pad + static_cast<int>(ky * dil);
                                        if (yi < 0 || yi >= static_cast<int>(H_in)) continue;
                                        for (size_t xo = 0; xo < W_out; ++xo) {
                                            int xi = static_cast<int>(xo * str) - pad + static_cast<int>(kx * dil);
                                            if (xi < 0 || xi >= static_cast<int>(W_in)) continue;
                                            float go_val = go[co * (H_out * W_out) + yo * W_out + xo];
                                            gx[ci * (H_in * W_in) + yi * W_in + xi] += go_val * w_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            propagate_grad(input, grad_x);
        }
    }

    void release_variables() override {
        input = nullptr;
    }
};

// -------------------------------------------------------------
// Conv2d Implementation
// -------------------------------------------------------------
Conv2d::Conv2d(size_t in_channels, size_t out_channels, size_t kernel_size,
               size_t stride, int padding, size_t dilation,
               size_t groups, bool bias)
    : Module("Conv2d"), in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride),
      padding_(padding < 0 ? autopad(static_cast<int>(kernel_size), -1, static_cast<int>(dilation)) : padding),
      dilation_(dilation), groups_(groups), has_bias_(bias) {

    size_t weight_c_in = in_channels / groups;
    weight_ = Tensor::create({static_cast<int64_t>(out_channels),
                               static_cast<int64_t>(weight_c_in),
                               static_cast<int64_t>(kernel_size),
                               static_cast<int64_t>(kernel_size)}, true);

    // Kaiming uniform initialization
    float fan_in = static_cast<float>(weight_c_in * kernel_size * kernel_size);
    float bound = std::sqrt(1.0f / fan_in);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-bound, bound);
    for (size_t i = 0; i < weight_->numel(); ++i) {
        weight_->data()[i] = dist(gen);
    }
    register_parameter("weight", weight_);

    if (has_bias_) {
        bias_ = Tensor::create({static_cast<int64_t>(out_channels)}, true);
        bias_->zero_();
        register_parameter("bias", bias_);
    }
}

TensorPtr Conv2d::forward(const TensorPtr& input) {
    size_t H_in = input->dim(1);
    size_t W_in = input->dim(2);

    size_t effective_k = dilation_ * (kernel_size_ - 1) + 1;
    size_t H_out = (H_in + 2 * padding_ - effective_k) / stride_ + 1;
    size_t W_out = (W_in + 2 * padding_ - effective_k) / stride_ + 1;

    TensorPtr output = Tensor::create({static_cast<int64_t>(out_channels_),
                                       static_cast<int64_t>(H_out),
                                       static_cast<int64_t>(W_out)},
                                      input->requires_grad() || weight_->requires_grad());

    const float* x = input->data();
    const float* w = weight_->data();
    const float* b = has_bias_ ? bias_->data() : nullptr;
    float* y = output->data();

    if (groups_ == 1 && kernel_size_ == 1 && stride_ == 1 && padding_ == 0) {
        // Pointwise 1x1 Conv
        #pragma omp parallel for
        for (size_t co = 0; co < out_channels_; ++co) {
            float bias_val = b ? b[co] : 0.0f;
            for (size_t sp = 0; sp < H_out * W_out; ++sp) {
                float sum = bias_val;
                for (size_t ci = 0; ci < in_channels_; ++ci) {
                    sum += x[ci * (H_out * W_out) + sp] * w[co * in_channels_ + ci];
                }
                y[co * (H_out * W_out) + sp] = sum;
            }
        }
    } else if (groups_ == in_channels_ && in_channels_ == out_channels_) {
        // Depthwise Conv
        int pad = static_cast<int>(padding_);
        int dil = static_cast<int>(dilation_);
        int str = static_cast<int>(stride_);
        int K = static_cast<int>(kernel_size_);

        #pragma omp parallel for
        for (size_t c = 0; c < in_channels_; ++c) {
            float bias_val = b ? b[c] : 0.0f;
            for (size_t yo = 0; yo < H_out; ++yo) {
                int base_yi = static_cast<int>(yo * str) - pad;
                for (size_t xo = 0; xo < W_out; ++xo) {
                    int base_xi = static_cast<int>(xo * str) - pad;
                    float sum = bias_val;
                    int w_idx = 0;
                    for (int ky = 0; ky < K; ++ky) {
                        int yi = base_yi + ky * dil;
                        bool valid_y = (yi >= 0 && yi < static_cast<int>(H_in));
                        for (int kx = 0; kx < K; ++kx) {
                            int xi = base_xi + kx * dil;
                            if (valid_y && xi >= 0 && xi < static_cast<int>(W_in)) {
                                sum += x[c * (H_in * W_in) + yi * W_in + xi] * w[c * (K * K) + w_idx];
                            }
                            w_idx++;
                        }
                    }
                    y[c * (H_out * W_out) + yo * W_out + xo] = sum;
                }
            }
        }
    } else {
        // Standard General Conv
        int pad = static_cast<int>(padding_);
        int dil = static_cast<int>(dilation_);
        int str = static_cast<int>(stride_);
        int K = static_cast<int>(kernel_size_);

        #pragma omp parallel for
        for (size_t co = 0; co < out_channels_; ++co) {
            float bias_val = b ? b[co] : 0.0f;
            for (size_t yo = 0; yo < H_out; ++yo) {
                int base_yi = static_cast<int>(yo * str) - pad;
                for (size_t xo = 0; xo < W_out; ++xo) {
                    int base_xi = static_cast<int>(xo * str) - pad;
                    float sum = bias_val;
                    for (size_t ci = 0; ci < in_channels_; ++ci) {
                        int w_idx = 0;
                        for (int ky = 0; ky < K; ++ky) {
                            int yi = base_yi + ky * dil;
                            bool valid_y = (yi >= 0 && yi < static_cast<int>(H_in));
                            for (int kx = 0; kx < K; ++kx) {
                                int xi = base_xi + kx * dil;
                                if (valid_y && xi >= 0 && xi < static_cast<int>(W_in)) {
                                    sum += x[ci * (H_in * W_in) + yi * W_in + xi] *
                                           w[co * (in_channels_ * K * K) + ci * (K * K) + w_idx];
                                }
                                w_idx++;
                            }
                        }
                    }
                    y[co * (H_out * W_out) + yo * W_out + xo] = sum;
                }
            }
        }
    }

    if (input->requires_grad() || weight_->requires_grad()) {
        auto node = std::make_shared<Conv2dNode>();
        node->input = input;
        node->weight = weight_;
        node->bias = bias_;
        node->stride = stride_;
        node->padding = padding_;
        node->dilation = dilation_;
        node->groups = groups_;
        output->set_grad_fn(node);
    }

    return output;
}

// -------------------------------------------------------------
// GroupNorm Autograd Node & Layer
// -------------------------------------------------------------
struct GroupNormNode : public AutogradNode {
    TensorPtr input;
    TensorPtr weight;
    TensorPtr bias;
    std::vector<float> saved_mean;
    std::vector<float> saved_rstd;
    size_t num_groups;
    float eps;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        std::vector<std::shared_ptr<AutogradNode>> res;
        if (input && input->grad_fn()) res.push_back(input->grad_fn());
        if (weight && weight->grad_fn()) res.push_back(weight->grad_fn());
        if (bias && bias->grad_fn()) res.push_back(bias->grad_fn());
        return res;
    }

    void backward(const TensorPtr& grad_output) override {
        const float* go = grad_output->data();
        const float* x = input->data();
        const float* g = weight ? weight->data() : nullptr;

        size_t C = input->dim(0);
        size_t H = input->dim(1);
        size_t W = input->dim(2);
        size_t c_per_g = C / num_groups;
        size_t M = c_per_g * H * W;

        // 1. Weight & Bias gradients
        if (weight && weight->requires_grad()) {
            TensorPtr grad_w = Tensor::zeros(weight->shape());
            TensorPtr grad_b = Tensor::zeros(bias->shape());
            float* gw = grad_w->data();
            float* gb = grad_b->data();

            for (size_t grp = 0; grp < num_groups; ++grp) {
                float mean = saved_mean[grp];
                float rstd = saved_rstd[grp];
                for (size_t cg = 0; cg < c_per_g; ++cg) {
                    size_t c = grp * c_per_g + cg;
                    float sum_w = 0.0f;
                    float sum_b = 0.0f;
                    for (size_t sp = 0; sp < H * W; ++sp) {
                        float go_val = go[c * (H * W) + sp];
                        float x_val = x[c * (H * W) + sp];
                        sum_w += go_val * (x_val - mean) * rstd;
                        sum_b += go_val;
                    }
                    gw[c] = sum_w;
                    gb[c] = sum_b;
                }
            }
            propagate_grad(weight, grad_w);
            propagate_grad(bias, grad_b);
        }

        // 2. Input gradient
        if (input->requires_grad()) {
            TensorPtr grad_x = Tensor::zeros(input->shape());
            float* gx = grad_x->data();

            for (size_t grp = 0; grp < num_groups; ++grp) {
                float mean = saved_mean[grp];
                float rstd = saved_rstd[grp];

                float sum_dy = 0.0f;
                float sum_dy_xhat = 0.0f;

                for (size_t cg = 0; cg < c_per_g; ++cg) {
                    size_t c = grp * c_per_g + cg;
                    float g_val = g ? g[c] : 1.0f;
                    for (size_t sp = 0; sp < H * W; ++sp) {
                        float dy = go[c * (H * W) + sp] * g_val;
                        float xhat = (x[c * (H * W) + sp] - mean) * rstd;
                        sum_dy += dy;
                        sum_dy_xhat += dy * xhat;
                    }
                }

                for (size_t cg = 0; cg < c_per_g; ++cg) {
                    size_t c = grp * c_per_g + cg;
                    float g_val = g ? g[c] : 1.0f;
                    for (size_t sp = 0; sp < H * W; ++sp) {
                        float dy = go[c * (H * W) + sp] * g_val;
                        float xhat = (x[c * (H * W) + sp] - mean) * rstd;
                        gx[c * (H * W) + sp] = (rstd / float(M)) * (float(M) * dy - sum_dy - xhat * sum_dy_xhat);
                    }
                }
            }
            propagate_grad(input, grad_x);
        }
    }

    void release_variables() override {
        input = nullptr;
        saved_mean.clear();
        saved_mean.shrink_to_fit();
        saved_rstd.clear();
        saved_rstd.shrink_to_fit();
    }
};

GroupNorm::GroupNorm(size_t num_groups, size_t num_channels, float eps, bool affine)
    : Module("GroupNorm"), num_groups_(num_groups), num_channels_(num_channels), eps_(eps), affine_(affine) {
    if (affine_) {
        weight_ = Tensor::ones({static_cast<int64_t>(num_channels)}, true);
        bias_ = Tensor::zeros({static_cast<int64_t>(num_channels)}, true);
        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }
}

TensorPtr GroupNorm::forward(const TensorPtr& input) {
    size_t C = input->dim(0);
    size_t H = input->dim(1);
    size_t W = input->dim(2);
    size_t c_per_g = C / num_groups_;
    size_t M = c_per_g * H * W;

    TensorPtr output = Tensor::create(input->shape(), input->requires_grad() || (affine_ && weight_->requires_grad()));
    const float* x = input->data();
    const float* g = affine_ ? weight_->data() : nullptr;
    const float* b = affine_ ? bias_->data() : nullptr;
    float* y = output->data();

    std::vector<float> means(num_groups_);
    std::vector<float> rstds(num_groups_);

    for (size_t grp = 0; grp < num_groups_; ++grp) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (size_t cg = 0; cg < c_per_g; ++cg) {
            size_t c = grp * c_per_g + cg;
            for (size_t sp = 0; sp < H * W; ++sp) {
                float val = x[c * (H * W) + sp];
                sum += val;
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }
        }
        float mean = static_cast<float>(sum / double(M));
        float mean_sq = static_cast<float>(sum_sq / double(M));
        float var = std::max(mean_sq - mean * mean, 0.0f);
        float rstd = 1.0f / std::sqrt(var + eps_);

        means[grp] = mean;
        rstds[grp] = rstd;

        for (size_t cg = 0; cg < c_per_g; ++cg) {
            size_t c = grp * c_per_g + cg;
            float g_val = g ? g[c] : 1.0f;
            float b_val = b ? b[c] : 0.0f;
            for (size_t sp = 0; sp < H * W; ++sp) {
                float val = x[c * (H * W) + sp];
                y[c * (H * W) + sp] = g_val * (val - mean) * rstd + b_val;
            }
        }
    }

    if (output->requires_grad()) {
        auto node = std::make_shared<GroupNormNode>();
        node->input = input;
        node->weight = weight_;
        node->bias = bias_;
        node->saved_mean = std::move(means);
        node->saved_rstd = std::move(rstds);
        node->num_groups = num_groups_;
        node->eps = eps_;
        output->set_grad_fn(node);
    }

    return output;
}

// -------------------------------------------------------------
// SiLU Implementation
// -------------------------------------------------------------
struct SiLUNode : public AutogradNode {
    TensorPtr input;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_x = Tensor::zeros(input->shape());
        const float* x = input->data();
        const float* go = grad_output->data();
        float* gx = grad_x->data();
        size_t n = input->numel();
        for (size_t i = 0; i < n; ++i) {
            float sig = 1.0f / (1.0f + std::exp(-x[i]));
            float dsilu = sig * (1.0f + x[i] * (1.0f - sig));
            gx[i] = go[i] * dsilu;
        }
        propagate_grad(input, grad_x);
    }

    void release_variables() override {
        input = nullptr;
    }
};

TensorPtr SiLU::forward(const TensorPtr& input) {
    TensorPtr output = Tensor::create(input->shape(), input->requires_grad());
    const float* x = input->data();
    float* y = output->data();
    size_t n = input->numel();
    for (size_t i = 0; i < n; ++i) {
        y[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
    if (input->requires_grad()) {
        auto node = std::make_shared<SiLUNode>();
        node->input = input;
        output->set_grad_fn(node);
    }
    return output;
}

// -------------------------------------------------------------
// Sigmoid Implementation
// -------------------------------------------------------------
struct SigmoidNode : public AutogradNode {
    TensorPtr output;
    TensorPtr input;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_x = Tensor::zeros(input->shape());
        const float* y = output->data();
        const float* go = grad_output->data();
        float* gx = grad_x->data();
        size_t n = input->numel();
        for (size_t i = 0; i < n; ++i) {
            gx[i] = go[i] * y[i] * (1.0f - y[i]);
        }
        propagate_grad(input, grad_x);
    }

    void release_variables() override {
        input = nullptr;
        output = nullptr;
    }
};

TensorPtr Sigmoid::forward(const TensorPtr& input) {
    TensorPtr output = Tensor::create(input->shape(), input->requires_grad());
    const float* x = input->data();
    float* y = output->data();
    size_t n = input->numel();
    for (size_t i = 0; i < n; ++i) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
    if (input->requires_grad()) {
        auto node = std::make_shared<SigmoidNode>();
        node->input = input;
        node->output = output;
        output->set_grad_fn(node);
    }
    return output;
}

// -------------------------------------------------------------
// PixelShuffle Implementation
// -------------------------------------------------------------
struct PixelShuffleNode : public AutogradNode {
    TensorPtr input;
    size_t r;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_x = Tensor::zeros(input->shape());
        const float* go = grad_output->data();
        float* gx = grad_x->data();

        size_t C_out = grad_output->dim(0);
        size_t H_out = grad_output->dim(1);
        size_t W_out = grad_output->dim(2);
        size_t H_in = input->dim(1);
        size_t W_in = input->dim(2);

        for (size_t c = 0; c < C_out; ++c) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    size_t in_y = y / r;
                    size_t in_x = x / r;
                    size_t sub_y = y % r;
                    size_t sub_x = x % r;
                    size_t c_in = c * (r * r) + sub_y * r + sub_x;

                    gx[c_in * (H_in * W_in) + in_y * W_in + in_x] =
                        go[c * (H_out * W_out) + y * W_out + x];
                }
            }
        }
        propagate_grad(input, grad_x);
    }

    void release_variables() override {
        input = nullptr;
    }
};

TensorPtr PixelShuffle::forward(const TensorPtr& input) {
    size_t C_in = input->dim(0);
    size_t H_in = input->dim(1);
    size_t W_in = input->dim(2);
    size_t r2 = upscale_factor_ * upscale_factor_;

    if (C_in % r2 != 0) {
        throw ShapeError("PixelShuffle channels must be divisible by upscale_factor squared");
    }

    size_t C_out = C_in / r2;
    size_t H_out = H_in * upscale_factor_;
    size_t W_out = W_in * upscale_factor_;

    TensorPtr output = Tensor::create({static_cast<int64_t>(C_out),
                                       static_cast<int64_t>(H_out),
                                       static_cast<int64_t>(W_out)}, input->requires_grad());
    const float* in_data = input->data();
    float* out_data = output->data();

    for (size_t c = 0; c < C_out; ++c) {
        for (size_t y = 0; y < H_out; ++y) {
            for (size_t x = 0; x < W_out; ++x) {
                size_t in_y = y / upscale_factor_;
                size_t in_x = x / upscale_factor_;
                size_t sub_y = y % upscale_factor_;
                size_t sub_x = x % upscale_factor_;
                size_t c_in = c * r2 + sub_y * upscale_factor_ + sub_x;

                out_data[c * (H_out * W_out) + y * W_out + x] =
                    in_data[c_in * (H_in * W_in) + in_y * W_in + in_x];
            }
        }
    }

    if (input->requires_grad()) {
        auto node = std::make_shared<PixelShuffleNode>();
        node->input = input;
        node->r = upscale_factor_;
        output->set_grad_fn(node);
    }

    return output;
}

// -------------------------------------------------------------
// Upsample Implementation
// -------------------------------------------------------------
struct UpsampleNode : public AutogradNode {
    TensorPtr input;
    float scale;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_x = Tensor::zeros(input->shape());
        const float* go = grad_output->data();
        float* gx = grad_x->data();

        size_t C = input->dim(0);
        size_t H_in = input->dim(1);
        size_t W_in = input->dim(2);
        size_t H_out = grad_output->dim(1);
        size_t W_out = grad_output->dim(2);

        float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
        float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < H_out; ++y) {
                for (size_t x = 0; x < W_out; ++x) {
                    float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                    float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                    src_y = std::max(src_y, 0.0f);
                    src_x = std::max(src_x, 0.0f);

                    int y0 = static_cast<int>(std::floor(src_y));
                    int x0 = static_cast<int>(std::floor(src_x));
                    int y1 = std::min(y0 + 1, static_cast<int>(H_in) - 1);
                    int x1 = std::min(x0 + 1, static_cast<int>(W_in) - 1);
                    y0 = std::min(y0, static_cast<int>(H_in) - 1);
                    x0 = std::min(x0, static_cast<int>(W_in) - 1);

                    float ly = src_y - float(y0);
                    float lx = src_x - float(x0);
                    float hy = 1.0f - ly;
                    float hx = 1.0f - lx;

                    float go_val = go[c * (H_out * W_out) + y * W_out + x];
                    size_t in_offset = c * (H_in * W_in);

                    gx[in_offset + y0 * W_in + x0] += hy * hx * go_val;
                    gx[in_offset + y0 * W_in + x1] += hy * lx * go_val;
                    gx[in_offset + y1 * W_in + x0] += ly * hx * go_val;
                    gx[in_offset + y1 * W_in + x1] += ly * lx * go_val;
                }
            }
        }
        propagate_grad(input, grad_x);
    }

    void release_variables() override {
        input = nullptr;
    }
};

TensorPtr Upsample::forward(const TensorPtr& input) {
    size_t C = input->dim(0);
    size_t H_in = input->dim(1);
    size_t W_in = input->dim(2);

    size_t H_out = static_cast<size_t>(std::round(H_in * scale_factor_));
    size_t W_out = static_cast<size_t>(std::round(W_in * scale_factor_));

    TensorPtr output = Tensor::create({static_cast<int64_t>(C),
                                       static_cast<int64_t>(H_out),
                                       static_cast<int64_t>(W_out)}, input->requires_grad());
    const float* in_data = input->data();
    float* out_data = output->data();

    float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
    float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

    for (size_t c = 0; c < C; ++c) {
        for (size_t y = 0; y < H_out; ++y) {
            for (size_t x = 0; x < W_out; ++x) {
                float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
                src_y = std::max(src_y, 0.0f);
                src_x = std::max(src_x, 0.0f);

                int y0 = static_cast<int>(std::floor(src_y));
                int x0 = static_cast<int>(std::floor(src_x));
                int y1 = std::min(y0 + 1, static_cast<int>(H_in) - 1);
                int x1 = std::min(x0 + 1, static_cast<int>(W_in) - 1);
                y0 = std::min(y0, static_cast<int>(H_in) - 1);
                x0 = std::min(x0, static_cast<int>(W_in) - 1);

                float ly = src_y - float(y0);
                float lx = src_x - float(x0);
                float hy = 1.0f - ly;
                float hx = 1.0f - lx;

                size_t in_offset = c * (H_in * W_in);
                float v00 = in_data[in_offset + y0 * W_in + x0];
                float v01 = in_data[in_offset + y0 * W_in + x1];
                float v10 = in_data[in_offset + y1 * W_in + x0];
                float v11 = in_data[in_offset + y1 * W_in + x1];

                out_data[c * (H_out * W_out) + y * W_out + x] =
                    hy * (hx * v00 + lx * v01) + ly * (hx * v10 + lx * v11);
            }
        }
    }

    if (input->requires_grad()) {
        auto node = std::make_shared<UpsampleNode>();
        node->input = input;
        node->scale = scale_factor_;
        output->set_grad_fn(node);
    }

    return output;
}

// -------------------------------------------------------------
// MaxPool2d Autograd Node & Layer
// -------------------------------------------------------------
struct MaxPool2dNode : public AutogradNode {
    TensorPtr input;
    std::vector<int64_t> argmax_indices;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_x = Tensor::zeros(input->shape());
        const float* go = grad_output->data();
        float* gx = grad_x->data();
        size_t n = argmax_indices.size();

        for (size_t i = 0; i < n; ++i) {
            int64_t idx = argmax_indices[i];
            if (idx >= 0) {
                gx[idx] += go[i];
            }
        }
        propagate_grad(input, grad_x);
    }

    void release_variables() override {
        input = nullptr;
        argmax_indices.clear();
        argmax_indices.shrink_to_fit();
    }
};

TensorPtr MaxPool2d::forward(const TensorPtr& input) {
    size_t C = input->dim(0);
    size_t H_in = input->dim(1);
    size_t W_in = input->dim(2);

    int pad = static_cast<int>(padding_);
    int str = static_cast<int>(stride_);
    int K = static_cast<int>(kernel_size_);

    size_t H_out = (H_in + 2 * padding_ - kernel_size_) / stride_ + 1;
    size_t W_out = (W_in + 2 * padding_ - kernel_size_) / stride_ + 1;

    TensorPtr output = Tensor::create({static_cast<int64_t>(C),
                                       static_cast<int64_t>(H_out),
                                       static_cast<int64_t>(W_out)},
                                      input->requires_grad());

    const float* in_data = input->data();
    float* out_data = output->data();

    std::vector<int64_t> argmax_indices;
    if (input->requires_grad()) {
        argmax_indices.resize(C * H_out * W_out, -1);
    }

    #pragma omp parallel for
    for (size_t c = 0; c < C; ++c) {
        size_t in_c_offset = c * (H_in * W_in);
        size_t out_c_offset = c * (H_out * W_out);

        for (size_t yo = 0; yo < H_out; ++yo) {
            int base_yi = static_cast<int>(yo * str) - pad;
            for (size_t xo = 0; xo < W_out; ++xo) {
                int base_xi = static_cast<int>(xo * str) - pad;

                float max_val = -std::numeric_limits<float>::infinity();
                int64_t max_idx = -1;

                for (int ky = 0; ky < K; ++ky) {
                    int yi = base_yi + ky;
                    if (yi < 0 || yi >= static_cast<int>(H_in)) continue;

                    for (int kx = 0; kx < K; ++kx) {
                        int xi = base_xi + kx;
                        if (xi < 0 || xi >= static_cast<int>(W_in)) continue;

                        int64_t idx = static_cast<int64_t>(in_c_offset + yi * W_in + xi);
                        float val = in_data[idx];
                        if (val > max_val) {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                }

                size_t out_idx = out_c_offset + yo * W_out + xo;
                out_data[out_idx] = (max_idx >= 0) ? max_val : 0.0f;
                if (input->requires_grad()) {
                    argmax_indices[out_idx] = max_idx;
                }
            }
        }
    }

    if (input->requires_grad()) {
        auto node = std::make_shared<MaxPool2dNode>();
        node->input = input;
        node->argmax_indices = std::move(argmax_indices);
        output->set_grad_fn(node);
    }

    return output;
}

// -------------------------------------------------------------
// AdaptiveAvgPool2d Implementation
// -------------------------------------------------------------
struct AvgPoolNode : public AutogradNode {
    TensorPtr input;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_x = Tensor::zeros(input->shape());
        const float* go = grad_output->data();
        float* gx = grad_x->data();
        size_t C = input->dim(0);
        size_t H = input->dim(1);
        size_t W = input->dim(2);
        float norm = 1.0f / float(H * W);

        for (size_t c = 0; c < C; ++c) {
            float val = go[c] * norm;
            for (size_t sp = 0; sp < H * W; ++sp) {
                gx[c * (H * W) + sp] = val;
            }
        }
        propagate_grad(input, grad_x);
    }

    void release_variables() override {
        input = nullptr;
    }
};

TensorPtr AdaptiveAvgPool2d::forward(const TensorPtr& input) {
    size_t C = input->dim(0);
    size_t H = input->dim(1);
    size_t W = input->dim(2);

    TensorPtr output = Tensor::create({static_cast<int64_t>(C), 1, 1}, input->requires_grad());
    const float* in_data = input->data();
    float* out_data = output->data();

    float norm = 1.0f / float(H * W);
    for (size_t c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (size_t sp = 0; sp < H * W; ++sp) {
            sum += in_data[c * (H * W) + sp];
        }
        out_data[c] = sum * norm;
    }

    if (input->requires_grad()) {
        auto node = std::make_shared<AvgPoolNode>();
        node->input = input;
        output->set_grad_fn(node);
    }

    return output;
}

} // namespace soar::nn
