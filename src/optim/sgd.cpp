#include <soar/optim/sgd.hpp>
#include <cmath>
#include <algorithm>

namespace soar::optim {

SGD::SGD(std::vector<TensorPtr> params, SGDOptions options)
    : params_(std::move(params)), options_(options) {
    momentum_buffers_.resize(params_.size());
    has_momentum_buffer_.resize(params_.size(), false);
    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i]) {
            momentum_buffers_[i].resize(params_[i]->numel(), 0.0f);
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : params_) {
        if (p) {
            p->zero_grad();
        }
    }
}

float SGD::clip_grad_norm(float max_norm) {
    double total_norm_sq = 0.0;
    for (const auto& p : params_) {
        if (!p || !p->grad()) continue;
        const float* g = p->grad()->data();
        size_t n = p->numel();
        for (size_t i = 0; i < n; ++i) {
            total_norm_sq += static_cast<double>(g[i]) * static_cast<double>(g[i]);
        }
    }

    float total_norm = static_cast<float>(std::sqrt(total_norm_sq));
    if (total_norm > max_norm && total_norm > 1e-6f) {
        float clip_coef = max_norm / total_norm;
        for (auto& p : params_) {
            if (!p || !p->grad()) continue;
            float* g = p->grad()->data();
            size_t n = p->numel();
            for (size_t i = 0; i < n; ++i) {
                g[i] *= clip_coef;
            }
        }
    }
    return total_norm;
}

void SGD::step() {
    float lr = options_.lr;
    float weight_decay = options_.weight_decay;
    float momentum = options_.momentum;
    float dampening = options_.dampening;
    bool nesterov = options_.nesterov;

    for (size_t idx = 0; idx < params_.size(); ++idx) {
        auto& p = params_[idx];
        if (!p || !p->grad()) continue;

        float* theta = p->data();
        const float* grad = p->grad()->data();
        auto& buf = momentum_buffers_[idx];
        bool has_buf = has_momentum_buffer_[idx];
        size_t n = p->numel();

        #pragma omp parallel for if (n > 4096)
        for (size_t i = 0; i < n; ++i) {
            float d_p = grad[i];
            if (weight_decay != 0.0f) {
                d_p += weight_decay * theta[i];
            }
            if (momentum != 0.0f) {
                if (!has_buf) {
                    buf[i] = d_p;
                } else {
                    buf[i] = momentum * buf[i] + (1.0f - dampening) * d_p;
                }
                if (nesterov) {
                    d_p = d_p + momentum * buf[i];
                } else {
                    d_p = buf[i];
                }
            }
            theta[i] -= lr * d_p;
        }

        has_momentum_buffer_[idx] = (momentum != 0.0f);

        if (p->is_on_device()) {
            p->sync_to_device();
        }
    }
}

} // namespace soar::optim
