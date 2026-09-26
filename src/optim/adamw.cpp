#include <soar/optim/adamw.hpp>
#include <algorithm>
#include <cmath>

namespace soar::optim {

AdamW::AdamW(std::vector<TensorPtr> params, AdamWOptions options)
    : params_(std::move(params)), options_(options) {
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i]) {
            size_t n = params_[i]->numel();
            m_[i].resize(n, 0.0f);
            v_[i].resize(n, 0.0f);
        }
    }
}

void AdamW::zero_grad() {
    for (auto& p : params_) {
        if (p) {
            p->zero_grad();
        }
    }
}

float AdamW::clip_grad_norm(float max_norm) {
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

void AdamW::step() {
    step_count_++;
    float lr = options_.lr;
    float beta1 = options_.beta1;
    float beta2 = options_.beta2;
    float eps = options_.eps;
    float wd = options_.weight_decay;

    float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(step_count_));
    float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(step_count_));
    float sqrt_bc2 = std::sqrt(bias_correction2);
    float step_size = lr / bias_correction1;

    for (size_t idx = 0; idx < params_.size(); ++idx) {
        auto& p = params_[idx];
        if (!p || !p->grad()) continue;

        float* theta = p->data();
        const float* g = p->grad()->data();
        auto& m = m_[idx];
        auto& v = v_[idx];
        size_t n = p->numel();

        #pragma omp parallel for if (n > 4096)
        for (size_t i = 0; i < n; ++i) {
            float g_val = g[i];

            // 1. Decoupled weight decay
            if (wd != 0.0f) {
                theta[i] -= lr * wd * theta[i];
            }

            // 2. Moments update
            m[i] = beta1 * m[i] + (1.0f - beta1) * g_val;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g_val * g_val;

            // 3. Denominator with PyTorch-exact bias correction
            float denom = (std::sqrt(v[i]) / sqrt_bc2) + eps;

            // 4. Update parameter
            theta[i] -= step_size * (m[i] / denom);
        }

        if (p->is_on_device()) {
            p->sync_to_device();
        }
    }
}

CosineAnnealingLR::CosineAnnealingLR(AdamW& optimizer, size_t total_steps, float eta_min, size_t warmup_steps)
    : optimizer_(optimizer), base_lr_(optimizer.get_lr()), eta_min_(eta_min),
      total_steps_(total_steps), warmup_steps_(warmup_steps) {}

void CosineAnnealingLR::step() {
    current_step_++;
    if (current_step_ <= warmup_steps_ && warmup_steps_ > 0) {
        float lr = base_lr_ * static_cast<float>(current_step_) / static_cast<float>(warmup_steps_);
        optimizer_.set_lr(lr);
        return;
    }

    size_t adjusted_step = current_step_ - warmup_steps_;
    size_t adjusted_total = total_steps_ - warmup_steps_;

    if (adjusted_step > adjusted_total) {
        optimizer_.set_lr(eta_min_);
        return;
    }

    constexpr float pi = 3.14159265358979323846f;
    float progress = static_cast<float>(adjusted_step) / static_cast<float>(adjusted_total);
    float lr = eta_min_ + 0.5f * (base_lr_ - eta_min_) * (1.0f + std::cos(progress * pi));
    optimizer_.set_lr(lr);
}

} // namespace soar::optim
