#pragma once

#include <soar/tensor/tensor.hpp>
#include <vector>
#include <memory>
#include <cmath>

namespace soar::optim {

struct AdamWOptions {
    float lr{1e-3f};
    float beta1{0.9f};
    float beta2{0.999f};
    float eps{1e-8f};
    float weight_decay{1e-2f};
};

/**
 * @brief AdamW optimizer with decoupled weight decay and gradient clipping.
 */
class AdamW {
public:
    AdamW(std::vector<TensorPtr> params, AdamWOptions options = {});

    void step();
    void zero_grad();
    float clip_grad_norm(float max_norm);

    void set_lr(float lr) noexcept { options_.lr = lr; }
    [[nodiscard]] float get_lr() const noexcept { return options_.lr; }

private:
    std::vector<TensorPtr> params_;
    AdamWOptions options_;
    uint64_t step_count_{0};

    // Optimizer states: first and second moments
    std::vector<std::vector<float>> m_;
    std::vector<std::vector<float>> v_;
};

/**
 * @brief Cosine Annealing learning rate scheduler with optional linear warmup.
 */
class CosineAnnealingLR {
public:
    CosineAnnealingLR(AdamW& optimizer, size_t total_steps, float eta_min = 1e-6f, size_t warmup_steps = 0);
    void step();

private:
    AdamW& optimizer_;
    float base_lr_;
    float eta_min_;
    size_t total_steps_;
    size_t warmup_steps_;
    size_t current_step_{0};
};

} // namespace soar::optim
