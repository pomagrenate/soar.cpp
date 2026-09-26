#pragma once

#include <soar/tensor/tensor.hpp>
#include <vector>
#include <memory>

namespace soar::optim {

struct SGDOptions {
    float lr{1e-2f};
    float momentum{0.0f};
    float dampening{0.0f};
    float weight_decay{0.0f};
    bool nesterov{false};
};

/**
 * @brief Production SGD optimizer with Momentum, Dampening, Weight Decay, and Nesterov.
 * Matches PyTorch torch::optim::SGD mathematical specification.
 */
class SGD {
public:
    explicit SGD(std::vector<TensorPtr> params, SGDOptions options = {});

    void step();
    void zero_grad();
    float clip_grad_norm(float max_norm);

    void set_lr(float lr) noexcept { options_.lr = lr; }
    [[nodiscard]] float get_lr() const noexcept { return options_.lr; }
    [[nodiscard]] const SGDOptions& options() const noexcept { return options_; }

private:
    std::vector<TensorPtr> params_;
    SGDOptions options_;
    std::vector<std::vector<float>> momentum_buffers_;
    std::vector<bool> has_momentum_buffer_;
};

} // namespace soar::optim
