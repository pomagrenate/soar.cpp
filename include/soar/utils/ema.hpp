#pragma once

#include <soar/tensor/tensor.hpp>
#include <soar/nn/module.hpp>
#include <vector>
#include <memory>
#include <cstring>

namespace soar::utils {

class ModelEMA {
public:
    explicit ModelEMA(const std::shared_ptr<nn::Module>& model, float decay = 0.9999f)
        : decay_(decay) {
        init_shadow(model);
    }

    void update(const std::shared_ptr<nn::Module>& model) {
        auto params = model->parameters();
        for (size_t i = 0; i < params.size() && i < shadow_params_.size(); ++i) {
            const float* mp = params[i]->data();
            float* sp = shadow_params_[i]->data();
            size_t n = shadow_params_[i]->numel();
            for (size_t j = 0; j < n; ++j) {
                sp[j] = decay_ * sp[j] + (1.0f - decay_) * mp[j];
            }
        }
    }

    void apply_to(const std::shared_ptr<nn::Module>& model) const {
        auto params = model->parameters();
        for (size_t i = 0; i < params.size() && i < shadow_params_.size(); ++i) {
            const float* sp = shadow_params_[i]->data();
            float* mp = params[i]->data();
            size_t n = params[i]->numel();
            std::memcpy(mp, sp, n * sizeof(float));
        }
    }

    void copy_from(const std::shared_ptr<nn::Module>& model) {
        init_shadow(model);
    }

    [[nodiscard]] float decay() const noexcept { return decay_; }
    void set_decay(float d) noexcept { decay_ = d; }
    [[nodiscard]] const std::vector<TensorPtr>& shadow_params() const noexcept { return shadow_params_; }

private:
    void init_shadow(const std::shared_ptr<nn::Module>& model) {
        shadow_params_.clear();
        for (const auto& p : model->parameters()) {
            auto shadow_p = Tensor::create(p->shape(), false);
            std::memcpy(shadow_p->data(), p->data(), p->numel() * sizeof(float));
            shadow_params_.push_back(shadow_p);
        }
    }

    float decay_;
    std::vector<TensorPtr> shadow_params_;
};

} // namespace soar::utils
