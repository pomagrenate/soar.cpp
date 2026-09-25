#pragma once

#include <soar/nn/blocks.hpp>
#include <string>
#include <memory>

namespace soar::nn {

enum class ModelVariant {
    Nano,
    Small,
    Medium,
    Large,
    XLarge
};

/**
 * @brief Resolution-preserving High-Resolution Segmentation Network in pure C++20.
 */
class SOARModel : public Module {
public:
    SOARModel(size_t in_channels = 1, size_t num_classes = 1, ModelVariant variant = ModelVariant::Nano);
    ~SOARModel() override = default;

    TensorPtr forward(const TensorPtr& input) override;

    [[nodiscard]] size_t in_channels() const noexcept { return in_channels_; }
    [[nodiscard]] size_t num_classes() const noexcept { return num_classes_; }
    [[nodiscard]] ModelVariant variant() const noexcept { return variant_; }
    [[nodiscard]] size_t parameter_count() const;

    // Weight serialization & checkpoint loading
    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);

private:
    size_t in_channels_;
    size_t num_classes_;
    ModelVariant variant_;
    float depth_mult_{1.0f};
    float width_mult_{1.0f};

    // Backbone
    std::shared_ptr<CBA> l0_cba_;
    std::shared_ptr<Down> l1_down_;
    std::vector<std::shared_ptr<LKR>> l2_lkr_stack_;
    std::shared_ptr<Down> l3_down_;
    std::vector<std::shared_ptr<LKR>> l4_lkr_stack_;
    std::shared_ptr<Down> l5_down_;
    std::vector<std::shared_ptr<LKR>> l6_lkr_stack_;
    std::shared_ptr<Down> l7_down_;
    std::vector<std::shared_ptr<LKR>> l8_lkr_stack_;
    std::shared_ptr<Ctx> l9_ctx_;

    // Decoder
    std::shared_ptr<Fuse> l10_fuse_;
    std::vector<std::shared_ptr<LKR>> l11_lkr_stack_;
    std::shared_ptr<Fuse> l12_fuse_;
    std::vector<std::shared_ptr<LKR>> l13_lkr_stack_;
    std::shared_ptr<Fuse> l14_fuse_;
    std::shared_ptr<Agg> l15_agg_;
    std::shared_ptr<Fuse> l16_fuse_;

    // Head
    std::shared_ptr<SegHead> l17_head_;
};

} // namespace soar::nn
