#include <soar/nn/soar_model.hpp>
#include <soar/core/logging.hpp>

#include <fstream>
#include <cmath>
#include <cstring>

namespace soar::nn {

namespace {

size_t scale_channels(size_t c, float width_mult, size_t divisor = 8) {
    size_t scaled = static_cast<size_t>(std::ceil((static_cast<float>(c) * width_mult) / static_cast<float>(divisor))) * divisor;
    return std::max(divisor, scaled);
}

} // anonymous namespace

SOARModel::SOARModel(size_t in_channels, size_t num_classes, ModelVariant variant)
    : Module("SOARModel"), in_channels_(in_channels), num_classes_(num_classes), variant_(variant) {

    switch (variant_) {
        case ModelVariant::Nano:
            depth_mult_ = 0.50f;
            width_mult_ = 0.50f;
            break;
        case ModelVariant::Small:
            depth_mult_ = 0.67f;
            width_mult_ = 0.75f;
            break;
        case ModelVariant::Medium:
            depth_mult_ = 1.00f;
            width_mult_ = 1.00f;
            break;
        case ModelVariant::Large:
            depth_mult_ = 1.25f;
            width_mult_ = 1.25f;
            break;
        case ModelVariant::XLarge:
            depth_mult_ = 1.50f;
            width_mult_ = 1.50f;
            break;
    }

    auto wc = [this](size_t c) -> size_t {
        return scale_channels(c, width_mult_);
    };

    size_t c0 = wc(16);
    size_t c1 = wc(32);
    size_t c2 = wc(32);
    size_t c3 = wc(64);
    size_t c4 = wc(64);
    size_t c5 = wc(128);
    size_t c6 = wc(128);
    size_t c7 = wc(256);
    size_t c8 = wc(256);
    size_t c9 = wc(256);

    // Backbone layers
    l0_cba_ = std::make_shared<CBA>(in_channels_, c0, 3, 2);
    l1_down_ = std::make_shared<Down>(c0, c1);
    l2_lkr_ = std::make_shared<LKR>(c2, 5, 2.0f);
    l3_down_ = std::make_shared<Down>(c2, c3);
    l4_lkr_ = std::make_shared<LKR>(c4, 7, 2.0f);
    l5_down_ = std::make_shared<Down>(c4, c5);

    register_submodule("l0_cba", l0_cba_);
    register_submodule("l1_down", l1_down_);
    register_submodule("l2_lkr", l2_lkr_);
    register_submodule("l3_down", l3_down_);
    register_submodule("l4_lkr", l4_lkr_);
    register_submodule("l5_down", l5_down_);

    size_t n6 = std::max(size_t(1), static_cast<size_t>(std::round(2.0f * depth_mult_)));
    for (size_t i = 0; i < n6; ++i) {
        auto block = std::make_shared<LKR>(c6, 7, 2.0f);
        register_submodule("l6_lkr_" + std::to_string(i), block);
        l6_lkr_stack_.push_back(block);
    }

    l7_down_ = std::make_shared<Down>(c6, c7);
    l8_lkr_ = std::make_shared<LKR>(c8, 7, 2.0f);
    l9_ctx_ = std::make_shared<Ctx>(c8, c9);

    register_submodule("l7_down", l7_down_);
    register_submodule("l8_lkr", l8_lkr_);
    register_submodule("l9_ctx", l9_ctx_);

    // Decoder layers
    size_t c10 = wc(128);
    size_t c11 = wc(128);
    size_t c12 = wc(64);
    size_t c13 = wc(64);
    size_t c14 = wc(32);
    size_t c15 = wc(32);
    size_t c16 = wc(16);

    l10_fuse_ = std::make_shared<Fuse>(c9, c6, c10);
    l11_lkr_ = std::make_shared<LKR>(c11, 7, 2.0f);
    l12_fuse_ = std::make_shared<Fuse>(c11, c4, c12);
    l13_lkr_ = std::make_shared<LKR>(c13, 7, 2.0f);
    l14_fuse_ = std::make_shared<Fuse>(c13, c2, c14);
    l15_agg_ = std::make_shared<Agg>(std::vector<size_t>{c14, c13, c11}, c15);
    l16_fuse_ = std::make_shared<Fuse>(c15, c0, c16);

    register_submodule("l10_fuse", l10_fuse_);
    register_submodule("l11_lkr", l11_lkr_);
    register_submodule("l12_fuse", l12_fuse_);
    register_submodule("l13_lkr", l13_lkr_);
    register_submodule("l14_fuse", l14_fuse_);
    register_submodule("l15_agg", l15_agg_);
    register_submodule("l16_fuse", l16_fuse_);

    // Head
    size_t mid_head = wc(16);
    l17_head_ = std::make_shared<SegHead>(c16, num_classes_, mid_head, 2, 0.01f);
    register_submodule("l17_head", l17_head_);

    SOAR_LOG_INFO("SOARModel instantiated with {} parameters.", parameter_count());
}

size_t SOARModel::parameter_count() const {
    size_t count = 0;
    for (const auto& p : parameters()) {
        count += p->numel();
    }
    return count;
}

TensorPtr SOARModel::forward(const TensorPtr& input) {
    // 0: Stem detail skip (s2)
    TensorPtr s2 = l0_cba_->forward(input);

    // 1: Downsample to s4
    TensorPtr s4 = l1_down_->forward(s2);

    // 2: P2 lightweight refinement (s4)
    TensorPtr p2 = l2_lkr_->forward(s4);

    // 3: Downsample to s8
    TensorPtr s8 = l3_down_->forward(p2);

    // 4: P3 (s8)
    TensorPtr p3 = l4_lkr_->forward(s8);

    // 5: Downsample to s16
    TensorPtr s16 = l5_down_->forward(p3);

    // 6: P4 stack (s16)
    TensorPtr p4 = s16;
    for (auto& block : l6_lkr_stack_) {
        p4 = block->forward(p4);
    }

    // 7: Downsample to s32
    TensorPtr s32 = l7_down_->forward(p4);

    // 8: P5 (s32)
    TensorPtr p5 = l8_lkr_->forward(s32);

    // 9: Context multi-scale pooling (s32)
    TensorPtr p5_ctx = l9_ctx_->forward(p5);

    // 10: Fuse s32 + s16 -> s16
    TensorPtr d16 = l10_fuse_->forward({p5_ctx, p4});

    // 11: LKR s16
    TensorPtr d16_lkr = l11_lkr_->forward(d16);

    // 12: Fuse s16 + s8 -> s8
    TensorPtr d8 = l12_fuse_->forward({d16_lkr, p3});

    // 13: LKR s8
    TensorPtr d8_lkr = l13_lkr_->forward(d8);

    // 14: Fuse s8 + s4 -> s4
    TensorPtr d4 = l14_fuse_->forward({d8_lkr, p2});

    // 15: Multi-scale semantic aggregation (s4)
    TensorPtr agg4 = l15_agg_->forward({d4, d8_lkr, d16_lkr});

    // 16: Fuse s4 + s2 (stem detail) -> s2
    TensorPtr d2 = l16_fuse_->forward({agg4, s2});

    // 17: Sub-pixel PixelShuffle head (s2 -> s1 = native resolution)
    return l17_head_->forward(d2);
}

void SOARModel::save_weights(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw DeviceError("Failed to open file for weight saving: " + path);
    }

    uint32_t magic = 0x534F4152; // 'SOAR'
    uint32_t version = 1;
    uint32_t variant_id = static_cast<uint32_t>(variant_);
    auto named_params = named_parameters();
    uint32_t num_tensors = static_cast<uint32_t>(named_params.size());

    out.write(reinterpret_cast<const char*>(&magic), 4);
    out.write(reinterpret_cast<const char*>(&version), 4);
    out.write(reinterpret_cast<const char*>(&variant_id), 4);
    out.write(reinterpret_cast<const char*>(&num_tensors), 4);

    for (const auto& [name, param] : named_params) {
        char name_buf[128]{0};
        std::strncpy(name_buf, name.c_str(), 127);
        out.write(name_buf, 128);

        uint32_t ndim = static_cast<uint32_t>(param->ndim());
        out.write(reinterpret_cast<const char*>(&ndim), 4);

        uint32_t shape_buf[4]{1, 1, 1, 1};
        for (size_t d = 0; d < param->ndim() && d < 4; ++d) {
            shape_buf[d] = static_cast<uint32_t>(param->dim(d));
        }
        out.write(reinterpret_cast<const char*>(shape_buf), 16);

        uint64_t num_bytes = param->bytes();
        out.write(reinterpret_cast<const char*>(&num_bytes), 8);
        out.write(reinterpret_cast<const char*>(param->data()), num_bytes);
    }
    SOAR_LOG_INFO("Successfully saved {} model tensors to {}", num_tensors, path);
}

void SOARModel::load_weights(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw DeviceError("Failed to open file for weight loading: " + path);
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t variant_id = 0;
    uint32_t num_tensors = 0;

    in.read(reinterpret_cast<char*>(&magic), 4);
    in.read(reinterpret_cast<char*>(&version), 4);
    in.read(reinterpret_cast<char*>(&variant_id), 4);
    in.read(reinterpret_cast<char*>(&num_tensors), 4);

    if (magic != 0x534F4152) {
        throw DeviceError("Invalid magic header in weight file: " + path);
    }

    auto named_params = named_parameters();
    std::map<std::string, TensorPtr> param_map(named_params.begin(), named_params.end());

    size_t loaded_count = 0;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        char name_buf[128]{0};
        in.read(name_buf, 128);
        std::string name(name_buf);

        uint32_t ndim = 0;
        in.read(reinterpret_cast<char*>(&ndim), 4);

        uint32_t shape_buf[4]{0};
        in.read(reinterpret_cast<char*>(shape_buf), 16);

        uint64_t num_bytes = 0;
        in.read(reinterpret_cast<char*>(&num_bytes), 8);

        auto it = param_map.find(name);
        if (it != param_map.end()) {
            if (it->second->bytes() == num_bytes) {
                in.read(reinterpret_cast<char*>(it->second->data()), num_bytes);
                loaded_count++;
            } else {
                in.seekg(num_bytes, std::ios::cur);
            }
        } else {
            in.seekg(num_bytes, std::ios::cur);
        }
    }
    SOAR_LOG_INFO("Loaded {}/{} matching tensors from {}", loaded_count, named_params.size(), path);
}

} // namespace soar::nn
