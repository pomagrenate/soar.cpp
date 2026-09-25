#include <soar/nn/soar_model.hpp>
#include <soar/autograd/node.hpp>
#include <soar/core/logging.hpp>

#include <fstream>
#include <cmath>
#include <cstring>

namespace soar::nn {

// -------------------------------------------------------------
// Auto-padding and unpadding autograd nodes
// -------------------------------------------------------------
struct AutoPadNode : public AutogradNode {
    TensorPtr input;
    std::vector<size_t> src_indices;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_in = Tensor::zeros(input->shape());
        const float* go = grad_output->data();
        float* gi = grad_in->data();
        size_t n = src_indices.size();
        for (size_t i = 0; i < n; ++i) {
            gi[src_indices[i]] += go[i];
        }
        propagate_grad(input, grad_in);
    }

    void release_variables() override {
        input = nullptr;
        src_indices.clear();
        src_indices.shrink_to_fit();
    }
};

struct UnpadNode : public AutogradNode {
    TensorPtr input;
    size_t orig_h;
    size_t orig_w;

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        if (input && input->grad_fn()) return {input->grad_fn()};
        return {};
    }

    void backward(const TensorPtr& grad_output) override {
        if (!input->requires_grad()) return;
        TensorPtr grad_in = Tensor::zeros(input->shape());
        const float* go = grad_output->data();
        float* gi = grad_in->data();
        size_t C = input->dim(0);
        size_t H_pad = input->dim(1);
        size_t W_pad = input->dim(2);

        for (size_t c = 0; c < C; ++c) {
            for (size_t y = 0; y < orig_h; ++y) {
                for (size_t x = 0; x < orig_w; ++x) {
                    gi[c * (H_pad * W_pad) + y * W_pad + x] = go[c * (orig_h * orig_w) + y * orig_w + x];
                }
            }
        }
        propagate_grad(input, grad_in);
    }

    void release_variables() override {
        input = nullptr;
    }
};

static TensorPtr pad_spatial(const TensorPtr& input, size_t ph, size_t pw) {
    size_t C = input->dim(0);
    size_t H = input->dim(1);
    size_t W = input->dim(2);
    size_t H_pad = H + ph;
    size_t W_pad = W + pw;

    TensorPtr padded = Tensor::create({static_cast<int64_t>(C),
                                      static_cast<int64_t>(H_pad),
                                      static_cast<int64_t>(W_pad)},
                                     input->requires_grad());
    const float* in_data = input->data();
    float* out_data = padded->data();

    bool reflect = (ph < H && pw < W);
    std::vector<size_t> src_indices;
    if (input->requires_grad()) {
        src_indices.resize(C * H_pad * W_pad);
    }

    for (size_t c = 0; c < C; ++c) {
        for (size_t y = 0; y < H_pad; ++y) {
            size_t y_src = y;
            if (y >= H) {
                y_src = reflect ? (2 * (H - 1) - y) : (H - 1);
            }
            for (size_t x = 0; x < W_pad; ++x) {
                size_t x_src = x;
                if (x >= W) {
                    x_src = reflect ? (2 * (W - 1) - x) : (W - 1);
                }
                size_t src_idx = c * (H * W) + y_src * W + x_src;
                size_t dst_idx = c * (H_pad * W_pad) + y * W_pad + x;
                out_data[dst_idx] = in_data[src_idx];
                if (input->requires_grad()) {
                    src_indices[dst_idx] = src_idx;
                }
            }
        }
    }

    if (input->requires_grad()) {
        auto node = std::make_shared<AutoPadNode>();
        node->input = input;
        node->src_indices = std::move(src_indices);
        padded->set_grad_fn(node);
    }
    return padded;
}

static TensorPtr unpad_spatial(const TensorPtr& input, size_t orig_h, size_t orig_w) {
    size_t C = input->dim(0);
    size_t H_pad = input->dim(1);
    size_t W_pad = input->dim(2);

    TensorPtr sliced = Tensor::create({static_cast<int64_t>(C),
                                      static_cast<int64_t>(orig_h),
                                      static_cast<int64_t>(orig_w)},
                                     input->requires_grad());
    const float* in_data = input->data();
    float* out_data = sliced->data();

    for (size_t c = 0; c < C; ++c) {
        for (size_t y = 0; y < orig_h; ++y) {
            for (size_t x = 0; x < orig_w; ++x) {
                out_data[c * (orig_h * orig_w) + y * orig_w + x] =
                    in_data[c * (H_pad * W_pad) + y * W_pad + x];
            }
        }
    }

    if (input->requires_grad()) {
        auto node = std::make_shared<UnpadNode>();
        node->input = input;
        node->orig_h = orig_h;
        node->orig_w = orig_w;
        sliced->set_grad_fn(node);
    }
    return sliced;
}

SOARModel::SOARModel(size_t in_channels, size_t num_classes, ModelVariant variant)
    : Module("SOARModel"), in_channels_(in_channels), num_classes_(num_classes), variant_(variant) {

    struct ArchConfig {
        size_t c0, c1, c2, c3, c4, c5, c6, c7, c8, c9;
        size_t c10, c11, c12, c13, c14, c15, c16, c17;
        size_t n2{1}, n4{1}, n6{2}, n8{1}, n11{1}, n13{1};
    };
    ArchConfig cfg{};

    switch (variant_) {
        case ModelVariant::Nano:
            cfg = {16, 32, 32, 64, 64, 128, 128, 256, 256, 256,
                   128, 128, 64, 64, 32, 32, 16, 16,
                   1, 1, 2, 1, 1, 1};
            break;
        case ModelVariant::Small:
            cfg = {24, 48, 48, 96, 96, 192, 192, 384, 384, 384,
                   192, 192, 96, 96, 48, 48, 24, 24,
                   1, 2, 3, 2, 1, 1};
            break;
        case ModelVariant::Medium:
            cfg = {32, 64, 64, 128, 128, 256, 256, 512, 512, 512,
                   256, 256, 128, 128, 64, 64, 32, 32,
                   2, 3, 5, 3, 1, 1};
            break;
        case ModelVariant::Large:
            cfg = {32, 64, 64, 160, 160, 320, 320, 640, 640, 640,
                   320, 320, 160, 160, 64, 64, 32, 32,
                   2, 3, 7, 4, 2, 1};
            break;
        case ModelVariant::XLarge:
            cfg = {40, 80, 80, 192, 192, 384, 384, 768, 768, 768,
                   384, 384, 192, 192, 80, 80, 40, 40,
                   2, 4, 8, 5, 2, 2};
            break;
        default:
            break;
    }

    // Backbone layers
    l0_cba_ = std::make_shared<CBA>(in_channels_, cfg.c0, 3, 2);
    l1_down_ = std::make_shared<Down>(cfg.c0, cfg.c1);
    register_submodule("l0_cba", l0_cba_);
    register_submodule("l1_down", l1_down_);

    for (size_t i = 0; i < cfg.n2; ++i) {
        auto block = std::make_shared<LKR>(cfg.c2, 5, 2.0f);
        register_submodule(cfg.n2 == 1 ? "l2_lkr" : "l2_lkr_" + std::to_string(i), block);
        l2_lkr_stack_.push_back(block);
    }

    l3_down_ = std::make_shared<Down>(cfg.c2, cfg.c3);
    register_submodule("l3_down", l3_down_);

    for (size_t i = 0; i < cfg.n4; ++i) {
        auto block = std::make_shared<LKR>(cfg.c4, 7, 2.0f);
        register_submodule(cfg.n4 == 1 ? "l4_lkr" : "l4_lkr_" + std::to_string(i), block);
        l4_lkr_stack_.push_back(block);
    }

    l5_down_ = std::make_shared<Down>(cfg.c4, cfg.c5);
    register_submodule("l5_down", l5_down_);

    for (size_t i = 0; i < cfg.n6; ++i) {
        auto block = std::make_shared<LKR>(cfg.c6, 7, 2.0f);
        register_submodule("l6_lkr_" + std::to_string(i), block);
        l6_lkr_stack_.push_back(block);
    }

    l7_down_ = std::make_shared<Down>(cfg.c6, cfg.c7);
    register_submodule("l7_down", l7_down_);

    for (size_t i = 0; i < cfg.n8; ++i) {
        auto block = std::make_shared<LKR>(cfg.c8, 7, 2.0f);
        register_submodule(cfg.n8 == 1 ? "l8_lkr" : "l8_lkr_" + std::to_string(i), block);
        l8_lkr_stack_.push_back(block);
    }

    l9_ctx_ = std::make_shared<Ctx>(cfg.c8, cfg.c9);
    register_submodule("l9_ctx", l9_ctx_);

    // Decoder layers
    l10_fuse_ = std::make_shared<Fuse>(cfg.c9, cfg.c6, cfg.c10);
    register_submodule("l10_fuse", l10_fuse_);

    for (size_t i = 0; i < cfg.n11; ++i) {
        auto block = std::make_shared<LKR>(cfg.c11, 7, 2.0f);
        register_submodule(cfg.n11 == 1 ? "l11_lkr" : "l11_lkr_" + std::to_string(i), block);
        l11_lkr_stack_.push_back(block);
    }

    l12_fuse_ = std::make_shared<Fuse>(cfg.c11, cfg.c4, cfg.c12);
    register_submodule("l12_fuse", l12_fuse_);

    for (size_t i = 0; i < cfg.n13; ++i) {
        auto block = std::make_shared<LKR>(cfg.c13, 7, 2.0f);
        register_submodule(cfg.n13 == 1 ? "l13_lkr" : "l13_lkr_" + std::to_string(i), block);
        l13_lkr_stack_.push_back(block);
    }

    l14_fuse_ = std::make_shared<Fuse>(cfg.c13, cfg.c2, cfg.c14);
    l15_agg_ = std::make_shared<Agg>(std::vector<size_t>{cfg.c14, cfg.c13, cfg.c11}, cfg.c15);
    l16_fuse_ = std::make_shared<Fuse>(cfg.c15, cfg.c0, cfg.c16);
    register_submodule("l14_fuse", l14_fuse_);
    register_submodule("l15_agg", l15_agg_);
    register_submodule("l16_fuse", l16_fuse_);

    // Head
    l17_head_ = std::make_shared<SegHead>(cfg.c16, num_classes_, cfg.c17, 2, 0.01f);
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
    constexpr size_t DIVISOR = 32;
    size_t H = input->dim(1);
    size_t W = input->dim(2);
    size_t ph = (DIVISOR - (H % DIVISOR)) % DIVISOR;
    size_t pw = (DIVISOR - (W % DIVISOR)) % DIVISOR;

    TensorPtr x = input;
    if (ph > 0 || pw > 0) {
        x = pad_spatial(input, ph, pw);
    }

    // 0: Stem detail skip (s2)
    TensorPtr s2 = l0_cba_->forward(x);

    // 1: Downsample to s4
    TensorPtr s4 = l1_down_->forward(s2);

    // 2: P2 lightweight refinement (s4)
    TensorPtr p2 = s4;
    for (auto& b : l2_lkr_stack_) p2 = b->forward(p2);

    // 3: Downsample to s8
    TensorPtr s8 = l3_down_->forward(p2);

    // 4: P3 (s8)
    TensorPtr p3 = s8;
    for (auto& b : l4_lkr_stack_) p3 = b->forward(p3);

    // 5: Downsample to s16
    TensorPtr s16 = l5_down_->forward(p3);

    // 6: P4 stack (s16)
    TensorPtr p4 = s16;
    for (auto& b : l6_lkr_stack_) p4 = b->forward(p4);

    // 7: Downsample to s32
    TensorPtr s32 = l7_down_->forward(p4);

    // 8: P5 (s32)
    TensorPtr p5 = s32;
    for (auto& b : l8_lkr_stack_) p5 = b->forward(p5);

    // 9: Context multi-scale pooling (s32)
    TensorPtr p5_ctx = l9_ctx_->forward(p5);

    // 10: Fuse s32 + s16 -> s16
    TensorPtr d16 = l10_fuse_->forward({p5_ctx, p4});

    // 11: LKR s16
    TensorPtr d16_lkr = d16;
    for (auto& b : l11_lkr_stack_) d16_lkr = b->forward(d16_lkr);

    // 12: Fuse s16 + s8 -> s8
    TensorPtr d8 = l12_fuse_->forward({d16_lkr, p3});

    // 13: LKR s8
    TensorPtr d8_lkr = d8;
    for (auto& b : l13_lkr_stack_) d8_lkr = b->forward(d8_lkr);

    // 14: Fuse s8 + s4 -> s4
    TensorPtr d4 = l14_fuse_->forward({d8_lkr, p2});

    // 15: Multi-scale semantic aggregation (s4)
    TensorPtr agg4 = l15_agg_->forward({d4, d8_lkr, d16_lkr});

    // 16: Fuse s4 + s2 (stem detail) -> s2
    TensorPtr d2 = l16_fuse_->forward({agg4, s2});

    // 17: Sub-pixel PixelShuffle head (s2 -> s1 = native resolution)
    TensorPtr out = l17_head_->forward(d2);

    if (ph > 0 || pw > 0) {
        out = unpad_spatial(out, H, W);
    }
    return out;
}

void SOARModel::save_weights(const std::string& path) const {
    FILE* out = std::fopen(path.c_str(), "wb");
    if (!out) {
        throw DeviceError("Failed to open file for weight saving: " + path);
    }

    uint32_t magic = 0x534F4152; // 'SOAR'
    uint32_t version = 1;
    uint32_t variant_id = static_cast<uint32_t>(variant_);
    auto named_params = named_parameters();
    uint32_t num_tensors = static_cast<uint32_t>(named_params.size());

    std::fwrite(&magic, 4, 1, out);
    std::fwrite(&version, 4, 1, out);
    std::fwrite(&variant_id, 4, 1, out);
    std::fwrite(&num_tensors, 4, 1, out);

    for (const auto& [name, param] : named_params) {
        char name_buf[128]{0};
        std::strncpy(name_buf, name.c_str(), 127);
        std::fwrite(name_buf, 128, 1, out);

        uint32_t ndim = static_cast<uint32_t>(param->ndim());
        std::fwrite(&ndim, 4, 1, out);

        uint32_t shape_buf[4]{1, 1, 1, 1};
        for (size_t d = 0; d < param->ndim() && d < 4; ++d) {
            shape_buf[d] = static_cast<uint32_t>(param->dim(d));
        }
        std::fwrite(shape_buf, 16, 1, out);

        uint64_t num_bytes = param->bytes();
        std::fwrite(&num_bytes, 8, 1, out);
        std::fwrite(param->data(), 1, num_bytes, out);
    }
    std::fclose(out);
    SOAR_LOG_INFO("Successfully saved {} model tensors to {}", num_tensors, path);
}

void SOARModel::load_weights(const std::string& path) {
    FILE* in = std::fopen(path.c_str(), "rb");
    if (!in) {
        throw DeviceError("Failed to open file for weight loading: " + path);
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t variant_id = 0;
    uint32_t num_tensors = 0;

    if (std::fread(&magic, 4, 1, in) != 1 ||
        std::fread(&version, 4, 1, in) != 1 ||
        std::fread(&variant_id, 4, 1, in) != 1 ||
        std::fread(&num_tensors, 4, 1, in) != 1) {
        std::fclose(in);
        throw DeviceError("Failed to read header from weight file: " + path);
    }

    if (magic != 0x534F4152) {
        std::fclose(in);
        throw DeviceError("Invalid magic header in weight file: " + path);
    }

    auto named_params = named_parameters();
    std::map<std::string, TensorPtr> param_map(named_params.begin(), named_params.end());

    size_t loaded_count = 0;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        char name_buf[128]{0};
        if (std::fread(name_buf, 128, 1, in) != 1) break;
        std::string name(name_buf);

        uint32_t ndim = 0;
        if (std::fread(&ndim, 4, 1, in) != 1) break;

        uint32_t shape_buf[4]{0};
        if (std::fread(shape_buf, 16, 1, in) != 1) break;

        uint64_t num_bytes = 0;
        if (std::fread(&num_bytes, 8, 1, in) != 1) break;

        auto it = param_map.find(name);
        if (it != param_map.end()) {
            if (it->second->bytes() == num_bytes) {
                if (std::fread(it->second->data(), 1, num_bytes, in) == num_bytes) {
                    loaded_count++;
                }
            } else {
                std::fseek(in, static_cast<long>(num_bytes), SEEK_CUR);
            }
        } else {
            std::fseek(in, static_cast<long>(num_bytes), SEEK_CUR);
        }
    }
    std::fclose(in);
    SOAR_LOG_INFO("Loaded {}/{} matching tensors from {}", loaded_count, named_params.size(), path);
}

} // namespace soar::nn
