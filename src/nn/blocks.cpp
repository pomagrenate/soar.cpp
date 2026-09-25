#include <soar/nn/blocks.hpp>
#include <soar/autograd/node.hpp>
#include <soar/core/logging.hpp>

#include <cmath>
#include <cstring>
#include <algorithm>

namespace soar::nn {

// -------------------------------------------------------------
// Math Helper Autograd Nodes
// -------------------------------------------------------------
struct AddNode : public AutogradNode {
    TensorPtr a;
    TensorPtr b;
    void backward(const TensorPtr& grad_output) override {
        if (a && a->requires_grad()) {
            accumulate_grad(a->grad(), grad_output);
            if (a->grad_fn()) a->grad_fn()->backward(grad_output);
        }
        if (b && b->requires_grad()) {
            accumulate_grad(b->grad(), grad_output);
            if (b->grad_fn()) b->grad_fn()->backward(grad_output);
        }
    }
};

TensorPtr add_tensors(const TensorPtr& a, const TensorPtr& b) {
    if (a->shape() != b->shape()) {
        throw ShapeError("Cannot add tensors with different shapes");
    }
    TensorPtr output = Tensor::create(a->shape(), a->requires_grad() || b->requires_grad());
    const float* p_a = a->data();
    const float* p_b = b->data();
    float* p_y = output->data();
    size_t n = a->numel();
    for (size_t i = 0; i < n; ++i) {
        p_y[i] = p_a[i] + p_b[i];
    }
    if (output->requires_grad()) {
        auto node = std::make_shared<AddNode>();
        node->a = a;
        node->b = b;
        output->set_grad_fn(node);
    }
    return output;
}

struct MulBroadcastNode : public AutogradNode {
    TensorPtr a; // [C, H, W]
    TensorPtr b; // [C, 1, 1]
    void backward(const TensorPtr& grad_output) override {
        size_t C = a->dim(0);
        size_t H = a->dim(1);
        size_t W = a->dim(2);
        const float* go = grad_output->data();
        const float* p_a = a->data();
        const float* p_b = b->data();

        if (a->requires_grad()) {
            TensorPtr grad_a = Tensor::zeros(a->shape());
            float* ga = grad_a->data();
            for (size_t c = 0; c < C; ++c) {
                float b_val = p_b[c];
                for (size_t sp = 0; sp < H * W; ++sp) {
                    ga[c * (H * W) + sp] = go[c * (H * W) + sp] * b_val;
                }
            }
            accumulate_grad(a->grad(), grad_a);
            if (a->grad_fn()) a->grad_fn()->backward(grad_a);
        }

        if (b->requires_grad()) {
            TensorPtr grad_b = Tensor::zeros(b->shape());
            float* gb = grad_b->data();
            for (size_t c = 0; c < C; ++c) {
                float sum = 0.0f;
                for (size_t sp = 0; sp < H * W; ++sp) {
                    sum += go[c * (H * W) + sp] * p_a[c * (H * W) + sp];
                }
                gb[c] = sum;
            }
            accumulate_grad(b->grad(), grad_b);
            if (b->grad_fn()) b->grad_fn()->backward(grad_b);
        }
    }
};

TensorPtr mul_tensors(const TensorPtr& a, const TensorPtr& b) {
    size_t C = a->dim(0);
    size_t H = a->dim(1);
    size_t W = a->dim(2);

    TensorPtr output = Tensor::create(a->shape(), a->requires_grad() || b->requires_grad());
    const float* p_a = a->data();
    const float* p_b = b->data();
    float* p_y = output->data();

    if (b->numel() == C && (b->ndim() == 1 || (b->dim(1) == 1 && b->dim(2) == 1))) {
        // Channel-wise broadcast
        for (size_t c = 0; c < C; ++c) {
            float b_val = p_b[c];
            for (size_t sp = 0; sp < H * W; ++sp) {
                p_y[c * (H * W) + sp] = p_a[c * (H * W) + sp] * b_val;
            }
        }
        if (output->requires_grad()) {
            auto node = std::make_shared<MulBroadcastNode>();
            node->a = a;
            node->b = b;
            output->set_grad_fn(node);
        }
    } else {
        // Elementwise
        size_t n = a->numel();
        for (size_t i = 0; i < n; ++i) {
            p_y[i] = p_a[i] * p_b[i];
        }
    }
    return output;
}

struct ConvexNode : public AutogradNode {
    TensorPtr g;
    TensorPtr a;
    TensorPtr b;
    void backward(const TensorPtr& grad_output) override {
        const float* go = grad_output->data();
        const float* p_g = g->data();
        const float* p_a = a->data();
        const float* p_b = b->data();
        size_t n = a->numel();

        if (a->requires_grad()) {
            TensorPtr grad_a = Tensor::zeros(a->shape());
            float* ga = grad_a->data();
            for (size_t i = 0; i < n; ++i) ga[i] = go[i] * p_g[i];
            accumulate_grad(a->grad(), grad_a);
            if (a->grad_fn()) a->grad_fn()->backward(grad_a);
        }
        if (b->requires_grad()) {
            TensorPtr grad_b = Tensor::zeros(b->shape());
            float* gb = grad_b->data();
            for (size_t i = 0; i < n; ++i) gb[i] = go[i] * (1.0f - p_g[i]);
            accumulate_grad(b->grad(), grad_b);
            if (b->grad_fn()) b->grad_fn()->backward(grad_b);
        }
        if (g->requires_grad()) {
            TensorPtr grad_g = Tensor::zeros(g->shape());
            float* gg = grad_g->data();
            for (size_t i = 0; i < n; ++i) gg[i] = go[i] * (p_a[i] - p_b[i]);
            accumulate_grad(g->grad(), grad_g);
            if (g->grad_fn()) g->grad_fn()->backward(grad_g);
        }
    }
};

TensorPtr convex_combination(const TensorPtr& g, const TensorPtr& a, const TensorPtr& b) {
    TensorPtr output = Tensor::create(a->shape(), g->requires_grad() || a->requires_grad() || b->requires_grad());
    const float* p_g = g->data();
    const float* p_a = a->data();
    const float* p_b = b->data();
    float* p_y = output->data();
    size_t n = a->numel();

    for (size_t i = 0; i < n; ++i) {
        p_y[i] = p_g[i] * p_a[i] + (1.0f - p_g[i]) * p_b[i];
    }

    if (output->requires_grad()) {
        auto node = std::make_shared<ConvexNode>();
        node->g = g;
        node->a = a;
        node->b = b;
        output->set_grad_fn(node);
    }
    return output;
}

struct ConcatNode : public AutogradNode {
    std::vector<TensorPtr> inputs;
    void backward(const TensorPtr& grad_output) override {
        const float* go = grad_output->data();
        size_t H = inputs[0]->dim(1);
        size_t W = inputs[0]->dim(2);

        size_t c_offset = 0;
        for (auto& inp : inputs) {
            size_t c_curr = inp->dim(0);
            if (inp->requires_grad()) {
                TensorPtr grad_i = Tensor::zeros(inp->shape());
                float* gi = grad_i->data();
                for (size_t c = 0; c < c_curr; ++c) {
                    std::memcpy(gi + c * (H * W), go + (c_offset + c) * (H * W), H * W * sizeof(float));
                }
                accumulate_grad(inp->grad(), grad_i);
                if (inp->grad_fn()) inp->grad_fn()->backward(grad_i);
            }
            c_offset += c_curr;
        }
    }
};

TensorPtr concat_channels(const std::vector<TensorPtr>& tensors) {
    if (tensors.empty()) throw ShapeError("Cannot concatenate empty tensor list");
    size_t H = tensors[0]->dim(1);
    size_t W = tensors[0]->dim(2);
    size_t total_c = 0;
    bool req_grad = false;

    for (const auto& t : tensors) {
        total_c += t->dim(0);
        if (t->requires_grad()) req_grad = true;
    }

    TensorPtr output = Tensor::create({static_cast<int64_t>(total_c), static_cast<int64_t>(H), static_cast<int64_t>(W)}, req_grad);
    float* out_data = output->data();

    size_t c_offset = 0;
    for (const auto& t : tensors) {
        size_t c_curr = t->dim(0);
        const float* in_data = t->data();
        for (size_t c = 0; c < c_curr; ++c) {
            std::memcpy(out_data + (c_offset + c) * (H * W), in_data + c * (H * W), H * W * sizeof(float));
        }
        c_offset += c_curr;
    }

    if (req_grad) {
        auto node = std::make_shared<ConcatNode>();
        node->inputs = tensors;
        output->set_grad_fn(node);
    }

    return output;
}

TensorPtr resize_bilinear(const TensorPtr& input, size_t target_h, size_t target_w) {
    if (input->dim(1) == target_h && input->dim(2) == target_w) {
        return input;
    }
    float scale = float(target_h) / float(input->dim(1));
    auto up = std::make_shared<Upsample>(scale);
    return up->forward(input);
}

// -------------------------------------------------------------
// CBA Implementation
// -------------------------------------------------------------
CBA::CBA(size_t c1, size_t c2, size_t k, size_t s, int p, size_t d, size_t g, bool act)
    : Module("CBA") {
    conv_ = std::make_shared<Conv2d>(c1, c2, k, s, p, d, g, false);
    norm_ = std::make_shared<GroupNorm>(select_group_count(static_cast<int>(c2)), c2);
    register_submodule("conv", conv_);
    register_submodule("norm", norm_);

    if (act) {
        act_ = std::make_shared<SiLU>();
        register_submodule("act", act_);
    }
}

TensorPtr CBA::forward(const TensorPtr& input) {
    TensorPtr y = norm_->forward(conv_->forward(input));
    return act_ ? act_->forward(y) : y;
}

// -------------------------------------------------------------
// Down Implementation
// -------------------------------------------------------------
Down::Down(size_t c1, size_t c2) : Module("Down") {
    dw_ = std::make_shared<CBA>(c1, c1, 3, 2, -1, 1, c1, false);
    pw_ = std::make_shared<CBA>(c1, c2, 1, 1, -1, 1, 1, true);
    register_submodule("dw", dw_);
    register_submodule("pw", pw_);
}

TensorPtr Down::forward(const TensorPtr& input) {
    return pw_->forward(dw_->forward(input));
}

// -------------------------------------------------------------
// LKR Implementation
// -------------------------------------------------------------
LKR::LKR(size_t c, size_t k, float e) : Module("LKR") {
    size_t h = std::max(static_cast<size_t>(c * e), size_t(8));
    dw_ = std::make_shared<CBA>(c, c, k, 1, -1, 1, c, false);
    pw1_ = std::make_shared<CBA>(c, h, 1, 1, -1, 1, 1, true);
    pw2_ = std::make_shared<CBA>(h, c, 1, 1, -1, 1, 1, false);

    // Identity initialization: zero-out pw2 norm weights
    pw2_->norm()->weight()->zero_();

    register_submodule("dw", dw_);
    register_submodule("pw1", pw1_);
    register_submodule("pw2", pw2_);
}

TensorPtr LKR::forward(const TensorPtr& input) {
    return add_tensors(input, pw2_->forward(pw1_->forward(dw_->forward(input))));
}

// -------------------------------------------------------------
// Ctx Implementation
// -------------------------------------------------------------
Ctx::Ctx(size_t c1, size_t c2, const std::vector<size_t>& dilations)
    : Module("Ctx"), c1_(c1), c2_(c2), add_(c1 == c2) {
    size_t h = c2 / 2;
    cv1_ = std::make_shared<CBA>(c1, h, 1);
    register_submodule("cv1", cv1_);

    for (size_t i = 0; i < dilations.size(); ++i) {
        auto branch = std::make_shared<CBA>(h, h, 3, 1, -1, dilations[i], h, false);
        register_submodule("branch_" + std::to_string(i), branch);
        branches_.push_back(branch);
    }

    cv2_ = std::make_shared<CBA>(h * (1 + dilations.size()), c2, 1);
    register_submodule("cv2", cv2_);

    pool_ = std::make_shared<AdaptiveAvgPool2d>();
    gate_conv_ = std::make_shared<Conv2d>(c1, c2, 1, 1, 0, 1, 1, true);
    gate_sig_ = std::make_shared<Sigmoid>();
    register_submodule("gate_conv", gate_conv_);
}

TensorPtr Ctx::forward(const TensorPtr& input) {
    TensorPtr y = cv1_->forward(input);
    std::vector<TensorPtr> cat_list = { y };
    for (auto& b : branches_) {
        cat_list.push_back(b->forward(y));
    }
    TensorPtr out = cv2_->forward(concat_channels(cat_list));
    TensorPtr gate = gate_sig_->forward(gate_conv_->forward(pool_->forward(input)));
    TensorPtr gated_out = mul_tensors(out, gate);
    return add_ ? add_tensors(input, gated_out) : gated_out;
}

// -------------------------------------------------------------
// Fuse Implementation
// -------------------------------------------------------------
Fuse::Fuse(size_t c_low, size_t c_skip, size_t c2) : Module("Fuse") {
    low_ = std::make_shared<CBA>(c_low, c2, 1, 1, -1, 1, 1, false);
    skip_ = std::make_shared<CBA>(c_skip, c2, 1, 1, -1, 1, 1, false);
    gate_dw_ = std::make_shared<CBA>(c2, c2, 3, 1, -1, 1, c2, false);
    gate_conv_ = std::make_shared<Conv2d>(c2, c2, 1, 1, 0, 1, 1, true);
    gate_sig_ = std::make_shared<Sigmoid>();
    out_dw_ = std::make_shared<CBA>(c2, c2, 3, 1, -1, 1, c2, false);
    out_pw_ = std::make_shared<CBA>(c2, c2, 1, 1, -1, 1, 1, true);

    register_submodule("low", low_);
    register_submodule("skip", skip_);
    register_submodule("gate_dw", gate_dw_);
    register_submodule("gate_conv", gate_conv_);
    register_submodule("out_dw", out_dw_);
    register_submodule("out_pw", out_pw_);
}

TensorPtr Fuse::forward(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() < 2) throw ShapeError("Fuse expects 2 input tensors: [low, skip]");
    TensorPtr low_x = inputs[0];
    TensorPtr skip_x = inputs[1];

    TensorPtr a = skip_->forward(skip_x);
    TensorPtr b = resize_bilinear(low_->forward(low_x), a->dim(1), a->dim(2));

    TensorPtr sum_ab = add_tensors(a, b);
    TensorPtr gate = gate_sig_->forward(gate_conv_->forward(gate_dw_->forward(sum_ab)));
    TensorPtr fused = convex_combination(gate, a, b);

    return out_pw_->forward(out_dw_->forward(fused));
}

// -------------------------------------------------------------
// Agg Implementation
// -------------------------------------------------------------
Agg::Agg(const std::vector<size_t>& chs, size_t c2) : Module("Agg") {
    size_t p = std::max(c2 / 2, size_t(8));
    for (size_t i = 0; i < chs.size(); ++i) {
        auto pr = std::make_shared<CBA>(chs[i], p, 1);
        register_submodule("proj_" + std::to_string(i), pr);
        proj_.push_back(pr);
    }
    fuse_ = std::make_shared<CBA>(p * chs.size(), c2, 1);
    register_submodule("fuse", fuse_);
}

TensorPtr Agg::forward(const std::vector<TensorPtr>& inputs) {
    if (inputs.size() != proj_.size()) throw ShapeError("Agg inputs size mismatch");
    size_t target_h = inputs[0]->dim(1);
    size_t target_w = inputs[0]->dim(2);

    std::vector<TensorPtr> ys;
    for (size_t i = 0; i < inputs.size(); ++i) {
        TensorPtr y = proj_[i]->forward(inputs[i]);
        if (y->dim(1) != target_h || y->dim(2) != target_w) {
            y = resize_bilinear(y, target_h, target_w);
        }
        ys.push_back(y);
    }

    return fuse_->forward(concat_channels(ys));
}

// -------------------------------------------------------------
// SegHead Implementation
// -------------------------------------------------------------
SegHead::SegHead(size_t c1, size_t nc, size_t mid, size_t r, float prior)
    : Module("SegHead") {
    refine_dw_ = std::make_shared<CBA>(c1, c1, 3, 1, -1, 1, c1, false);
    refine_pw_ = std::make_shared<CBA>(c1, mid, 1, 1, -1, 1, 1, true);
    pred_ = std::make_shared<Conv2d>(mid, nc * r * r, 1, 1, 0, 1, 1, true);
    shuffle_ = std::make_shared<PixelShuffle>(r);

    if (prior > 0.0f) {
        float b_init = -std::log((1.0f - prior) / prior);
        pred_->bias()->fill_(b_init);
    }

    register_submodule("refine_dw", refine_dw_);
    register_submodule("refine_pw", refine_pw_);
    register_submodule("pred", pred_);
}

TensorPtr SegHead::forward(const TensorPtr& input) {
    return shuffle_->forward(pred_->forward(refine_pw_->forward(refine_dw_->forward(input))));
}

} // namespace soar::nn
