#include <soar/engine/trainer.hpp>
#include <soar/cuda/cuda_runtime.hpp>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>

namespace soar::engine {

Trainer::Trainer(std::shared_ptr<nn::SOARModel> model,
                 losses::CompositeLoss loss_fn,
                 optim::AdamW optimizer,
                 std::unique_ptr<optim::CosineAnnealingLR> scheduler,
                 size_t accumulate_grad_batches,
                 float grad_clip)
    : model_(std::move(model)), loss_fn_(loss_fn),
      optimizer_(std::move(optimizer)), scheduler_(std::move(scheduler)),
      accumulate_grad_batches_(std::max(size_t(1), accumulate_grad_batches)),
      grad_clip_(grad_clip) {}

StepMetrics Trainer::compute_metrics(const TensorPtr& logits, const TensorPtr& masks, float loss_val, float bce_l, float dice_l) {
    StepMetrics m;
    m.loss = loss_val;
    m.bce_loss = bce_l;
    m.dice_loss = dice_l;
    m.lr = optimizer_.get_lr();

    if (logits->is_cuda()) logits->sync_to_host();
    if (masks->is_cuda()) masks->sync_to_host();

    const float* z = logits->data();
    const float* y = masks->data();
    size_t n = logits->numel();

    double inter = 0.0;
    double sum_pred = 0.0;
    double sum_target = 0.0;

    #pragma omp parallel for reduction(+:inter, sum_pred, sum_target) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        float p = (z[i] >= 0.0f) ? 1.0f : 0.0f;
        float t = (y[i] >= 0.5f) ? 1.0f : 0.0f;
        if (p > 0.5f && t > 0.5f) inter += 1.0;
        if (p > 0.5f) sum_pred += 1.0;
        if (t > 0.5f) sum_target += 1.0;
    }

    double union_val = sum_pred + sum_target - inter;
    m.iou_score = static_cast<float>((inter + 1e-7) / (union_val + 1e-7));
    m.dice_score = static_cast<float>((2.0 * inter + 1e-7) / (sum_pred + sum_target + 1e-7));
    return m;
}

StepMetrics Trainer::train_step(const TensorPtr& images, const TensorPtr& masks, bool is_accumulating) {
    model_->train(true);

    size_t B = (images->ndim() == 4) ? images->dim(0) : 1;
    size_t C = (images->ndim() == 4) ? images->dim(1) : images->dim(0);
    size_t H = (images->ndim() == 4) ? images->dim(2) : images->dim(1);
    size_t W = (images->ndim() == 4) ? images->dim(3) : images->dim(2);

    if (B <= 1) {
        TensorPtr img = images;
        TensorPtr msk = masks;
        if (images->ndim() == 4) {
            img = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, images->is_cuda());
            if (images->is_cuda()) {
                soar::cuda::cudaMemcpyAsync(img->cuda_data(), images->cuda_data(), C * H * W * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
            } else {
                std::memcpy(img->data(), images->data(), C * H * W * sizeof(float));
            }
        }
        if (masks->ndim() == 4) {
            size_t mc = masks->dim(1);
            msk = Tensor::create({static_cast<int64_t>(mc), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, masks->is_cuda());
            if (masks->is_cuda()) {
                soar::cuda::cudaMemcpyAsync(msk->cuda_data(), masks->cuda_data(), mc * H * W * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
            } else {
                std::memcpy(msk->data(), masks->data(), mc * H * W * sizeof(float));
            }
        }

        if (model_->is_cuda()) {
            if (!img->is_cuda()) img->to_cuda();
            if (!msk->is_cuda()) msk->to_cuda();
        }

        img->set_requires_grad(false);
        TensorPtr logits = model_->forward(img);

        float bce_l = 0.0f;
        float dice_l = 0.0f;
        TensorPtr loss = loss_fn_.forward(logits, msk, bce_l, dice_l);

        float scale = 1.0f / static_cast<float>(accumulate_grad_batches_);
        TensorPtr grad_out = Tensor::create({1});
        grad_out->item() = scale;
        loss->backward(grad_out);

        if (!is_accumulating) {
            optimizer_.clip_grad_norm(grad_clip_);
            optimizer_.step();
            optimizer_.zero_grad();
        }

        StepMetrics m = compute_metrics(logits, msk, loss->item(), bce_l, dice_l);
        logits->set_grad_fn(nullptr);
        loss->set_grad_fn(nullptr);
        return m;
    }

    // Multi-sample batch: process micro-batches of 1 sample to strictly cap VRAM
    double total_loss = 0.0;
    double total_bce = 0.0;
    double total_dice = 0.0;
    double total_iou = 0.0;
    double total_dice_score = 0.0;

    size_t img_sample_numel = C * H * W;
    size_t msk_c = (masks->ndim() == 4) ? masks->dim(1) : 1;
    size_t msk_sample_numel = msk_c * H * W;

    for (size_t b = 0; b < B; ++b) {
        TensorPtr img_b = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, images->is_cuda());
        if (images->is_cuda()) {
            soar::cuda::cudaMemcpyAsync(img_b->cuda_data(), images->cuda_data() + b * img_sample_numel, img_sample_numel * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
        } else {
            std::memcpy(img_b->data(), images->data() + b * img_sample_numel, img_sample_numel * sizeof(float));
        }

        TensorPtr msk_b = Tensor::create({static_cast<int64_t>(msk_c), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, masks->is_cuda());
        if (masks->is_cuda()) {
            soar::cuda::cudaMemcpyAsync(msk_b->cuda_data(), masks->cuda_data() + b * msk_sample_numel, msk_sample_numel * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
        } else {
            std::memcpy(msk_b->data(), masks->data() + b * msk_sample_numel, msk_sample_numel * sizeof(float));
        }

        if (model_->is_cuda()) {
            if (!img_b->is_cuda()) img_b->to_cuda();
            if (!msk_b->is_cuda()) msk_b->to_cuda();
        }

        img_b->set_requires_grad(false);
        TensorPtr logits_b = model_->forward(img_b);

        float bce_sub = 0.0f;
        float dice_sub = 0.0f;
        TensorPtr loss_b = loss_fn_.forward(logits_b, msk_b, bce_sub, dice_sub);

        float scale = 1.0f / static_cast<float>(B * accumulate_grad_batches_);
        TensorPtr grad_out = Tensor::create({1});
        grad_out->item() = scale;
        loss_b->backward(grad_out);

        StepMetrics sub_m = compute_metrics(logits_b, msk_b, loss_b->item(), bce_sub, dice_sub);
        total_loss += sub_m.loss;
        total_bce += sub_m.bce_loss;
        total_dice += sub_m.dice_loss;
        total_iou += sub_m.iou_score;
        total_dice_score += sub_m.dice_score;

        logits_b->set_grad_fn(nullptr);
        loss_b->set_grad_fn(nullptr);
    }

    if (!is_accumulating) {
        optimizer_.clip_grad_norm(grad_clip_);
        optimizer_.step();
        optimizer_.zero_grad();
    }

    StepMetrics m;
    m.loss = static_cast<float>(total_loss / static_cast<double>(B));
    m.bce_loss = static_cast<float>(total_bce / static_cast<double>(B));
    m.dice_loss = static_cast<float>(total_dice / static_cast<double>(B));
    m.iou_score = static_cast<float>(total_iou / static_cast<double>(B));
    m.dice_score = static_cast<float>(total_dice_score / static_cast<double>(B));
    m.lr = optimizer_.get_lr();
    return m;
}

void Trainer::step_scheduler() {
    if (scheduler_) {
        scheduler_->step();
    }
}

StepMetrics Trainer::evaluate_step(const TensorPtr& images, const TensorPtr& masks) {
    model_->eval();

    size_t B = (images->ndim() == 4) ? images->dim(0) : 1;
    size_t C = (images->ndim() == 4) ? images->dim(1) : images->dim(0);
    size_t H = (images->ndim() == 4) ? images->dim(2) : images->dim(1);
    size_t W = (images->ndim() == 4) ? images->dim(3) : images->dim(2);

    if (B <= 1) {
        TensorPtr img = images;
        TensorPtr msk = masks;
        if (images->ndim() == 4) {
            img = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, images->is_cuda());
            if (images->is_cuda()) {
                soar::cuda::cudaMemcpyAsync(img->cuda_data(), images->cuda_data(), C * H * W * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
            } else {
                std::memcpy(img->data(), images->data(), C * H * W * sizeof(float));
            }
        }
        if (masks->ndim() == 4) {
            size_t mc = masks->dim(1);
            msk = Tensor::create({static_cast<int64_t>(mc), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, masks->is_cuda());
            if (masks->is_cuda()) {
                soar::cuda::cudaMemcpyAsync(msk->cuda_data(), masks->cuda_data(), mc * H * W * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
            } else {
                std::memcpy(msk->data(), masks->data(), mc * H * W * sizeof(float));
            }
        }

        if (model_->is_cuda()) {
            if (!img->is_cuda()) img->to_cuda();
            if (!msk->is_cuda()) msk->to_cuda();
        }

        TensorPtr logits = model_->forward(img);
        float bce_l = 0.0f;
        float dice_l = 0.0f;
        TensorPtr loss = loss_fn_.forward(logits, msk, bce_l, dice_l);
        return compute_metrics(logits, msk, loss->item(), bce_l, dice_l);
    }

    double total_loss = 0.0;
    double total_bce = 0.0;
    double total_dice = 0.0;
    double total_iou = 0.0;
    double total_dice_score = 0.0;

    size_t img_sample_numel = C * H * W;
    size_t msk_c = (masks->ndim() == 4) ? masks->dim(1) : 1;
    size_t msk_sample_numel = msk_c * H * W;

    for (size_t b = 0; b < B; ++b) {
        TensorPtr img_b = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, images->is_cuda());
        if (images->is_cuda()) {
            soar::cuda::cudaMemcpyAsync(img_b->cuda_data(), images->cuda_data() + b * img_sample_numel, img_sample_numel * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
        } else {
            std::memcpy(img_b->data(), images->data() + b * img_sample_numel, img_sample_numel * sizeof(float));
        }

        TensorPtr msk_b = Tensor::create({static_cast<int64_t>(msk_c), static_cast<int64_t>(H), static_cast<int64_t>(W)}, false, masks->is_cuda());
        if (masks->is_cuda()) {
            soar::cuda::cudaMemcpyAsync(msk_b->cuda_data(), masks->cuda_data() + b * msk_sample_numel, msk_sample_numel * sizeof(float), soar::cuda::cudaMemcpyDeviceToDevice);
        } else {
            std::memcpy(msk_b->data(), masks->data() + b * msk_sample_numel, msk_sample_numel * sizeof(float));
        }

        if (model_->is_cuda()) {
            if (!img_b->is_cuda()) img_b->to_cuda();
            if (!msk_b->is_cuda()) msk_b->to_cuda();
        }

        TensorPtr logits_b = model_->forward(img_b);
        float bce_sub = 0.0f;
        float dice_sub = 0.0f;
        TensorPtr loss_b = loss_fn_.forward(logits_b, msk_b, bce_sub, dice_sub);

        StepMetrics sub_m = compute_metrics(logits_b, msk_b, loss_b->item(), bce_sub, dice_sub);
        total_loss += sub_m.loss;
        total_bce += sub_m.bce_loss;
        total_dice += sub_m.dice_loss;
        total_iou += sub_m.iou_score;
        total_dice_score += sub_m.dice_score;
    }

    StepMetrics m;
    m.loss = static_cast<float>(total_loss / static_cast<double>(B));
    m.bce_loss = static_cast<float>(total_bce / static_cast<double>(B));
    m.dice_loss = static_cast<float>(total_dice / static_cast<double>(B));
    m.iou_score = static_cast<float>(total_iou / static_cast<double>(B));
    m.dice_score = static_cast<float>(total_dice_score / static_cast<double>(B));
    m.lr = optimizer_.get_lr();
    return m;
}

} // namespace soar::engine
