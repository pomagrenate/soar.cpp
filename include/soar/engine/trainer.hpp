#pragma once

#include <soar/nn/soar_model.hpp>
#include <soar/losses/losses.hpp>
#include <soar/optim/adamw.hpp>
#include <soar/cuda/cuda_allocator.hpp>
#include <memory>
#include <vector>

namespace soar::engine {

struct StepMetrics {
    float loss{0.0f};
    float bce_loss{0.0f};
    float dice_loss{0.0f};
    float dice_score{0.0f};
    float iou_score{0.0f};
    float lr{0.0f};
};

/**
 * @brief Async staging buffer for H→D transfers.
 * 
 * Uses pinned host memory to enable DMA engine transfers
 * that can overlap with GPU compute.
 */
class AsyncStagingBuffer {
public:
    AsyncStagingBuffer(size_t buffer_size_bytes);
    ~AsyncStagingBuffer();
    
    void* host_ptr() const noexcept { return host_ptr_; }
    size_t size() const noexcept { return size_; }
    
private:
    void* host_ptr_{nullptr};
    size_t size_{0};
};

/**
 * @brief High-performance C++20 Segmentation Trainer for SOAR architectures.
 *        Supports gradient accumulation, proper epoch-level scheduler period,
 *        and numerical stabilization matching PyTorch / segres.
 *        Phase 3.1: Async H→D training pipeline with pinned staging buffers.
 */
class Trainer {
public:
    Trainer(std::shared_ptr<nn::SOARModel> model,
            losses::CompositeLoss loss_fn,
            optim::AdamW optimizer,
            std::unique_ptr<optim::CosineAnnealingLR> scheduler = nullptr,
            size_t accumulate_grad_batches = 1,
            float grad_clip = 2.0f);

    /**
     * @brief Train step supporting gradient accumulation.
     * @param images Input image tensor.
     * @param masks Ground truth mask tensor.
     * @param is_accumulating If true, gradients accumulate without calling optimizer.step().
     */
    StepMetrics train_step(const TensorPtr& images, const TensorPtr& masks, bool is_accumulating = false);

    /**
     * @brief Step learning rate scheduler once per epoch (training period).
     */
    void step_scheduler();

    StepMetrics evaluate_step(const TensorPtr& images, const TensorPtr& masks);

    [[nodiscard]] std::shared_ptr<nn::SOARModel> model() const noexcept { return model_; }
    [[nodiscard]] optim::AdamW& optimizer() noexcept { return optimizer_; }
    [[nodiscard]] const optim::AdamW& optimizer() const noexcept { return optimizer_; }

    [[nodiscard]] size_t accumulate_grad_batches() const noexcept { return accumulate_grad_batches_; }
    void set_accumulate_grad_batches(size_t n) noexcept { accumulate_grad_batches_ = std::max(size_t(1), n); }

    [[nodiscard]] float grad_clip() const noexcept { return grad_clip_; }
    void set_grad_clip(float clip) noexcept { grad_clip_ = clip; }

    [[nodiscard]] losses::CompositeLoss& loss_fn() noexcept { return loss_fn_; }
    
    /**
     * @brief Enable/disable async H→D pipeline with pinned staging buffers.
     * @param enable If true, uses pinned memory for async transfers (requires CUDA).
     */
    void enable_async_pipeline(bool enable) noexcept { async_pipeline_enabled_ = enable; }
    [[nodiscard]] bool async_pipeline_enabled() const noexcept { return async_pipeline_enabled_; }

private:
    StepMetrics compute_metrics(const TensorPtr& logits, const TensorPtr& masks, float loss_val, float bce_l, float dice_l);
    
    /**
     * @brief Async H→D transfer using pinned staging buffer.
     * Copies data from source tensor to pinned staging, then async to device.
     */
    void async_transfer_to_device(const TensorPtr& src, TensorPtr& dst, const std::shared_ptr<AsyncStagingBuffer>& staging);

    std::shared_ptr<nn::SOARModel> model_;
    losses::CompositeLoss loss_fn_;
    optim::AdamW optimizer_;
    std::unique_ptr<optim::CosineAnnealingLR> scheduler_;
    size_t accumulate_grad_batches_{1};
    float grad_clip_{2.0f};
    
    // Async pipeline state
    bool async_pipeline_enabled_{false};
    std::vector<std::shared_ptr<AsyncStagingBuffer>> staging_buffers_;
    size_t staging_buffer_index_{0};
};

} // namespace soar::engine
