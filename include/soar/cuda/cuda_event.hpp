#pragma once

#include <soar/cuda/cuda_runtime.hpp>
#include <soar/cuda/cuda_stream.hpp>
#include <utility>
#include <optional>

namespace soar::cuda {

/**
 * @brief C++20 CUDA Event abstraction mirroring PyTorch c10::cuda::CUDAEvent.
 *
 * Movable RAII wrapper around cudaEvent_t. Lazily created on first record.
 * Supports cross-stream synchronization via block(stream) using cudaStreamWaitEvent.
 */
class CudaEvent {
public:
    explicit CudaEvent(unsigned int flags = cudaEventDisableTiming) noexcept
        : flags_(flags) {}

    ~CudaEvent() {
        destroy();
    }

    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    CudaEvent(CudaEvent&& other) noexcept {
        move_from(std::move(other));
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            destroy();
            move_from(std::move(other));
        }
        return *this;
    }

    [[nodiscard]] cudaEvent_t event() const noexcept { return event_; }
    operator cudaEvent_t() const noexcept { return event_; }

    [[nodiscard]] bool is_created() const noexcept { return is_created_; }
    [[nodiscard]] int device_index() const noexcept { return device_index_; }
    [[nodiscard]] unsigned int flags() const noexcept { return flags_; }

    void record(const CudaStream& stream = CudaStream::getDefaultStream());
    void record_once(const CudaStream& stream = CudaStream::getDefaultStream());
    void block(const CudaStream& stream) const;
    void synchronize() const;
    [[nodiscard]] bool query() const noexcept;
    [[nodiscard]] float elapsed_time(const CudaEvent& end) const;

private:
    void ensure_created(int device_index);
    void destroy() noexcept;
    void move_from(CudaEvent&& other) noexcept;

    cudaEvent_t event_{nullptr};
    unsigned int flags_{cudaEventDisableTiming};
    int device_index_{0};
    bool is_created_{false};
    bool was_recorded_{false};
};

} // namespace soar::cuda
