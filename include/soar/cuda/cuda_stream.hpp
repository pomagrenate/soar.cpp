#pragma once

#include <soar/cuda/cuda_runtime.hpp>
#include <cstdint>
#include <array>
#include <atomic>
#include <memory>

namespace soar::cuda {

class CudaEvent;

/**
 * @brief C++20 CUDA Stream abstraction mirroring PyTorch c10::cuda::CUDAStream.
 *
 * Provides non-blocking streams (`cudaStreamNonBlocking`), stream pooling
 * (32 streams per device per priority level), round-robin dispatch,
 * and lightweight stream synchronization via wait_event.
 */
class CudaStream {
public:
    static constexpr int kStreamsPerPool = 32;
    static constexpr int kMaxPriorities = 2; // 0 = default/low, 1 = high

    CudaStream() noexcept;
    explicit CudaStream(cudaStream_t raw_stream, int device_index = 0, int priority = 0) noexcept;
    ~CudaStream() = default;

    // Movable, copyable value-type semantics (like PyTorch c10::cuda::CUDAStream)
    CudaStream(const CudaStream&) noexcept = default;
    CudaStream& operator=(const CudaStream&) noexcept = default;
    CudaStream(CudaStream&&) noexcept = default;
    CudaStream& operator=(CudaStream&&) noexcept = default;

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }
    operator cudaStream_t() const noexcept { return stream_; }

    [[nodiscard]] int device_index() const noexcept { return device_index_; }
    [[nodiscard]] int priority() const noexcept { return priority_; }

    void synchronize() const;
    [[nodiscard]] bool query() const noexcept;
    void wait_event(const CudaEvent& event) const;

    bool operator==(const CudaStream& other) const noexcept {
        return stream_ == other.stream_ && device_index_ == other.device_index_;
    }
    bool operator!=(const CudaStream& other) const noexcept {
        return !(*this == other);
    }

    // Factory methods
    static CudaStream getDefaultStream(int device_index = 0);
    static CudaStream getStreamFromPool(bool is_high_priority = false, int device_index = 0);
    static CudaStream create(unsigned int flags = cudaStreamNonBlocking, int priority = 0, int device_index = 0);

private:
    cudaStream_t stream_{nullptr};
    int device_index_{0};
    int priority_{0};
};

} // namespace soar::cuda
