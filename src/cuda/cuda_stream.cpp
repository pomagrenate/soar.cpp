#include <soar/cuda/cuda_stream.hpp>
#include <soar/cuda/cuda_event.hpp>
#include <mutex>
#include <vector>

namespace soar::cuda {

namespace {

struct StreamPool {
    std::array<std::array<cudaStream_t, CudaStream::kStreamsPerPool>, CudaStream::kMaxPriorities> streams{};
    std::array<std::atomic<uint32_t>, CudaStream::kMaxPriorities> counters{};
    std::once_flag init_flag;

    void init() {
        std::call_once(init_flag, [this]() {
            for (int p = 0; p < CudaStream::kMaxPriorities; ++p) {
                counters[p].store(0, std::memory_order_relaxed);
                int priority_val = (p == 1) ? -1 : 0; // -1 is higher priority in CUDA
                for (int i = 0; i < CudaStream::kStreamsPerPool; ++i) {
                    cudaStream_t s = nullptr;
                    SOAR_CUDA_CHECK(cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, priority_val));
                    streams[p][i] = s;
                }
            }
        });
    }
};

StreamPool g_device_pools[1]; // Expandable to max GPUs

} // namespace

CudaStream::CudaStream() noexcept
    : stream_(nullptr), device_index_(0), priority_(0) {}

CudaStream::CudaStream(cudaStream_t raw_stream, int device_index, int priority) noexcept
    : stream_(raw_stream), device_index_(device_index), priority_(priority) {}

void CudaStream::synchronize() const {
    SOAR_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool CudaStream::query() const noexcept {
    return cudaStreamQuery(stream_) == cudaSuccess;
}

void CudaStream::wait_event(const CudaEvent& event) const {
    event.block(*this);
}

CudaStream CudaStream::getDefaultStream(int device_index) {
    return CudaStream(nullptr, device_index, 0);
}

CudaStream CudaStream::getStreamFromPool(bool is_high_priority, int device_index) {
    auto& pool = g_device_pools[device_index];
    pool.init();
    int p = is_high_priority ? 1 : 0;
    uint32_t idx = pool.counters[p].fetch_add(1, std::memory_order_relaxed) % kStreamsPerPool;
    cudaStream_t s = pool.streams[p][idx];
    return CudaStream(s, device_index, is_high_priority ? 1 : 0);
}

CudaStream CudaStream::create(unsigned int flags, int priority, int device_index) {
    cudaStream_t raw = nullptr;
    SOAR_CUDA_CHECK(cudaStreamCreateWithPriority(&raw, flags, priority));
    return CudaStream(raw, device_index, priority);
}

} // namespace soar::cuda
