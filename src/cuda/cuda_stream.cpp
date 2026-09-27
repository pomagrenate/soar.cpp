#include <soar/cuda/cuda_stream.hpp>
#include <soar/cuda/cuda_event.hpp>
#include <mutex>
#include <vector>

namespace soar::cuda {

// ============================================================
//  Stream pool — per-device, lazy init, round-robin
// ============================================================

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

StreamPool g_device_pools[kMaxDevices];

} // namespace

// ============================================================
//  CudaStream value-type methods
// ============================================================

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

// ============================================================
//  Thread-local current stream per device
//  (mirrors c10::cuda::getCurrentCUDAStream / setCurrentCUDAStream)
//
//  Each OS thread has its own "current stream" per device.
//  Initially points to the default (null) stream for every device.
//  CudaStreamGuard uses set/get to implement RAII stream switching.
// ============================================================

namespace {

// Thread-local current streams, indexed by device index
thread_local CudaStream tl_current_streams[kMaxDevices];
thread_local bool tl_initialized = false;

void ensure_tl_initialized() {
    if (!tl_initialized) {
        for (int d = 0; d < kMaxDevices; ++d) {
            tl_current_streams[d] = CudaStream::getDefaultStream(d);
        }
        tl_initialized = true;
    }
}

} // namespace

CudaStream getCurrentStream(int device_index) {
    ensure_tl_initialized();
    return tl_current_streams[device_index];
}

void setCurrentStream(CudaStream stream) {
    ensure_tl_initialized();
    int dev = stream.device_index();
    tl_current_streams[dev] = stream;
    // Also push to CUDA driver so async APIs use this stream by default
    // (cudaSetDevice doesn't change the stream, but callers may query it)
}

// ============================================================
//  CudaDeviceGuard implementation
//  Saves current device, sets requested device, restores on exit.
// ============================================================

CudaDeviceGuard::CudaDeviceGuard(int device_index) {
    SOAR_CUDA_CHECK(cudaGetDevice(&original_device_));
    if (device_index != original_device_) {
        SOAR_CUDA_CHECK(cudaSetDevice(device_index));
    }
}

CudaDeviceGuard::~CudaDeviceGuard() {
    // Restore original device on scope exit (best-effort; suppress error in dtor)
    cudaSetDevice(original_device_);
}

// ============================================================
//  CudaStreamGuard implementation
//  Saves current stream for the stream's device, activates new stream,
//  restores on exit. Also switches the active device.
// ============================================================

CudaStreamGuard::CudaStreamGuard(CudaStream stream)
    : current_stream_(stream) {
    int dev = stream.device_index();
    ensure_tl_initialized();
    original_stream_ = tl_current_streams[dev];

    // Switch device if needed
    int current_dev = 0;
    cudaGetDevice(&current_dev);
    if (dev != current_dev) {
        SOAR_CUDA_CHECK(cudaSetDevice(dev));
    }

    // Set the new current stream
    tl_current_streams[dev] = stream;
}

CudaStreamGuard::~CudaStreamGuard() {
    int dev = current_stream_.device_index();
    // Restore previous stream for this device
    tl_current_streams[dev] = original_stream_;
    // Restore device (best-effort)
    cudaSetDevice(original_stream_.device_index());
}

} // namespace soar::cuda
