#include <soar/cuda/cuda_runtime.hpp>
#include <chrono>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <algorithm>
#include "palloc.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace soar::cuda {

namespace {

struct InternalEvent {
    unsigned int flags{0};
    std::atomic<bool> recorded{false};
    std::chrono::high_resolution_clock::time_point timestamp;
};

struct InternalStream {
    unsigned int flags{0};
    int priority{0};
    std::atomic<bool> busy{false};
};

std::mutex g_cuda_mutex;
int g_current_device = 0;

} // namespace

const char* cudaGetErrorString(cudaError_t error) noexcept {
    switch (error) {
        case cudaSuccess: return "cudaSuccess: no errors";
        case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation: out of memory";
        case cudaErrorInitializationError: return "cudaErrorInitializationError";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
        case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice";
        case cudaErrorInvalidValue: return "cudaErrorInvalidValue";
        case cudaErrorInvalidPitchValue: return "cudaErrorInvalidPitchValue";
        case cudaErrorInvalidSymbol: return "cudaErrorInvalidSymbol";
        case cudaErrorNotReady: return "cudaErrorNotReady: event/stream not ready";
        default: return "cudaErrorUnknown: unspecified error";
    }
}

cudaError_t cudaGetLastError() noexcept {
    return cudaSuccess;
}

cudaError_t cudaGetDeviceCount(int* count) noexcept {
    if (!count) return cudaErrorInvalidValue;
    *count = 1;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) noexcept {
    if (device < 0 || device >= 1) return cudaErrorInvalidDevice;
    g_current_device = device;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) noexcept {
    if (!device) return cudaErrorInvalidValue;
    *device = g_current_device;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) noexcept {
    if (!pStream) return cudaErrorInvalidValue;
    auto s = new (std::nothrow) InternalStream();
    if (!s) return cudaErrorMemoryAllocation;
    s->flags = flags;
    s->priority = 0;
    *pStream = reinterpret_cast<cudaStream_t>(s);
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) noexcept {
    if (!pStream) return cudaErrorInvalidValue;
    auto s = new (std::nothrow) InternalStream();
    if (!s) return cudaErrorMemoryAllocation;
    s->flags = flags;
    s->priority = priority;
    *pStream = reinterpret_cast<cudaStream_t>(s);
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) noexcept {
    if (!stream) return cudaSuccess;
    auto s = reinterpret_cast<InternalStream*>(stream);
    delete s;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) noexcept {
    if (stream) {
        auto s = reinterpret_cast<InternalStream*>(stream);
        s->busy.store(false, std::memory_order_release);
    }
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) noexcept {
    if (!stream) return cudaSuccess;
    auto s = reinterpret_cast<InternalStream*>(stream);
    return s->busy.load(std::memory_order_acquire) ? cudaErrorNotReady : cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int /*flags*/) noexcept {
    if (!event) return cudaErrorInvalidValue;
    auto e = reinterpret_cast<InternalEvent*>(event);
    if (!e->recorded.load(std::memory_order_acquire)) {
        return cudaErrorNotReady;
    }
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* phEvent, unsigned int flags) noexcept {
    if (!phEvent) return cudaErrorInvalidValue;
    auto e = new (std::nothrow) InternalEvent();
    if (!e) return cudaErrorMemoryAllocation;
    e->flags = flags;
    e->recorded.store(false, std::memory_order_relaxed);
    *phEvent = reinterpret_cast<cudaEvent_t>(e);
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t hEvent) noexcept {
    if (!hEvent) return cudaSuccess;
    auto e = reinterpret_cast<InternalEvent*>(hEvent);
    delete e;
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t hEvent, cudaStream_t stream) noexcept {
    if (!hEvent) return cudaErrorInvalidValue;
    auto e = reinterpret_cast<InternalEvent*>(hEvent);
    e->timestamp = std::chrono::high_resolution_clock::now();
    e->recorded.store(true, std::memory_order_release);
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t hEvent) noexcept {
    if (!hEvent) return cudaErrorInvalidValue;
    auto e = reinterpret_cast<InternalEvent*>(hEvent);
    if (!e->recorded.load(std::memory_order_acquire)) {
        return cudaErrorNotReady;
    }
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t hEvent) noexcept {
    if (!hEvent) return cudaErrorInvalidValue;
    auto e = reinterpret_cast<InternalEvent*>(hEvent);
    return e->recorded.load(std::memory_order_acquire) ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) noexcept {
    if (!ms || !start || !end) return cudaErrorInvalidValue;
    auto e_start = reinterpret_cast<InternalEvent*>(start);
    auto e_end = reinterpret_cast<InternalEvent*>(end);
    if (!e_start->recorded.load(std::memory_order_acquire) || !e_end->recorded.load(std::memory_order_acquire)) {
        return cudaErrorNotReady;
    }
    auto duration = std::chrono::duration<float, std::milli>(e_end->timestamp - e_start->timestamp);
    *ms = duration.count();
    return cudaSuccess;
}

cudaError_t cudaMalloc(void** devPtr, size_t size) noexcept {
    if (!devPtr) return cudaErrorInvalidValue;
    if (size == 0) {
        *devPtr = nullptr;
        return cudaSuccess;
    }
    // High performance 64-byte aligned allocation via palloc
    void* ptr = ::pa_malloc_aligned(size, 64);
    if (!ptr) return cudaErrorMemoryAllocation;
    *devPtr = ptr;
    return cudaSuccess;
}

cudaError_t cudaFree(void* devPtr) noexcept {
    if (!devPtr) return cudaSuccess;
    ::pa_free_aligned(devPtr, 64);
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int /*flags*/) noexcept {
    if (!pHost) return cudaErrorInvalidValue;
    if (size == 0) {
        *pHost = nullptr;
        return cudaSuccess;
    }
    // Allocate 64-byte aligned pinned host memory
    void* ptr = ::pa_malloc_aligned(size, 64);
    if (!ptr) return cudaErrorMemoryAllocation;
    *pHost = ptr;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void* pHost) noexcept {
    if (!pHost) return cudaSuccess;
    ::pa_free_aligned(pHost, 64);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind /*kind*/) noexcept {
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return cudaErrorInvalidValue;
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t /*stream*/) noexcept {
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) noexcept {
    if (count == 0) return cudaSuccess;
    if (!devPtr) return cudaErrorInvalidValue;
    std::memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t /*stream*/) noexcept {
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaDeviceSynchronize() noexcept {
    return cudaSuccess;
}

} // namespace soar::cuda
