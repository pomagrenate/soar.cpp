#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <iostream>

namespace soar::cuda {

// CUDA Runtime API types and enumerations
using cudaStream_t = struct CUstream_st*;
using cudaEvent_t = struct CUevent_st*;

enum cudaError_t : int {
    cudaSuccess = 0,
    cudaErrorMissingConfiguration = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorLaunchFailure = 4,
    cudaErrorPriorLaunchFailure = 5,
    cudaErrorLaunchTimeout = 6,
    cudaErrorLaunchOutOfResources = 7,
    cudaErrorInvalidDeviceFunction = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidDevice = 10,
    cudaErrorInvalidValue = 11,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorNotReady = 600,
    cudaErrorUnknown = 999
};

enum cudaMemcpyKind : int {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

constexpr unsigned int cudaStreamDefault = 0x00;
constexpr unsigned int cudaStreamNonBlocking = 0x01;

constexpr unsigned int cudaEventDefault = 0x00;
constexpr unsigned int cudaEventDisableTiming = 0x02;
constexpr unsigned int cudaEventInterprocess = 0x04;

constexpr unsigned int cudaHostAllocDefault = 0x00;
constexpr unsigned int cudaHostAllocPortable = 0x01;
constexpr unsigned int cudaHostAllocMapped = 0x02;
constexpr unsigned int cudaHostAllocWriteCombined = 0x04;

// Exception helper
class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t err, const std::string& msg, const char* file, int line)
        : std::runtime_error(msg + " (CUDA error " + std::to_string(static_cast<int>(err)) + " at " + file + ":" + std::to_string(line) + ")"),
          error_code_(err) {}

    [[nodiscard]] cudaError_t error_code() const noexcept { return error_code_; }

private:
    cudaError_t error_code_;
};

#define SOAR_CUDA_CHECK(expr)                                                          \
    do {                                                                               \
        cudaError_t status = (expr);                                                   \
        if (status != soar::cuda::cudaSuccess) {                                       \
            throw soar::cuda::CudaException(                                           \
                status, soar::cuda::cudaGetErrorString(status), __FILE__, __LINE__);   \
        }                                                                              \
    } while (0)

#define SOAR_CUDA_WARN(expr)                                                           \
    do {                                                                               \
        cudaError_t status = (expr);                                                   \
        if (status != soar::cuda::cudaSuccess) {                                       \
            std::cerr << "CUDA Warning: " << soar::cuda::cudaGetErrorString(status)    \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;           \
        }                                                                              \
    } while (0)

// Core CUDA Runtime Function Declarations
const char* cudaGetErrorString(cudaError_t error) noexcept;
cudaError_t cudaGetLastError() noexcept;

cudaError_t cudaGetDeviceCount(int* count) noexcept;
cudaError_t cudaSetDevice(int device) noexcept;
cudaError_t cudaGetDevice(int* device) noexcept;

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) noexcept;
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority) noexcept;
cudaError_t cudaStreamDestroy(cudaStream_t stream) noexcept;
cudaError_t cudaStreamSynchronize(cudaStream_t stream) noexcept;
cudaError_t cudaStreamQuery(cudaStream_t stream) noexcept;
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0) noexcept;

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* phEvent, unsigned int flags) noexcept;
cudaError_t cudaEventDestroy(cudaEvent_t hEvent) noexcept;
cudaError_t cudaEventRecord(cudaEvent_t hEvent, cudaStream_t stream = nullptr) noexcept;
cudaError_t cudaEventSynchronize(cudaEvent_t hEvent) noexcept;
cudaError_t cudaEventQuery(cudaEvent_t hEvent) noexcept;
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) noexcept;

cudaError_t cudaMalloc(void** devPtr, size_t size) noexcept;
cudaError_t cudaFree(void* devPtr) noexcept;
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags) noexcept;
cudaError_t cudaFreeHost(void* pHost) noexcept;

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) noexcept;
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) noexcept;
cudaError_t cudaMemset(void* devPtr, int value, size_t count) noexcept;
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) noexcept;

cudaError_t cudaDeviceSynchronize() noexcept;

} // namespace soar::cuda
