#include <soar/cuda/cuda_runtime.hpp>
#include <chrono>
#include <cstring>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <iostream>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static void* load_shared_lib(const char* name) { return reinterpret_cast<void*>(LoadLibraryA(name)); }
static void* get_symbol(void* h, const char* s) { return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(h), s)); }
#else
#include <dlfcn.h>
static void* load_shared_lib(const char* name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
static void* get_symbol(void* h, const char* s) { return dlsym(h, s); }
#endif

namespace soar::cuda {

namespace {

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUstream_drv;
typedef void* CUevent_drv;
typedef unsigned long long CUdeviceptr;

constexpr int CUDA_SUCCESS = 0;
constexpr int CU_CTX_SCHED_AUTO = 0x00;

using PFN_cuInit = CUresult (*)(unsigned int Flags);
using PFN_cuDeviceGetCount = CUresult (*)(int *count);
using PFN_cuDeviceGet = CUresult (*)(CUdevice *device, int ordinal);
using PFN_cuDeviceGetName = CUresult (*)(char *name, int len, CUdevice dev);
using PFN_cuDeviceTotalMem = CUresult (*)(size_t *bytes, CUdevice dev);
using PFN_cuDeviceGetAttribute = CUresult (*)(int *pi, int attrib, CUdevice dev);
using PFN_cuCtxCreate = CUresult (*)(CUcontext *pctx, unsigned int flags, CUdevice dev);
using PFN_cuCtxDestroy = CUresult (*)(CUcontext ctx);
using PFN_cuCtxSetCurrent = CUresult (*)(CUcontext ctx);
using PFN_cuCtxSynchronize = CUresult (*)(void);
using PFN_cuMemAlloc = CUresult (*)(CUdeviceptr *dptr, size_t bytesize);
using PFN_cuMemFree = CUresult (*)(CUdeviceptr dptr);
using PFN_cuMemAllocHost = CUresult (*)(void **pp, size_t bytesize);
using PFN_cuMemFreeHost = CUresult (*)(void *p);
using PFN_cuMemcpyHtoD = CUresult (*)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount);
using PFN_cuMemcpyDtoH = CUresult (*)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount);
using PFN_cuMemcpyDtoD = CUresult (*)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount);
using PFN_cuMemcpyHtoDAsync = CUresult (*)(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream_drv hStream);
using PFN_cuMemcpyDtoHAsync = CUresult (*)(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream_drv hStream);
using PFN_cuMemsetD8 = CUresult (*)(CUdeviceptr dstDevice, unsigned char uc, size_t N);
using PFN_cuMemsetD8Async = CUresult (*)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream_drv hStream);
using PFN_cuMemGetInfo = CUresult (*)(size_t *free, size_t *total);
using PFN_cuStreamCreate = CUresult (*)(CUstream_drv *phStream, unsigned int Flags);
using PFN_cuStreamDestroy = CUresult (*)(CUstream_drv hStream);
using PFN_cuStreamSynchronize = CUresult (*)(CUstream_drv hStream);
using PFN_cuStreamQuery = CUresult (*)(CUstream_drv hStream);
using PFN_cuStreamWaitEvent = CUresult (*)(CUstream_drv hStream, CUevent_drv hEvent, unsigned int Flags);
using PFN_cuEventCreate = CUresult (*)(CUevent_drv *phEvent, unsigned int Flags);
using PFN_cuEventDestroy = CUresult (*)(CUevent_drv hEvent);
using PFN_cuEventRecord = CUresult (*)(CUevent_drv hEvent, CUstream_drv hStream);
using PFN_cuEventSynchronize = CUresult (*)(CUevent_drv hEvent);
using PFN_cuEventQuery = CUresult (*)(CUevent_drv hEvent);
using PFN_cuEventElapsedTime = CUresult (*)(float *pMilliseconds, CUevent_drv hStart, CUevent_drv hEnd);

PFN_cuInit pfn_cuInit = nullptr;
PFN_cuDeviceGetCount pfn_cuDeviceGetCount = nullptr;
PFN_cuDeviceGet pfn_cuDeviceGet = nullptr;
PFN_cuDeviceGetName pfn_cuDeviceGetName = nullptr;
PFN_cuDeviceTotalMem pfn_cuDeviceTotalMem = nullptr;
PFN_cuDeviceGetAttribute pfn_cuDeviceGetAttribute = nullptr;
PFN_cuCtxCreate pfn_cuCtxCreate = nullptr;
PFN_cuCtxDestroy pfn_cuCtxDestroy = nullptr;
PFN_cuCtxSetCurrent pfn_cuCtxSetCurrent = nullptr;
PFN_cuCtxSynchronize pfn_cuCtxSynchronize = nullptr;
PFN_cuMemAlloc pfn_cuMemAlloc = nullptr;
PFN_cuMemFree pfn_cuMemFree = nullptr;
PFN_cuMemAllocHost pfn_cuMemAllocHost = nullptr;
PFN_cuMemFreeHost pfn_cuMemFreeHost = nullptr;
PFN_cuMemcpyHtoD pfn_cuMemcpyHtoD = nullptr;
PFN_cuMemcpyDtoH pfn_cuMemcpyDtoH = nullptr;
PFN_cuMemcpyDtoD pfn_cuMemcpyDtoD = nullptr;
PFN_cuMemcpyHtoDAsync pfn_cuMemcpyHtoDAsync = nullptr;
PFN_cuMemcpyDtoHAsync pfn_cuMemcpyDtoHAsync = nullptr;
PFN_cuMemsetD8 pfn_cuMemsetD8 = nullptr;
PFN_cuMemsetD8Async pfn_cuMemsetD8Async = nullptr;
PFN_cuMemGetInfo pfn_cuMemGetInfo = nullptr;
PFN_cuStreamCreate pfn_cuStreamCreate = nullptr;
PFN_cuStreamDestroy pfn_cuStreamDestroy = nullptr;
PFN_cuStreamSynchronize pfn_cuStreamSynchronize = nullptr;
PFN_cuStreamQuery pfn_cuStreamQuery = nullptr;
PFN_cuStreamWaitEvent pfn_cuStreamWaitEvent = nullptr;
PFN_cuEventCreate pfn_cuEventCreate = nullptr;
PFN_cuEventDestroy pfn_cuEventDestroy = nullptr;
PFN_cuEventRecord pfn_cuEventRecord = nullptr;
PFN_cuEventSynchronize pfn_cuEventSynchronize = nullptr;
PFN_cuEventQuery pfn_cuEventQuery = nullptr;
PFN_cuEventElapsedTime pfn_cuEventElapsedTime = nullptr;

void* g_cuda_driver_lib = nullptr;
bool g_cuda_initialized = false;
bool g_has_real_cuda = false;
int g_cuda_device_count = 0;
CUdevice g_cuDevice = 0;
CUcontext g_cuContext = nullptr;
std::mutex g_cuda_init_mutex;
int g_current_device = 0;

void ensure_cuda_driver_initialized() {
    std::lock_guard<std::mutex> lock(g_cuda_init_mutex);
    if (g_cuda_initialized) return;
    g_cuda_initialized = true;

    std::vector<const char*> candidate_names = {
#if defined(_WIN32)
        "nvcuda.dll",
        "C:\\Windows\\System32\\nvcuda.dll"
#else
        "libcuda.so.1",
        "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib64/libcuda.so",
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so"
#endif
    };

    for (const char* name : candidate_names) {
        g_cuda_driver_lib = load_shared_lib(name);
        if (g_cuda_driver_lib) break;
    }

    if (!g_cuda_driver_lib) {
        g_cuda_device_count = 0;
        g_has_real_cuda = false;
        return;
    }

    pfn_cuInit = reinterpret_cast<PFN_cuInit>(get_symbol(g_cuda_driver_lib, "cuInit"));
    pfn_cuDeviceGetCount = reinterpret_cast<PFN_cuDeviceGetCount>(get_symbol(g_cuda_driver_lib, "cuDeviceGetCount"));
    pfn_cuDeviceGet = reinterpret_cast<PFN_cuDeviceGet>(get_symbol(g_cuda_driver_lib, "cuDeviceGet"));
    pfn_cuDeviceGetName = reinterpret_cast<PFN_cuDeviceGetName>(get_symbol(g_cuda_driver_lib, "cuDeviceGetName"));
    pfn_cuDeviceTotalMem = reinterpret_cast<PFN_cuDeviceTotalMem>(get_symbol(g_cuda_driver_lib, "cuDeviceTotalMem_v2"));
    if (!pfn_cuDeviceTotalMem) pfn_cuDeviceTotalMem = reinterpret_cast<PFN_cuDeviceTotalMem>(get_symbol(g_cuda_driver_lib, "cuDeviceTotalMem"));
    pfn_cuDeviceGetAttribute = reinterpret_cast<PFN_cuDeviceGetAttribute>(get_symbol(g_cuda_driver_lib, "cuDeviceGetAttribute"));
    pfn_cuCtxCreate = reinterpret_cast<PFN_cuCtxCreate>(get_symbol(g_cuda_driver_lib, "cuCtxCreate_v2"));
    if (!pfn_cuCtxCreate) pfn_cuCtxCreate = reinterpret_cast<PFN_cuCtxCreate>(get_symbol(g_cuda_driver_lib, "cuCtxCreate"));
    pfn_cuCtxDestroy = reinterpret_cast<PFN_cuCtxDestroy>(get_symbol(g_cuda_driver_lib, "cuCtxDestroy_v2"));
    if (!pfn_cuCtxDestroy) pfn_cuCtxDestroy = reinterpret_cast<PFN_cuCtxDestroy>(get_symbol(g_cuda_driver_lib, "cuCtxDestroy"));
    pfn_cuCtxSetCurrent = reinterpret_cast<PFN_cuCtxSetCurrent>(get_symbol(g_cuda_driver_lib, "cuCtxSetCurrent"));
    pfn_cuCtxSynchronize = reinterpret_cast<PFN_cuCtxSynchronize>(get_symbol(g_cuda_driver_lib, "cuCtxSynchronize"));
    pfn_cuMemAlloc = reinterpret_cast<PFN_cuMemAlloc>(get_symbol(g_cuda_driver_lib, "cuMemAlloc_v2"));
    if (!pfn_cuMemAlloc) pfn_cuMemAlloc = reinterpret_cast<PFN_cuMemAlloc>(get_symbol(g_cuda_driver_lib, "cuMemAlloc"));
    pfn_cuMemFree = reinterpret_cast<PFN_cuMemFree>(get_symbol(g_cuda_driver_lib, "cuMemFree_v2"));
    if (!pfn_cuMemFree) pfn_cuMemFree = reinterpret_cast<PFN_cuMemFree>(get_symbol(g_cuda_driver_lib, "cuMemFree"));
    pfn_cuMemAllocHost = reinterpret_cast<PFN_cuMemAllocHost>(get_symbol(g_cuda_driver_lib, "cuMemAllocHost_v2"));
    if (!pfn_cuMemAllocHost) pfn_cuMemAllocHost = reinterpret_cast<PFN_cuMemAllocHost>(get_symbol(g_cuda_driver_lib, "cuMemAllocHost"));
    pfn_cuMemFreeHost = reinterpret_cast<PFN_cuMemFreeHost>(get_symbol(g_cuda_driver_lib, "cuMemFreeHost"));
    pfn_cuMemcpyHtoD = reinterpret_cast<PFN_cuMemcpyHtoD>(get_symbol(g_cuda_driver_lib, "cuMemcpyHtoD_v2"));
    if (!pfn_cuMemcpyHtoD) pfn_cuMemcpyHtoD = reinterpret_cast<PFN_cuMemcpyHtoD>(get_symbol(g_cuda_driver_lib, "cuMemcpyHtoD"));
    pfn_cuMemcpyDtoH = reinterpret_cast<PFN_cuMemcpyDtoH>(get_symbol(g_cuda_driver_lib, "cuMemcpyDtoH_v2"));
    if (!pfn_cuMemcpyDtoH) pfn_cuMemcpyDtoH = reinterpret_cast<PFN_cuMemcpyDtoH>(get_symbol(g_cuda_driver_lib, "cuMemcpyDtoH"));
    pfn_cuMemcpyDtoD = reinterpret_cast<PFN_cuMemcpyDtoD>(get_symbol(g_cuda_driver_lib, "cuMemcpyDtoD_v2"));
    if (!pfn_cuMemcpyDtoD) pfn_cuMemcpyDtoD = reinterpret_cast<PFN_cuMemcpyDtoD>(get_symbol(g_cuda_driver_lib, "cuMemcpyDtoD"));
    pfn_cuMemcpyHtoDAsync = reinterpret_cast<PFN_cuMemcpyHtoDAsync>(get_symbol(g_cuda_driver_lib, "cuMemcpyHtoDAsync_v2"));
    if (!pfn_cuMemcpyHtoDAsync) pfn_cuMemcpyHtoDAsync = reinterpret_cast<PFN_cuMemcpyHtoDAsync>(get_symbol(g_cuda_driver_lib, "cuMemcpyHtoDAsync"));
    pfn_cuMemcpyDtoHAsync = reinterpret_cast<PFN_cuMemcpyDtoHAsync>(get_symbol(g_cuda_driver_lib, "cuMemcpyDtoHAsync_v2"));
    if (!pfn_cuMemcpyDtoHAsync) pfn_cuMemcpyDtoHAsync = reinterpret_cast<PFN_cuMemcpyDtoHAsync>(get_symbol(g_cuda_driver_lib, "cuMemcpyDtoHAsync"));
    pfn_cuMemsetD8 = reinterpret_cast<PFN_cuMemsetD8>(get_symbol(g_cuda_driver_lib, "cuMemsetD8_v2"));
    if (!pfn_cuMemsetD8) pfn_cuMemsetD8 = reinterpret_cast<PFN_cuMemsetD8>(get_symbol(g_cuda_driver_lib, "cuMemsetD8"));
    pfn_cuMemsetD8Async = reinterpret_cast<PFN_cuMemsetD8Async>(get_symbol(g_cuda_driver_lib, "cuMemsetD8Async"));
    pfn_cuMemGetInfo = reinterpret_cast<PFN_cuMemGetInfo>(get_symbol(g_cuda_driver_lib, "cuMemGetInfo_v2"));
    if (!pfn_cuMemGetInfo) pfn_cuMemGetInfo = reinterpret_cast<PFN_cuMemGetInfo>(get_symbol(g_cuda_driver_lib, "cuMemGetInfo"));
    pfn_cuStreamCreate = reinterpret_cast<PFN_cuStreamCreate>(get_symbol(g_cuda_driver_lib, "cuStreamCreate"));
    pfn_cuStreamDestroy = reinterpret_cast<PFN_cuStreamDestroy>(get_symbol(g_cuda_driver_lib, "cuStreamDestroy_v2"));
    if (!pfn_cuStreamDestroy) pfn_cuStreamDestroy = reinterpret_cast<PFN_cuStreamDestroy>(get_symbol(g_cuda_driver_lib, "cuStreamDestroy"));
    pfn_cuStreamSynchronize = reinterpret_cast<PFN_cuStreamSynchronize>(get_symbol(g_cuda_driver_lib, "cuStreamSynchronize"));
    pfn_cuStreamQuery = reinterpret_cast<PFN_cuStreamQuery>(get_symbol(g_cuda_driver_lib, "cuStreamQuery"));
    pfn_cuStreamWaitEvent = reinterpret_cast<PFN_cuStreamWaitEvent>(get_symbol(g_cuda_driver_lib, "cuStreamWaitEvent"));
    pfn_cuEventCreate = reinterpret_cast<PFN_cuEventCreate>(get_symbol(g_cuda_driver_lib, "cuEventCreate"));
    pfn_cuEventDestroy = reinterpret_cast<PFN_cuEventDestroy>(get_symbol(g_cuda_driver_lib, "cuEventDestroy_v2"));
    if (!pfn_cuEventDestroy) pfn_cuEventDestroy = reinterpret_cast<PFN_cuEventDestroy>(get_symbol(g_cuda_driver_lib, "cuEventDestroy"));
    pfn_cuEventRecord = reinterpret_cast<PFN_cuEventRecord>(get_symbol(g_cuda_driver_lib, "cuEventRecord"));
    pfn_cuEventSynchronize = reinterpret_cast<PFN_cuEventSynchronize>(get_symbol(g_cuda_driver_lib, "cuEventSynchronize"));
    pfn_cuEventQuery = reinterpret_cast<PFN_cuEventQuery>(get_symbol(g_cuda_driver_lib, "cuEventQuery"));
    pfn_cuEventElapsedTime = reinterpret_cast<PFN_cuEventElapsedTime>(get_symbol(g_cuda_driver_lib, "cuEventElapsedTime"));

    if (!pfn_cuInit || !pfn_cuDeviceGetCount || !pfn_cuDeviceGet || !pfn_cuMemAlloc) {
        g_cuda_device_count = 0;
        g_has_real_cuda = false;
        return;
    }

    CUresult res = pfn_cuInit(0);
    if (res != CUDA_SUCCESS) {
        g_cuda_device_count = 0;
        g_has_real_cuda = false;
        return;
    }

    res = pfn_cuDeviceGetCount(&g_cuda_device_count);
    if (res != CUDA_SUCCESS || g_cuda_device_count <= 0) {
        g_cuda_device_count = 0;
        g_has_real_cuda = false;
        return;
    }

    res = pfn_cuDeviceGet(&g_cuDevice, 0);
    if (res == CUDA_SUCCESS && pfn_cuCtxCreate) {
        res = pfn_cuCtxCreate(&g_cuContext, CU_CTX_SCHED_AUTO, g_cuDevice);
        if (res == CUDA_SUCCESS && g_cuContext) {
            g_has_real_cuda = true;
        }
    }
}

} // anonymous namespace

const char* cudaGetErrorString(cudaError_t error) noexcept {
    switch (error) {
        case cudaSuccess: return "cudaSuccess: no errors";
        case cudaErrorMissingConfiguration: return "cudaErrorMissingConfiguration";
        case cudaErrorMemoryAllocation: return "cudaErrorMemoryAllocation: out of memory on device";
        case cudaErrorInitializationError: return "cudaErrorInitializationError: failed to initialize CUDA driver";
        case cudaErrorLaunchFailure: return "cudaErrorLaunchFailure";
        case cudaErrorPriorLaunchFailure: return "cudaErrorPriorLaunchFailure";
        case cudaErrorLaunchTimeout: return "cudaErrorLaunchTimeout";
        case cudaErrorLaunchOutOfResources: return "cudaErrorLaunchOutOfResources";
        case cudaErrorInvalidDeviceFunction: return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration: return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice: return "cudaErrorInvalidDevice: no physical CUDA GPU device found";
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
    ensure_cuda_driver_initialized();
    *count = g_cuda_device_count;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) noexcept {
    ensure_cuda_driver_initialized();
    if (!g_has_real_cuda || device < 0 || device >= g_cuda_device_count) {
        return cudaErrorInvalidDevice;
    }
    g_current_device = device;
    if (pfn_cuDeviceGet && pfn_cuCtxSetCurrent) {
        CUdevice dev = 0;
        pfn_cuDeviceGet(&dev, device);
        // Bind context if needed
    }
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int* device) noexcept {
    if (!device) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (!g_has_real_cuda) return cudaErrorInvalidDevice;
    *device = g_current_device;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) noexcept {
    if (!pStream) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuStreamCreate) {
        CUstream_drv s = nullptr;
        CUresult res = pfn_cuStreamCreate(&s, flags);
        if (res != CUDA_SUCCESS) return cudaErrorMemoryAllocation;
        *pStream = reinterpret_cast<cudaStream_t>(s);
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int /*priority*/) noexcept {
    return cudaStreamCreateWithFlags(pStream, flags);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) noexcept {
    if (!stream) return cudaSuccess;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuStreamDestroy) {
        pfn_cuStreamDestroy(reinterpret_cast<CUstream_drv>(stream));
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) noexcept {
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuStreamSynchronize) {
        pfn_cuStreamSynchronize(reinterpret_cast<CUstream_drv>(stream));
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaStreamQuery(cudaStream_t stream) noexcept {
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuStreamQuery) {
        CUresult res = pfn_cuStreamQuery(reinterpret_cast<CUstream_drv>(stream));
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorNotReady;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) noexcept {
    if (!event) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuStreamWaitEvent) {
        CUresult res = pfn_cuStreamWaitEvent(reinterpret_cast<CUstream_drv>(stream), reinterpret_cast<CUevent_drv>(event), flags);
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* phEvent, unsigned int flags) noexcept {
    if (!phEvent) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuEventCreate) {
        CUevent_drv ev = nullptr;
        CUresult res = pfn_cuEventCreate(&ev, flags);
        if (res != CUDA_SUCCESS) return cudaErrorMemoryAllocation;
        *phEvent = reinterpret_cast<cudaEvent_t>(ev);
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaEventDestroy(cudaEvent_t hEvent) noexcept {
    if (!hEvent) return cudaSuccess;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuEventDestroy) {
        pfn_cuEventDestroy(reinterpret_cast<CUevent_drv>(hEvent));
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaEventRecord(cudaEvent_t hEvent, cudaStream_t stream) noexcept {
    if (!hEvent) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuEventRecord) {
        CUresult res = pfn_cuEventRecord(reinterpret_cast<CUevent_drv>(hEvent), reinterpret_cast<CUstream_drv>(stream));
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaEventSynchronize(cudaEvent_t hEvent) noexcept {
    if (!hEvent) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuEventSynchronize) {
        CUresult res = pfn_cuEventSynchronize(reinterpret_cast<CUevent_drv>(hEvent));
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaEventQuery(cudaEvent_t hEvent) noexcept {
    if (!hEvent) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuEventQuery) {
        CUresult res = pfn_cuEventQuery(reinterpret_cast<CUevent_drv>(hEvent));
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorNotReady;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) noexcept {
    if (!ms || !start || !end) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuEventElapsedTime) {
        CUresult res = pfn_cuEventElapsedTime(ms, reinterpret_cast<CUevent_drv>(start), reinterpret_cast<CUevent_drv>(end));
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaMalloc(void** devPtr, size_t size) noexcept {
    if (!devPtr) return cudaErrorInvalidValue;
    if (size == 0) {
        *devPtr = nullptr;
        return cudaSuccess;
    }
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuMemAlloc) {
        CUdeviceptr dptr = 0;
        CUresult res = pfn_cuMemAlloc(&dptr, size);
        if (res != CUDA_SUCCESS) return cudaErrorMemoryAllocation;
        *devPtr = reinterpret_cast<void*>(static_cast<uintptr_t>(dptr));
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaFree(void* devPtr) noexcept {
    if (!devPtr) return cudaSuccess;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuMemFree) {
        pfn_cuMemFree(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(devPtr)));
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int /*flags*/) noexcept {
    if (!pHost) return cudaErrorInvalidValue;
    if (size == 0) {
        *pHost = nullptr;
        return cudaSuccess;
    }
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuMemAllocHost) {
        CUresult res = pfn_cuMemAllocHost(pHost, size);
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorMemoryAllocation;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaFreeHost(void* pHost) noexcept {
    if (!pHost) return cudaSuccess;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuMemFreeHost) {
        pfn_cuMemFreeHost(pHost);
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) noexcept {
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (kind == cudaMemcpyHostToHost) {
        std::memcpy(dst, src, count);
        return cudaSuccess;
    }
    if (g_has_real_cuda) {
        CUresult res = CUDA_SUCCESS;
        if (kind == cudaMemcpyHostToDevice && pfn_cuMemcpyHtoD) {
            res = pfn_cuMemcpyHtoD(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dst)), src, count);
        } else if (kind == cudaMemcpyDeviceToHost && pfn_cuMemcpyDtoH) {
            res = pfn_cuMemcpyDtoH(dst, static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(src)), count);
        } else if (kind == cudaMemcpyDeviceToDevice && pfn_cuMemcpyDtoD) {
            res = pfn_cuMemcpyDtoD(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dst)),
                                   static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(src)), count);
        }
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) noexcept {
    if (count == 0) return cudaSuccess;
    if (!dst || !src) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (kind == cudaMemcpyHostToHost) {
        std::memcpy(dst, src, count);
        return cudaSuccess;
    }
    if (g_has_real_cuda) {
        CUresult res = CUDA_SUCCESS;
        if (kind == cudaMemcpyHostToDevice && pfn_cuMemcpyHtoDAsync) {
            res = pfn_cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(dst)), src, count, reinterpret_cast<CUstream_drv>(stream));
        } else if (kind == cudaMemcpyDeviceToHost && pfn_cuMemcpyDtoHAsync) {
            res = pfn_cuMemcpyDtoHAsync(dst, static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(src)), count, reinterpret_cast<CUstream_drv>(stream));
        } else {
            return cudaMemcpy(dst, src, count, kind);
        }
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaMemset(void* devPtr, int value, size_t count) noexcept {
    if (count == 0) return cudaSuccess;
    if (!devPtr) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuMemsetD8) {
        CUresult res = pfn_cuMemsetD8(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(devPtr)), static_cast<unsigned char>(value), count);
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) noexcept {
    if (count == 0) return cudaSuccess;
    if (!devPtr) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuMemsetD8Async) {
        CUresult res = pfn_cuMemsetD8Async(static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(devPtr)), static_cast<unsigned char>(value), count, reinterpret_cast<CUstream_drv>(stream));
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) noexcept {
    if (!free_bytes || !total_bytes) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuMemGetInfo) {
        CUresult res = pfn_cuMemGetInfo(free_bytes, total_bytes);
        return (res == CUDA_SUCCESS) ? cudaSuccess : cudaErrorLaunchFailure;
    }
    return cudaErrorInvalidDevice;
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) noexcept {
    if (!prop) return cudaErrorInvalidValue;
    ensure_cuda_driver_initialized();
    if (!g_has_real_cuda || device < 0 || device >= g_cuda_device_count) {
        return cudaErrorInvalidDevice;
    }
    CUdevice dev = 0;
    if (pfn_cuDeviceGet) pfn_cuDeviceGet(&dev, device);
    if (pfn_cuDeviceGetName) pfn_cuDeviceGetName(prop->name, 255, dev);
    if (pfn_cuDeviceTotalMem) pfn_cuDeviceTotalMem(&prop->totalGlobalMem, dev);
    if (pfn_cuDeviceGetAttribute) {
        pfn_cuDeviceGetAttribute(&prop->major, 75 /*CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR*/, dev);
        pfn_cuDeviceGetAttribute(&prop->minor, 76 /*CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR*/, dev);
        pfn_cuDeviceGetAttribute(&prop->multiProcessorCount, 16 /*CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT*/, dev);
    }
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize() noexcept {
    ensure_cuda_driver_initialized();
    if (g_has_real_cuda && pfn_cuCtxSynchronize) {
        pfn_cuCtxSynchronize();
        return cudaSuccess;
    }
    return cudaErrorInvalidDevice;
}

} // namespace soar::cuda
