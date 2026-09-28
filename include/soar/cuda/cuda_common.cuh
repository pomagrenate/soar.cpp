#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <limits>

namespace soar::cuda {

// Mirroring PyTorch at::cuda::detail::CUDA_NUM_THREADS
constexpr int CUDA_NUM_THREADS = 1024;
constexpr int C10_WARP_SIZE = 32;

#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)                                 \
  int64_t _i_n_d_e_x = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;      \
  for (index_type i = _i_n_d_e_x; _i_n_d_e_x < (n);                            \
       _i_n_d_e_x += blockDim.x * gridDim.x, i = _i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int64_t)

inline int GET_BLOCKS(const int64_t N, const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
    if (N <= 0) return 1;
    int64_t block_num = (N - 1) / max_threads_per_block + 1;
    if (block_num > 65535) block_num = 65535; // Maximum grid dim x for compatibility
    return static_cast<int>(block_num);
}

// ============================================================
//  Error-handling macros
// ============================================================

#define SOAR_CUDA_CHECK(call)                                                    \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

#define SOAR_CUDA_KERNEL_LAUNCH_CHECK()                                         \
    do {                                                                        \
        cudaError_t err__ = cudaGetLastError();                                 \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA kernel launch failed at %s:%d: %s\n",    \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// Debug-only kernel launch check (no-op in release builds).
// Called after every kernel launch in .cu files to detect errors early.
#ifdef NDEBUG
#   define SOAR_CUDA_KERNEL_LAUNCH_CHECK_DEBUG() ((void)0)
#else
#   define SOAR_CUDA_KERNEL_LAUNCH_CHECK_DEBUG() SOAR_CUDA_KERNEL_LAUNCH_CHECK()
#endif




// ============================================================
//  Warp/Block reductions
// ============================================================

// Warp reduce sum across 32 lanes (mirrors PyTorch WarpReduceSum)
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduce sum across up to 1024 threads using shared memory.
// WARNING: return value is only valid for thread 0.
template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    __syncthreads(); // prevent races when BlockReduces are called in a row
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    T bsum = (lane < num_warps) ? shared[lane] : T(0);
    if (wid == 0) {
        bsum = warp_reduce_sum(bsum);
    }
    return bsum;
}

// ============================================================
//  Vectorized load/store (mirrors PyTorch MemoryAccess.cuh)
//  VecLoad<float,4> enables 128-bit loads (float4) in one inst.
// ============================================================

template <typename T, int N>
struct alignas(N * sizeof(T)) VecLoad {
    T val[N];
};

// Check that pointer is naturally aligned for N-element vector loads
template <typename T, int N>
__host__ __device__ __forceinline__ bool is_vec_aligned(const T* p) {
    return (reinterpret_cast<uintptr_t>(p) % (N * sizeof(T))) == 0;
}

// Number of elements each thread processes per vectorized iteration (ILP)
// Matches PyTorch's kILP = 4 in MultiTensorApply.cuh
constexpr int kILP = 4;

// ============================================================
//  Welford online mean/variance accumulator
//
//  Numerically stable alternative to sum / sum-of-squares.
//  Prevents catastrophic cancellation in GroupNorm moments.
//  Reference: Welford (1962); PyTorch group_norm_kernel.cu.
// ============================================================
struct WelfordData {
    float mean{0.f};
    float m2{0.f};    // sum of squared deviations from running mean
    int64_t n{0};     // count

    __device__ __forceinline__ WelfordData() = default;
    __device__ __forceinline__ WelfordData(float m, float s, int64_t cnt)
        : mean(m), m2(s), n(cnt) {}

    // Online single-element update (Welford one-pass recurrence)
    __device__ __forceinline__ void update(float x) {
        ++n;
        float delta = x - mean;
        mean += delta / static_cast<float>(n);
        m2   += delta * (x - mean);
    }

    // Combine two independent WelfordData accumulators (Chan parallel variant)
    __device__ __forceinline__ WelfordData combine(const WelfordData& other) const {
        if (other.n == 0) return *this;
        if (n == 0) return other;
        int64_t total  = n + other.n;
        float delta    = other.mean - mean;
        float fn       = static_cast<float>(n);
        float fo       = static_cast<float>(other.n);
        float ft       = static_cast<float>(total);
        float new_mean = mean + delta * fo / ft;
        float new_m2   = m2 + other.m2 + delta * delta * fn * fo / ft;
        return WelfordData(new_mean, new_m2, total);
    }

    // Biased variance (denominator n, matching PyTorch GroupNorm convention)
    __device__ __forceinline__ float variance() const {
        return (n > 0) ? (m2 / static_cast<float>(n)) : 0.f;
    }
};

// Warp-level Welford reduction (all 32 lanes combined into lane 0)
__inline__ __device__ WelfordData warp_reduce_welford(WelfordData val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        WelfordData other;
        other.mean = __shfl_down_sync(0xffffffff, val.mean, offset);
        other.m2   = __shfl_down_sync(0xffffffff, val.m2,   offset);
        other.n    = __shfl_down_sync(0xffffffff, (int)val.n, offset);
        val = val.combine(other);
    }
    return val;
}

// Block-level Welford reduction using shared memory.
// Result is valid only in thread 0.
__inline__ __device__ WelfordData block_reduce_welford(WelfordData val) {
    // alignas before __shared__ avoids nvcc non-trivial constructor warnings
    alignas(WelfordData) __shared__ char shared_bytes[sizeof(WelfordData) * 32];
    WelfordData* shared = reinterpret_cast<WelfordData*>(shared_bytes);

    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;

    val = warp_reduce_welford(val);
    __syncthreads(); // prevent races when BlockReduces are called in a row
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    WelfordData bval = (lane < num_warps) ? shared[lane] : WelfordData{};
    if (wid == 0) bval = warp_reduce_welford(bval);
    return bval; // valid only in thread 0
}

} // namespace soar::cuda
