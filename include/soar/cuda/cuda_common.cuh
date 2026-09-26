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

// Warp reduce sum across 32 lanes (mirrors PyTorch ReduceSum32)
template <typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduce sum across up to 1024 threads using shared memory
template <typename T>
__inline__ __device__ T block_reduce_sum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
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

} // namespace soar::cuda
