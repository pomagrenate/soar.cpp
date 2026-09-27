#pragma once

#include <soar/core/types.hpp>
#include <cstddef>
#include <cstring>

namespace soar::core {

/**
 * @brief Simple SIMD utilities for CPU operations.
 * 
 * Inspired by PyTorch's CPU vectorization, this provides basic
 * SIMD operations for 2048×2048 high-resolution workloads.
 * 
 * Current implementation:
 * - Uses compiler auto-vectorization hints
 * - Placeholder for explicit AVX2/AVX-512 paths
 * - Focus on contiguous float32 operations
 * 
 * Future expansion would add runtime CPU feature detection
 * and explicit SIMD intrinsics based on measured performance.
 */

/**
 * @brief Simple vector addition with SIMD hints.
 */
inline void vec_add(float* dst, const float* src1, const float* src2, size_t n) {
    // Compiler will auto-vectorize this loop
    // Future: explicit AVX2/AVX-512 paths with runtime detection
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src1[i] + src2[i];
    }
}

/**
 * @brief Simple vector multiplication with SIMD hints.
 */
inline void vec_mul(float* dst, const float* src1, const float* src2, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src1[i] * src2[i];
    }
}

/**
 * @brief Simple vector subtraction with SIMD hints.
 */
inline void vec_sub(float* dst, const float* src1, const float* src2, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src1[i] - src2[i];
    }
}

/**
 * @brief Simple scalar multiplication with SIMD hints.
 */
inline void vec_scale(float* dst, const float* src, float scale, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i] * scale;
    }
}

/**
 * @brief Simple fill with SIMD hints.
 */
inline void vec_fill(float* dst, float value, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = value;
    }
}

/**
 * @brief Simple copy with SIMD hints.
 */
inline void vec_copy(float* dst, const float* src, size_t n) {
    // Use memcpy for large contiguous blocks (usually optimized)
    if (n > 64) {
        std::memcpy(dst, src, n * sizeof(float));
    } else {
        #pragma omp simd
        for (size_t i = 0; i < n; ++i) {
            dst[i] = src[i];
        }
    }
}

/**
 * @brief ReLU activation with SIMD hints.
 */
inline void vec_relu(float* dst, const float* src, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = src[i] > 0.0f ? src[i] : 0.0f;
    }
}

/**
 * @brief Sigmoid activation with SIMD hints.
 */
inline void vec_sigmoid(float* dst, const float* src, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
    }
}

/**
 * @brief Tanh activation with SIMD hints.
 */
inline void vec_tanh(float* dst, const float* src, size_t n) {
    #pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        dst[i] = std::tanh(src[i]);
    }
}

/**
 * @brief Compute sum of vector (reduction) with SIMD hints.
 */
inline float vec_sum(const float* src, size_t n) {
    float sum = 0.0f;
    #pragma omp simd reduction(+:sum)
    for (size_t i = 0; i < n; ++i) {
        sum += src[i];
    }
    return sum;
}

/**
 * @brief Compute mean of vector with SIMD hints.
 */
inline float vec_mean(const float* src, size_t n) {
    if (n == 0) return 0.0f;
    return vec_sum(src, n) / static_cast<float>(n);
}

} // namespace soar::core