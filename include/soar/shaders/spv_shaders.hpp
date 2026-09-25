#pragma once

#include <cstdint>
#include <span>

namespace soar::shaders {

inline constexpr uint32_t SPV_BILINEAR_RESIZE[] = 
#include "bilinear_resize.spv.h"
;

inline std::span<const uint32_t> get_bilinear_resize() {
    return SPV_BILINEAR_RESIZE;
}

inline constexpr uint32_t SPV_CONV2D_1X1[] = 
#include "conv2d_1x1.spv.h"
;

inline std::span<const uint32_t> get_conv2d_1x1() {
    return SPV_CONV2D_1X1;
}

inline constexpr uint32_t SPV_CONV2D_DW_3X3[] = 
#include "conv2d_dw_3x3.spv.h"
;

inline std::span<const uint32_t> get_conv2d_dw_3x3() {
    return SPV_CONV2D_DW_3X3;
}

inline constexpr uint32_t SPV_CONV2D_DW_5X5[] = 
#include "conv2d_dw_5x5.spv.h"
;

inline std::span<const uint32_t> get_conv2d_dw_5x5() {
    return SPV_CONV2D_DW_5X5;
}

inline constexpr uint32_t SPV_CONV2D_DW_7X7[] = 
#include "conv2d_dw_7x7.spv.h"
;

inline std::span<const uint32_t> get_conv2d_dw_7x7() {
    return SPV_CONV2D_DW_7X7;
}

inline constexpr uint32_t SPV_CONVEX_FUSION[] = 
#include "convex_fusion.spv.h"
;

inline std::span<const uint32_t> get_convex_fusion() {
    return SPV_CONVEX_FUSION;
}

inline constexpr uint32_t SPV_GLOBAL_AVG_POOL[] = 
#include "global_avg_pool.spv.h"
;

inline std::span<const uint32_t> get_global_avg_pool() {
    return SPV_GLOBAL_AVG_POOL;
}

inline constexpr uint32_t SPV_GROUP_NORM_AFFINE_SILU[] = 
#include "group_norm_affine_silu.spv.h"
;

inline std::span<const uint32_t> get_group_norm_affine_silu() {
    return SPV_GROUP_NORM_AFFINE_SILU;
}

inline constexpr uint32_t SPV_GROUP_NORM_STATS[] = 
#include "group_norm_stats.spv.h"
;

inline std::span<const uint32_t> get_group_norm_stats() {
    return SPV_GROUP_NORM_STATS;
}

inline constexpr uint32_t SPV_PIXEL_SHUFFLE_2X[] = 
#include "pixel_shuffle_2x.spv.h"
;

inline std::span<const uint32_t> get_pixel_shuffle_2x() {
    return SPV_PIXEL_SHUFFLE_2X;
}

inline constexpr uint32_t SPV_SIGMOID[] = 
#include "sigmoid.spv.h"
;

inline std::span<const uint32_t> get_sigmoid() {
    return SPV_SIGMOID;
}

inline constexpr uint32_t SPV_TENSOR_ADD[] = 
#include "tensor_add.spv.h"
;

inline std::span<const uint32_t> get_tensor_add() {
    return SPV_TENSOR_ADD;
}

} // namespace soar::shaders
