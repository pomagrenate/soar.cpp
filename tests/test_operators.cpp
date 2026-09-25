#include <soar/vulkan/context.hpp>
#include <soar/vulkan/buffer.hpp>
#include <soar/vulkan/pipeline.hpp>
#include <soar/vulkan/command_queue.hpp>
#include <soar/shaders/spv_shaders.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <random>

namespace {

bool float_near(float a, float b, float tol = 1e-4f) {
    float diff = std::fabs(a - b);
    return diff <= tol;
}

} // anonymous namespace

void test_conv2d_1x1_operator() {
    std::cout << "[TEST] Running test_conv2d_1x1_operator..." << std::endl;
    soar::vk::VulkanContext ctx(false);
    soar::vk::CommandQueue queue(ctx);

    constexpr uint32_t H = 32;
    constexpr uint32_t W = 32;
    constexpr uint32_t C_in = 16;
    constexpr uint32_t C_out = 32;

    struct PushConstants {
        uint32_t H;
        uint32_t W;
        uint32_t C_in;
        uint32_t C_out;
        uint32_t has_bias;
    } pc{H, W, C_in, C_out, 1};

    size_t in_size = C_in * H * W * sizeof(float);
    size_t w_size = C_out * C_in * sizeof(float);
    size_t b_size = C_out * sizeof(float);
    size_t out_size = C_out * H * W * sizeof(float);

    std::vector<float> host_in(C_in * H * W);
    std::vector<float> host_w(C_out * C_in);
    std::vector<float> host_b(C_out);
    std::vector<float> cpu_ref(C_out * H * W, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : host_in) v = dist(rng);
    for (auto& v : host_w) v = dist(rng);
    for (auto& v : host_b) v = dist(rng);

    // CPU Reference computation
    for (uint32_t co = 0; co < C_out; ++co) {
        float b = host_b[co];
        for (uint32_t y = 0; y < H; ++y) {
            for (uint32_t x = 0; x < W; ++x) {
                float sum = b;
                for (uint32_t ci = 0; ci < C_in; ++ci) {
                    sum += host_in[ci * (H * W) + y * W + x] * host_w[co * C_in + ci];
                }
                cpu_ref[co * (H * W) + y * W + x] = sum;
            }
        }
    }

    // Allocate GPU buffers
    soar::vk::VulkanBuffer buf_in(ctx, in_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_w(ctx, w_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_b(ctx, b_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_out(ctx, out_size, soar::vk::BufferUsageType::StagingHost);

    buf_in.upload_host(host_in.data(), in_size);
    buf_w.upload_host(host_w.data(), w_size);
    buf_b.upload_host(host_b.data(), b_size);

    soar::vk::ComputePipeline pipeline(ctx, soar::shaders::get_conv2d_1x1(), 4, sizeof(PushConstants));
    const soar::vk::VulkanBuffer* bufs[] = { &buf_in, &buf_w, &buf_b, &buf_out };
    pipeline.bind_buffers(bufs);

    uint32_t gx = (W + 15) / 16;
    uint32_t gy = (H + 15) / 16;
    uint32_t gz = C_out;

    queue.execute_sync([&](VkCommandBuffer cmd) {
        pipeline.record_dispatch(cmd, gx, gy, gz, &pc, sizeof(pc));
    });

    std::vector<float> gpu_out(C_out * H * W);
    buf_out.download_host(gpu_out.data(), out_size);

    float max_diff = 0.0f;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        float diff = std::fabs(gpu_out[i] - cpu_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "  Conv2d 1x1 Max absolute difference: " << max_diff << std::endl;
    assert(max_diff < 1e-4f);
    std::cout << "  -> test_conv2d_1x1_operator PASSED" << std::endl;
}

void test_conv2d_dw_3x3_operator() {
    std::cout << "[TEST] Running test_conv2d_dw_3x3_operator (stride=1, pad=1)..." << std::endl;
    soar::vk::VulkanContext ctx(false);
    soar::vk::CommandQueue queue(ctx);

    constexpr uint32_t H = 32;
    constexpr uint32_t W = 32;
    constexpr uint32_t C = 16;

    struct PushConstants {
        uint32_t H_in;
        uint32_t W_in;
        uint32_t H_out;
        uint32_t W_out;
        uint32_t C;
        uint32_t stride;
        uint32_t dilation;
        uint32_t has_bias;
    } pc{H, W, H, W, C, 1, 1, 1};

    size_t in_size = C * H * W * sizeof(float);
    size_t w_size = C * 9 * sizeof(float);
    size_t b_size = C * sizeof(float);
    size_t out_size = C * H * W * sizeof(float);

    std::vector<float> host_in(C * H * W);
    std::vector<float> host_w(C * 9);
    std::vector<float> host_b(C);
    std::vector<float> cpu_ref(C * H * W, 0.0f);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& v : host_in) v = dist(rng);
    for (auto& v : host_w) v = dist(rng);
    for (auto& v : host_b) v = dist(rng);

    // CPU Reference
    for (uint32_t c = 0; c < C; ++c) {
        float b = host_b[c];
        for (int y = 0; y < static_cast<int>(H); ++y) {
            for (int x = 0; x < static_cast<int>(W); ++x) {
                float sum = b;
                int w_idx = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    int py = y + ky;
                    for (int kx = -1; kx <= 1; ++kx) {
                        int px = x + kx;
                        if (py >= 0 && py < static_cast<int>(H) && px >= 0 && px < static_cast<int>(W)) {
                            sum += host_in[c * (H * W) + py * W + px] * host_w[c * 9 + w_idx];
                        }
                        w_idx++;
                    }
                }
                cpu_ref[c * (H * W) + y * W + x] = sum;
            }
        }
    }

    soar::vk::VulkanBuffer buf_in(ctx, in_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_w(ctx, w_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_b(ctx, b_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_out(ctx, out_size, soar::vk::BufferUsageType::StagingHost);

    buf_in.upload_host(host_in.data(), in_size);
    buf_w.upload_host(host_w.data(), w_size);
    buf_b.upload_host(host_b.data(), b_size);

    soar::vk::ComputePipeline pipeline(ctx, soar::shaders::get_conv2d_dw_3x3(), 4, sizeof(PushConstants));
    const soar::vk::VulkanBuffer* bufs[] = { &buf_in, &buf_w, &buf_b, &buf_out };
    pipeline.bind_buffers(bufs);

    uint32_t gx = (W + 15) / 16;
    uint32_t gy = (H + 15) / 16;
    uint32_t gz = C;

    queue.execute_sync([&](VkCommandBuffer cmd) {
        pipeline.record_dispatch(cmd, gx, gy, gz, &pc, sizeof(pc));
    });

    std::vector<float> gpu_out(C * H * W);
    buf_out.download_host(gpu_out.data(), out_size);

    float max_diff = 0.0f;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        float diff = std::fabs(gpu_out[i] - cpu_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "  Depthwise 3x3 Max absolute difference: " << max_diff << std::endl;
    assert(max_diff < 1e-4f);
    std::cout << "  -> test_conv2d_dw_3x3_operator PASSED" << std::endl;
}

void test_group_norm_silu_operator() {
    std::cout << "[TEST] Running test_group_norm_silu_operator..." << std::endl;
    soar::vk::VulkanContext ctx(false);
    soar::vk::CommandQueue queue(ctx);

    constexpr uint32_t H = 32;
    constexpr uint32_t W = 32;
    constexpr uint32_t C = 16;
    constexpr uint32_t G = 4;
    constexpr float eps = 1e-5f;

    struct StatsPC {
        uint32_t H;
        uint32_t W;
        uint32_t C;
        uint32_t num_groups;
        float eps;
    } stats_pc{H, W, C, G, eps};

    struct AffinePC {
        uint32_t H;
        uint32_t W;
        uint32_t C;
        uint32_t num_groups;
        uint32_t apply_silu;
    } affine_pc{H, W, C, G, 1};

    size_t in_size = C * H * W * sizeof(float);
    size_t stats_size = 2 * G * sizeof(float);
    size_t param_size = C * sizeof(float);

    std::vector<float> host_in(C * H * W);
    std::vector<float> host_gamma(C, 1.2f);
    std::vector<float> host_beta(C, -0.3f);
    std::vector<float> cpu_ref(C * H * W, 0.0f);

    std::mt19937 rng(999);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : host_in) v = dist(rng);

    // CPU Reference GroupNorm with biased sample variance + SiLU
    uint32_t c_per_g = C / G;
    size_t elements_per_g = static_cast<size_t>(c_per_g) * H * W;

    for (uint32_t g = 0; g < G; ++g) {
        double sum = 0.0;
        double sum_sq = 0.0;
        for (uint32_t cg = 0; cg < c_per_g; ++cg) {
            uint32_t c = g * c_per_g + cg;
            for (uint32_t sp = 0; sp < H * W; ++sp) {
                float val = host_in[c * (H * W) + sp];
                sum += val;
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }
        }
        float mean = static_cast<float>(sum / elements_per_g);
        float mean_sq = static_cast<float>(sum_sq / elements_per_g);
        float variance = std::max(mean_sq - mean * mean, 0.0f);
        float rstd = 1.0f / std::sqrt(variance + eps);

        for (uint32_t cg = 0; cg < c_per_g; ++cg) {
            uint32_t c = g * c_per_g + cg;
            float g_val = host_gamma[c];
            float b_val = host_beta[c];
            for (uint32_t sp = 0; sp < H * W; ++sp) {
                float val = host_in[c * (H * W) + sp];
                float norm_val = g_val * (val - mean) * rstd + b_val;
                float silu_val = norm_val / (1.0f + std::exp(-norm_val));
                cpu_ref[c * (H * W) + sp] = silu_val;
            }
        }
    }

    soar::vk::VulkanBuffer buf_in(ctx, in_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_stats(ctx, stats_size, soar::vk::BufferUsageType::DeviceStorage);
    soar::vk::VulkanBuffer buf_gamma(ctx, param_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_beta(ctx, param_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_out(ctx, in_size, soar::vk::BufferUsageType::StagingHost);

    buf_in.upload_host(host_in.data(), in_size);
    buf_gamma.upload_host(host_gamma.data(), param_size);
    buf_beta.upload_host(host_beta.data(), param_size);

    soar::vk::ComputePipeline stats_pipeline(ctx, soar::shaders::get_group_norm_stats(), 2, sizeof(StatsPC));
    const soar::vk::VulkanBuffer* stats_bufs[] = { &buf_in, &buf_stats };
    stats_pipeline.bind_buffers(stats_bufs);

    soar::vk::ComputePipeline affine_pipeline(ctx, soar::shaders::get_group_norm_affine_silu(), 5, sizeof(AffinePC));
    const soar::vk::VulkanBuffer* affine_bufs[] = { &buf_in, &buf_stats, &buf_gamma, &buf_beta, &buf_out };
    affine_pipeline.bind_buffers(affine_bufs);

    queue.execute_sync([&](VkCommandBuffer cmd) {
        // Pass 1: compute stats
        stats_pipeline.record_dispatch(cmd, G, 1, 1, &stats_pc, sizeof(stats_pc));

        // Barrier: stats write -> read
        soar::vk::CommandQueue::memory_barrier(
            ctx,
            cmd,
            buf_stats,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT
        );

        // Pass 2: affine + SiLU
        affine_pipeline.record_dispatch(cmd, (W + 15) / 16, (H + 15) / 16, C, &affine_pc, sizeof(affine_pc));
    });

    std::vector<float> gpu_out(C * H * W);
    buf_out.download_host(gpu_out.data(), in_size);

    float max_diff = 0.0f;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        float diff = std::fabs(gpu_out[i] - cpu_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "  GroupNorm + SiLU Max absolute difference: " << max_diff << std::endl;
    assert(max_diff < 1e-4f);
    std::cout << "  -> test_group_norm_silu_operator PASSED" << std::endl;
}

void test_pixel_shuffle_2x_operator() {
    std::cout << "[TEST] Running test_pixel_shuffle_2x_operator..." << std::endl;
    soar::vk::VulkanContext ctx(false);
    soar::vk::CommandQueue queue(ctx);

    constexpr uint32_t H_in = 16;
    constexpr uint32_t W_in = 16;
    constexpr uint32_t C_out = 4;
    constexpr uint32_t C_in = C_out * 4; // 16
    constexpr uint32_t H_out = H_in * 2; // 32
    constexpr uint32_t W_out = W_in * 2; // 32

    struct PushConstants {
        uint32_t H_out;
        uint32_t W_out;
        uint32_t C_out;
    } pc{H_out, W_out, C_out};

    size_t in_size = C_in * H_in * W_in * sizeof(float);
    size_t out_size = C_out * H_out * W_out * sizeof(float);

    std::vector<float> host_in(C_in * H_in * W_in);
    std::vector<float> cpu_ref(C_out * H_out * W_out, 0.0f);

    for (size_t i = 0; i < host_in.size(); ++i) {
        host_in[i] = static_cast<float>(i);
    }

    // CPU Reference
    for (uint32_t c = 0; c < C_out; ++c) {
        for (uint32_t y = 0; y < H_out; ++y) {
            for (uint32_t x = 0; x < W_out; ++x) {
                uint32_t in_y = y / 2;
                uint32_t in_x = x / 2;
                uint32_t sub_y = y % 2;
                uint32_t sub_x = x % 2;
                uint32_t c_in = c * 4 + sub_y * 2 + sub_x;

                cpu_ref[c * (H_out * W_out) + y * W_out + x] =
                    host_in[c_in * (H_in * W_in) + in_y * W_in + in_x];
            }
        }
    }

    soar::vk::VulkanBuffer buf_in(ctx, in_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_out(ctx, out_size, soar::vk::BufferUsageType::StagingHost);

    buf_in.upload_host(host_in.data(), in_size);

    soar::vk::ComputePipeline pipeline(ctx, soar::shaders::get_pixel_shuffle_2x(), 2, sizeof(PushConstants));
    const soar::vk::VulkanBuffer* bufs[] = { &buf_in, &buf_out };
    pipeline.bind_buffers(bufs);

    queue.execute_sync([&](VkCommandBuffer cmd) {
        pipeline.record_dispatch(cmd, (W_out + 15) / 16, (H_out + 15) / 16, C_out, &pc, sizeof(pc));
    });

    std::vector<float> gpu_out(C_out * H_out * W_out);
    buf_out.download_host(gpu_out.data(), out_size);

    size_t mismatches = 0;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        if (gpu_out[i] != cpu_ref[i]) {
            mismatches++;
        }
    }

    assert(mismatches == 0);
    std::cout << "  PixelShuffle 2x byte-for-byte exact matches: " << gpu_out.size() << std::endl;
    std::cout << "  -> test_pixel_shuffle_2x_operator PASSED" << std::endl;
}

void test_bilinear_resize_operator() {
    std::cout << "[TEST] Running test_bilinear_resize_operator (align_corners=False)..." << std::endl;
    soar::vk::VulkanContext ctx(false);
    soar::vk::CommandQueue queue(ctx);

    constexpr uint32_t H_in = 8;
    constexpr uint32_t W_in = 8;
    constexpr uint32_t H_out = 16;
    constexpr uint32_t W_out = 16;
    constexpr uint32_t C = 4;

    struct PushConstants {
        uint32_t H_in;
        uint32_t W_in;
        uint32_t H_out;
        uint32_t W_out;
        uint32_t C;
    } pc{H_in, W_in, H_out, W_out, C};

    size_t in_size = C * H_in * W_in * sizeof(float);
    size_t out_size = C * H_out * W_out * sizeof(float);

    std::vector<float> host_in(C * H_in * W_in);
    std::vector<float> cpu_ref(C * H_out * W_out, 0.0f);

    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    for (auto& v : host_in) v = dist(rng);

    // CPU Reference PyTorch align_corners=False
    float scale_y = static_cast<float>(H_in) / static_cast<float>(H_out);
    float scale_x = static_cast<float>(W_in) / static_cast<float>(W_out);

    for (uint32_t c = 0; c < C; ++c) {
        for (uint32_t y = 0; y < H_out; ++y) {
            for (uint32_t x = 0; x < W_out; ++x) {
                float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
                float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;

                src_y = std::max(src_y, 0.0f);
                src_x = std::max(src_x, 0.0f);

                int y0 = static_cast<int>(std::floor(src_y));
                int x0 = static_cast<int>(std::floor(src_x));
                int y1 = std::min(y0 + 1, static_cast<int>(H_in) - 1);
                int x1 = std::min(x0 + 1, static_cast<int>(W_in) - 1);
                y0 = std::min(y0, static_cast<int>(H_in) - 1);
                x0 = std::min(x0, static_cast<int>(W_in) - 1);

                float ly = src_y - static_cast<float>(y0);
                float lx = src_x - static_cast<float>(x0);
                float hy = 1.0f - ly;
                float hx = 1.0f - lx;

                size_t offset = c * (H_in * W_in);
                float v00 = host_in[offset + y0 * W_in + x0];
                float v01 = host_in[offset + y0 * W_in + x1];
                float v10 = host_in[offset + y1 * W_in + x0];
                float v11 = host_in[offset + y1 * W_in + x1];

                float top = hx * v00 + lx * v01;
                float bottom = hx * v10 + lx * v11;
                cpu_ref[c * (H_out * W_out) + y * W_out + x] = hy * top + ly * bottom;
            }
        }
    }

    soar::vk::VulkanBuffer buf_in(ctx, in_size, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer buf_out(ctx, out_size, soar::vk::BufferUsageType::StagingHost);

    buf_in.upload_host(host_in.data(), in_size);

    soar::vk::ComputePipeline pipeline(ctx, soar::shaders::get_bilinear_resize(), 2, sizeof(PushConstants));
    const soar::vk::VulkanBuffer* bufs[] = { &buf_in, &buf_out };
    pipeline.bind_buffers(bufs);

    queue.execute_sync([&](VkCommandBuffer cmd) {
        pipeline.record_dispatch(cmd, (W_out + 15) / 16, (H_out + 15) / 16, C, &pc, sizeof(pc));
    });

    std::vector<float> gpu_out(C * H_out * W_out);
    buf_out.download_host(gpu_out.data(), out_size);

    float max_diff = 0.0f;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
        float diff = std::fabs(gpu_out[i] - cpu_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "  Bilinear Resize Max absolute difference: " << max_diff << std::endl;
    assert(max_diff < 1e-4f);
    std::cout << "  -> test_bilinear_resize_operator PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "  SOAR Mathematical Operator Tests      " << std::endl;
        std::cout << "========================================" << std::endl;

        test_conv2d_1x1_operator();
        test_conv2d_dw_3x3_operator();
        test_group_norm_silu_operator();
        test_pixel_shuffle_2x_operator();
        test_bilinear_resize_operator();

        std::cout << std::endl;
        std::cout << ">>> ALL OPERATOR MATHEMATICAL TESTS PASSED! <<<" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Operator test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
