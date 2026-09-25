#include <soar/vulkan/context.hpp>
#include <soar/vulkan/buffer.hpp>
#include <soar/vulkan/command_queue.hpp>
#include <soar/core/logging.hpp>
#include <iostream>
#include <vector>
#include <cassert>

void test_vulkan_context_initialization() {
    std::cout << "[TEST] Running test_vulkan_context_initialization..." << std::endl;
    soar::vk::VulkanContext ctx(false);
    assert(ctx.instance() != VK_NULL_HANDLE);
    assert(ctx.physical_device() != VK_NULL_HANDLE);
    assert(ctx.device() != VK_NULL_HANDLE);
    assert(ctx.compute_queue() != VK_NULL_HANDLE);
    assert(ctx.command_pool() != VK_NULL_HANDLE);

    const auto& info = ctx.device_info();
    std::cout << "  Device Name: " << info.device_name << std::endl;
    std::cout << "  Device Type: " << static_cast<int>(info.device_type) << std::endl;
    std::cout << "  API Version: " << VK_VERSION_MAJOR(info.api_version) << "."
              << VK_VERSION_MINOR(info.api_version) << "."
              << VK_VERSION_PATCH(info.api_version) << std::endl;
    std::cout << "  Compute Queue Family: " << ctx.compute_queue_family_index() << std::endl;
    std::cout << "  -> test_vulkan_context_initialization PASSED" << std::endl;
}

void test_vulkan_buffer_transfer() {
    std::cout << "[TEST] Running test_vulkan_buffer_transfer..." << std::endl;
    soar::vk::VulkanContext ctx(false);
    soar::vk::CommandQueue queue(ctx);

    constexpr size_t NUM_ELEMENTS = 1024 * 1024; // 1M floats = 4 MB
    constexpr size_t BYTES = NUM_ELEMENTS * sizeof(float);

    std::vector<float> host_src(NUM_ELEMENTS);
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        host_src[i] = static_cast<float>(i) * 0.5f + 1.234f;
    }

    // Allocate host staging buffer and device local storage buffer
    soar::vk::VulkanBuffer staging_src(ctx, BYTES, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer device_storage(ctx, BYTES, soar::vk::BufferUsageType::DeviceStorage);
    soar::vk::VulkanBuffer staging_dst(ctx, BYTES, soar::vk::BufferUsageType::StagingHost);

    // 1. Upload from CPU memory to staging_src
    staging_src.upload_host(host_src.data(), BYTES);

    // 2. Async copy staging_src -> device_storage, then device_storage -> staging_dst
    queue.execute_sync([&](VkCommandBuffer cmd) {
        device_storage.copy_from(cmd, staging_src, BYTES);

        // Memory barrier ensuring write completes before read
        soar::vk::CommandQueue::memory_barrier(
            ctx,
            cmd,
            device_storage,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT
        );

        staging_dst.copy_from(cmd, device_storage, BYTES);
    });

    // 3. Download from staging_dst to CPU memory
    std::vector<float> host_dst(NUM_ELEMENTS, 0.0f);
    staging_dst.download_host(host_dst.data(), BYTES);

    // 4. Verify byte-for-byte exact equality
    size_t mismatch_count = 0;
    for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
        if (host_dst[i] != host_src[i]) {
            if (mismatch_count < 5) {
                std::cerr << "Mismatch at " << i << ": expected " << host_src[i]
                          << ", got " << host_dst[i] << std::endl;
            }
            mismatch_count++;
        }
    }

    assert(mismatch_count == 0);
    std::cout << "  Transferred and verified " << (BYTES / (1024 * 1024))
              << " MB device memory with 0 mismatches!" << std::endl;
    std::cout << "  -> test_vulkan_buffer_transfer PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "  SOAR Vulkan Backend Verification     " << std::endl;
        std::cout << "========================================" << std::endl;

        test_vulkan_context_initialization();
        test_vulkan_buffer_transfer();

        std::cout << std::endl;
        std::cout << ">>> ALL VULKAN BACKEND TESTS PASSED! <<<" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Vulkan test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
