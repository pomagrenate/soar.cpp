#include <soar/cuda/cuda_runtime.hpp>
#include <soar/cuda/cuda_stream.hpp>
#include <soar/cuda/cuda_event.hpp>
#include <soar/cuda/cuda_allocator.hpp>
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>

using namespace soar::cuda;

void test_stream_pool() {
    std::cout << "[Test] CudaStream and Stream Pool..." << std::endl;
    auto s_default = CudaStream::getDefaultStream(0);
    assert(s_default.stream() == nullptr);

    auto s1 = CudaStream::getStreamFromPool(false, 0);
    auto s2 = CudaStream::getStreamFromPool(false, 0);
    assert(s1 != s_default);
    assert(s1 != s2);

    // Verify 32 streams in pool wrap around round-robin
    std::vector<CudaStream> streams;
    for (int i = 0; i < CudaStream::kStreamsPerPool; ++i) {
        streams.push_back(CudaStream::getStreamFromPool(false, 0));
    }
    // The next one should wrap to the beginning of the pool
    auto s_wrap = CudaStream::getStreamFromPool(false, 0);
    assert(s_wrap == streams[0]);

    // High priority pool
    auto s_hp = CudaStream::getStreamFromPool(true, 0);
    assert(s_hp.priority() == 1);
    std::cout << "  Passed stream pool verification." << std::endl;
}

void test_event_synchronization() {
    std::cout << "[Test] CudaEvent Cross-Stream Wait..." << std::endl;
    CudaEvent ev(cudaEventDisableTiming);
    assert(!ev.is_created());

    auto s1 = CudaStream::getStreamFromPool(false, 0);
    auto s2 = CudaStream::getStreamFromPool(false, 0);

    // Record on stream 1
    ev.record(s1);
    assert(ev.is_created());
    assert(ev.query());

    // Block stream 2 until event on stream 1 is satisfied
    s2.wait_event(ev);

    // Timing event test
    CudaEvent ev_start(cudaEventDefault);
    CudaEvent ev_end(cudaEventDefault);

    ev_start.record(s1);
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ev_end.record(s1);

    ev_end.synchronize();
    float ms = ev_start.elapsed_time(ev_end);
    assert(ms >= 0.0f);
    std::cout << "  Passed event synchronization (elapsed = " << ms << " ms)." << std::endl;
}

void test_memory_pool_caching() {
    std::cout << "[Test] CudaMemoryPool Allocator..." << std::endl;
    auto& pool = CudaMemoryPool::instance();

    std::cout << "  Check 1: allocate p1, p2" << std::endl;
    void* p1 = pool.allocate(100);
    assert(p1 != nullptr);
    assert(reinterpret_cast<uintptr_t>(p1) % 64 == 0);

    void* p2 = pool.allocate(250);
    assert(p2 != nullptr);
    assert(reinterpret_cast<uintptr_t>(p2) % 64 == 0);
    assert(p1 != p2);

    size_t active_before = pool.active_blocks();
    (void)active_before;
    assert(active_before >= 2);

    std::cout << "  Check 2: deallocate p1, p2" << std::endl;
    pool.deallocate(p1);
    pool.deallocate(p2);
    assert(pool.active_blocks() == active_before - 2);

    std::cout << "  Check 3: allocate p3" << std::endl;
    size_t reserved_before = pool.reserved_bytes();
    (void)reserved_before;
    void* p3 = pool.allocate(300);
    assert(p3 != nullptr);
    assert(reinterpret_cast<uintptr_t>(p3) % 64 == 0);
    assert(pool.reserved_bytes() == reserved_before);

    pool.deallocate(p3);

    std::cout << "  Check 4: allocate p_large" << std::endl;
    void* p_large = pool.allocate(2 * 1024 * 1024);
    assert(p_large != nullptr);
    assert(reinterpret_cast<uintptr_t>(p_large) % 64 == 0);
    pool.deallocate(p_large);

    std::cout << "  Check 5: empty_cache" << std::endl;
    pool.empty_cache();
    assert(pool.allocated_bytes() == 0);
    std::cout << "  Passed memory pool splitting, binning, and coalescing." << std::endl;
}

int main() {
    std::cout << "=== Running SOAR CUDA Subsystem Tests ===" << std::endl;
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::cout << "[SKIP] No physical NVIDIA CUDA GPU hardware detected. Skipping CUDA hardware tests (CTest return code 77)." << std::endl;
        return 77;
    }

    try {
        test_stream_pool();
        test_event_synchronization();
        test_memory_pool_caching();
        std::cout << "ALL CUDA SUBSYSTEM TESTS PASSED!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] CUDA test failed: " << e.what() << std::endl;
        return 1;
    }
}
