#include <iostream>
#include <cassert>
#include <vector>
#include <soar/memory/palloc_allocator.hpp>
#include <soar/tensor/tensor.hpp>
#include "soar/soar.hpp"

using namespace soar::core;
using namespace soar::memory;

void test_palloc_arena_alignment() {
    std::cout << "[TEST] Running test_palloc_arena_alignment...\n";
    PallocArena arena(1024 * 1024); // 1 MB capacity
    assert(arena.is_valid());

    // 1. Allocate 64-byte aligned vector
    void* ptr64 = arena.allocate_bytes(128, 64);
    assert(ptr64 != nullptr);
    assert(reinterpret_cast<uintptr_t>(ptr64) % 64 == 0);

    // 2. Allocate 256-byte aligned vector (Vulkan uniform buffer alignment)
    void* ptr256 = arena.allocate_bytes(512, 256);
    assert(ptr256 != nullptr);
    assert(reinterpret_cast<uintptr_t>(ptr256) % 256 == 0);

    // 3. Verify write and read
    float* fptr = static_cast<float*>(ptr64);
    fptr[0] = 3.14159f;
    fptr[31] = 2.71828f;
    assert(fptr[0] == 3.14159f);
    assert(fptr[31] == 2.71828f);

    // 4. Test O(1) bulk reset
    size_t before_reset = arena.allocated();
    assert(before_reset > 0);
    arena.reset();
    assert(arena.allocated() == 0);

    // Re-allocate after reset
    void* ptr_after = arena.allocate_bytes(128, 64);
    assert(ptr_after != nullptr);
    assert(reinterpret_cast<uintptr_t>(ptr_after) % 64 == 0);
    std::cout << "  -> test_palloc_arena_alignment PASSED\n";
}

void test_host_arena_tensor_view() {
    std::cout << "[TEST] Running test_host_arena_tensor_view...\n";
    HostArena host_arena(64 * 1024 * 1024); // 64 MB

    Shape shape(1, 3, 256, 256);
    auto tensor = host_arena.create_tensor<float>(shape, 64);
    assert(tensor.data() != nullptr);
    assert(tensor.shape() == shape);
    assert(tensor.numel() == 1 * 3 * 256 * 256);
    assert(tensor.bytes() == static_cast<size_t>(1 * 3 * 256 * 256 * sizeof(float)));

    // Test element indexing (b, c, y, x)
    tensor(0, 0, 0, 0) = 1.0f;
    tensor(0, 1, 128, 128) = 42.0f;
    tensor(0, 2, 255, 255) = 99.0f;

    assert(tensor(0, 0, 0, 0) == 1.0f);
    assert(tensor(0, 1, 128, 128) == 42.0f);
    assert(tensor(0, 2, 255, 255) == 99.0f);

    host_arena.reset();
    assert(host_arena.allocated() == 0);
    std::cout << "  -> test_host_arena_tensor_view PASSED\n";
}

void test_device_slab_sub_allocator() {
    std::cout << "[TEST] Running test_device_slab_sub_allocator...\n";
    size_t total_vram = 256 * 1024 * 1024; // 256 MB VRAM slab
    DeviceSlabSubAllocator slab(total_vram, 256);

    // 1. Allocate static weights
    auto w1 = slab.sub_allocate(1024 * 1024 * 4, Shape{64, 3, 3, 3}, DataType::Float32, "conv1_weight");
    assert(w1.ok());
    assert(w1.value().offset % 256 == 0);

    auto w2 = slab.sub_allocate(1024 * 1024 * 8, Shape{128, 64, 3, 3}, DataType::Float32, "conv2_weight");
    assert(w2.ok());
    assert(w2.value().offset % 256 == 0);

    // Freeze static parameters
    slab.freeze_static_parameters();
    size_t static_boundary = slab.static_boundary();
    assert(static_boundary > 0);
    assert(static_boundary % 256 == 0);

    // 2. Allocate dynamic activations (Pass 1)
    auto act1 = slab.sub_allocate(512 * 512 * 64 * 2, Shape{1, 64, 512, 512}, DataType::Float16, "act_p2");
    assert(act1.ok());
    assert(act1.value().offset >= static_boundary);
    assert(act1.value().offset % 256 == 0);

    auto act2 = slab.sub_allocate(256 * 256 * 128 * 2, Shape{1, 128, 256, 256}, DataType::Float16, "act_p3");
    assert(act2.ok());
    assert(act2.value().offset % 256 == 0);

    // 3. Reset dynamic activations (end of frame)
    slab.reset_dynamic_allocations();
    assert(slab.current_offset() == static_boundary);

    // 4. Allocate dynamic activations (Pass 2) -> must reuse identical offsets
    auto act1_re = slab.sub_allocate(512 * 512 * 64 * 2, Shape{1, 64, 512, 512}, DataType::Float16, "act_p2_re");
    assert(act1_re.ok());
    assert(act1_re.value().offset == act1.value().offset); // Perfect offset recycling!

    std::cout << "  -> test_device_slab_sub_allocator PASSED\n";
}

void test_palloc_allocator_and_tensor() {
    std::cout << "[TEST] Running test_palloc_allocator_and_tensor...\n";

    // 1. PallocAllocator vector allocation
    soar::memory::PallocVector<float> vec;
    vec.resize(1024, 42.0f);
    assert(vec.size() == 1024);
    assert(reinterpret_cast<uintptr_t>(vec.data()) % 64 == 0); // 64-byte aligned
    assert(vec[0] == 42.0f && vec[1023] == 42.0f);

    // 2. Tensor allocation using palloc
    auto tensor = soar::Tensor::randn({1, 3, 128, 128}, 0.0f, 1.0f);
    assert(tensor->numel() == 1 * 3 * 128 * 128);
    assert(reinterpret_cast<uintptr_t>(tensor->data()) % 64 == 0);

    // 3. Dynamic object allocation via overridden new/delete
    auto* dynamic_tensor = new soar::Tensor({1, 1, 64, 64});
    assert(dynamic_tensor != nullptr);
    assert(reinterpret_cast<uintptr_t>(dynamic_tensor->data()) % 64 == 0);
    delete dynamic_tensor;

    // 4. Verify palloc stats show active tracking
    auto stats = soar::memory::get_palloc_process_info();
    std::cout << "  palloc Peak RSS: " << (stats.peak_rss / 1024) << " KB, Current Commit: "
              << (stats.current_commit / 1024) << " KB\n";
    std::cout << "  -> test_palloc_allocator_and_tensor PASSED\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  SOAR Memory Subsystem Verification\n";
    std::cout << "========================================\n";

    test_palloc_arena_alignment();
    test_host_arena_tensor_view();
    test_device_slab_sub_allocator();
    test_palloc_allocator_and_tensor();

    std::cout << "\n>>> ALL MEMORY & ARENA TESTS PASSED SUCCESSFULLY! <<<\n";
    return 0;
}
