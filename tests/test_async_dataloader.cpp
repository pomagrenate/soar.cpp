#include <soar/data/dataloader.hpp>
#include <iostream>
#include <cassert>
#include <atomic>
#include <unordered_set>

using namespace soar;
using namespace soar::data;

class SyntheticDataset : public Dataset {
public:
    explicit SyntheticDataset(size_t num_samples) : num_samples_(num_samples) {}

    [[nodiscard]] size_t size() const override {
        return num_samples_;
    }

    void get_item(size_t index, std::span<float> out_data, std::span<float> out_target) override {
        for (size_t i = 0; i < out_data.size(); ++i) {
            out_data[i] = static_cast<float>(index * 10 + i);
        }
        for (size_t i = 0; i < out_target.size(); ++i) {
            out_target[i] = static_cast<float>(index % 2);
        }
    }

    [[nodiscard]] core::Shape data_shape() const override {
        return core::Shape({4});
    }

    [[nodiscard]] core::Shape target_shape() const override {
        return core::Shape({1});
    }

private:
    size_t num_samples_;
};

void test_async_dataloader_multithreaded() {
    std::cout << "[Test] Multi-threaded Asynchronous Prefetching DataLoader..." << std::endl;

    constexpr size_t total_samples = 128;
    constexpr size_t batch_size = 16;
    auto dataset = std::make_shared<SyntheticDataset>(total_samples);

    DataLoaderOptions options;
    options.batch_size = batch_size;
    options.workers = 4;
    options.prefetch_factor = 2;
    options.shuffle = false;
    options.drop_last = false;
    options.pin_memory = true;

    DataLoader loader(dataset, options);
    assert(loader.total_batches() == 8);

    size_t batch_count = 0;
    size_t total_elements_read = 0;

    for (const auto& batch : loader) {
        assert(batch.data != nullptr);
        assert(batch.target != nullptr);
        assert(batch.data->shape()[0] == static_cast<int64_t>(batch_size));
        assert(batch.data->shape()[1] == 4);
        assert(batch.target->shape()[0] == static_cast<int64_t>(batch_size));
        assert(batch.target->shape()[1] == 1);
        assert(batch.is_pinned);

        batch_count++;
        total_elements_read += batch.data->shape()[0];
    }

    assert(batch_count == 8);
    assert(total_elements_read == total_samples);
    std::cout << "  Passed asynchronous DataLoader prefetching with 4 worker threads." << std::endl;
}

int main() {
    std::cout << "=== Running SOAR DataLoader Tests ===" << std::endl;
    test_async_dataloader_multithreaded();
    std::cout << "ALL DATALOADER TESTS PASSED!" << std::endl;
    return 0;
}
