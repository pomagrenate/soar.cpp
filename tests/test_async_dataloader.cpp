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

void test_pytorch_exact_stateless_dataloader() {
    std::cout << "[Test] PyTorch Native StatelessDataLoader with OrderedSequencer..." << std::endl;

    struct NativeTorchDataset {
        using BatchType = std::vector<int>;
        using BatchRequestType = std::vector<size_t>;

        size_t total_samples = 64;

        [[nodiscard]] std::optional<size_t> size() const noexcept {
            return total_samples;
        }

        BatchType get_batch(const std::vector<size_t>& indices) const {
            // Simulate random worker sleep to test out-of-order reordering by OrderedSequencer
            if (!indices.empty() && indices[0] % 2 == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            std::vector<int> batch;
            batch.reserve(indices.size());
            for (size_t idx : indices) {
                batch.push_back(static_cast<int>(idx));
            }
            return batch;
        }
    };

    NativeTorchDataset ds;
    DataLoaderOptions opts;
    opts.batch_size = 8;
    opts.workers = 4;
    opts.enforce_ordering = true;

    auto loader = make_data_loader(ds, samplers::SequentialSampler(64), opts);

    // Epoch 1
    int expected_val = 0;
    size_t batch_count = 0;
    for (const auto& batch : *loader) {
        assert(batch.size() == 8);
        for (int val : batch) {
            assert(val == expected_val);
            expected_val++;
        }
        batch_count++;
    }
    assert(batch_count == 8);
    assert(expected_val == 64);
    std::cout << "  Epoch 1 passed: OrderedSequencer preserved exact order across 4 worker threads." << std::endl;

    // Epoch 2 (verifies sampler reset)
    expected_val = 0;
    batch_count = 0;
    for (const auto& batch : *loader) {
        assert(batch.size() == 8);
        for (int val : batch) {
            assert(val == expected_val);
            expected_val++;
        }
        batch_count++;
    }
    assert(batch_count == 8);
    assert(expected_val == 64);
    std::cout << "  Epoch 2 passed: DataLoader properly reset and re-streamed." << std::endl;
}

void test_pytorch_worker_exception() {
    std::cout << "[Test] PyTorch WorkerException propagation..." << std::endl;

    struct FaultyDataset {
        using BatchType = std::vector<int>;
        using BatchRequestType = std::vector<size_t>;

        [[nodiscard]] std::optional<size_t> size() const noexcept {
            return 32;
        }

        BatchType get_batch(const std::vector<size_t>& indices) const {
            for (size_t idx : indices) {
                if (idx == 10) {
                    throw std::runtime_error("Simulated corrupted image sector on index 10");
                }
            }
            return std::vector<int>(indices.begin(), indices.end());
        }
    };

    FaultyDataset ds;
    DataLoaderOptions opts;
    opts.batch_size = 4;
    opts.workers = 2;

    auto loader = make_data_loader(ds, samplers::SequentialSampler(32), opts);

    bool caught = false;
    try {
        for (const auto& batch : *loader) {
            (void)batch;
        }
    } catch (const WorkerException& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("Simulated corrupted image sector") != std::string::npos);
        std::cout << "  Successfully caught WorkerException: " << e.what() << std::endl;
    }
    assert(caught);
}

int main() {
    std::cout << "=== Running SOAR DataLoader Tests ===" << std::endl;
    test_async_dataloader_multithreaded();
    test_pytorch_exact_stateless_dataloader();
    test_pytorch_worker_exception();
    std::cout << "ALL DATALOADER TESTS PASSED!" << std::endl;
    return 0;
}
