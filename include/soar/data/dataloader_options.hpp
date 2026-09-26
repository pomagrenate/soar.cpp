#pragma once

#include <chrono>
#include <cstddef>
#include <optional>

namespace soar::data {

/// Options to configure a DataLoader matching PyTorch torch::data::DataLoaderOptions.
struct DataLoaderOptions {
    DataLoaderOptions() = default;
    /* implicit */ DataLoaderOptions(size_t batch_size)
        : batch_size(batch_size) {}

    /// The size of each batch to fetch.
    size_t batch_size{1};

    /// The number of worker threads to launch. If zero, the main thread will
    /// synchronously perform data loading.
    size_t workers{0};

    /// Prefetch factor: number of batches loaded in advance per worker.
    size_t prefetch_factor{2};

    /// The maximum number of jobs to enqueue for fetching by worker threads.
    /// Defaults to two times the number of worker threads (or workers * prefetch_factor).
    std::optional<size_t> max_jobs{std::nullopt};

    /// An optional limit on the time to wait for the next batch.
    std::optional<std::chrono::milliseconds> timeout{std::nullopt};

    /// Whether to enforce ordering of batches when multiple are loaded
    /// asynchronously by worker threads.
    bool enforce_ordering{true};

    /// Whether to omit the last batch if it contains less than batch_size examples.
    bool drop_last{false};

    /// Whether to shuffle indices.
    bool shuffle{true};

    /// Whether to pin memory buffers for zero-copy GPU DMA streaming.
    bool pin_memory{false};

    // PyTorch-style fluent builder methods
    DataLoaderOptions& set_batch_size(size_t val) { batch_size = val; return *this; }
    DataLoaderOptions& set_workers(size_t val) { workers = val; return *this; }
    DataLoaderOptions& set_prefetch_factor(size_t val) { prefetch_factor = val; return *this; }
    DataLoaderOptions& set_max_jobs(std::optional<size_t> val) { max_jobs = val; return *this; }
    DataLoaderOptions& set_timeout(std::optional<std::chrono::milliseconds> val) { timeout = val; return *this; }
    DataLoaderOptions& set_enforce_ordering(bool val) { enforce_ordering = val; return *this; }
    DataLoaderOptions& set_drop_last(bool val) { drop_last = val; return *this; }
    DataLoaderOptions& set_shuffle(bool val) { shuffle = val; return *this; }
    DataLoaderOptions& set_pin_memory(bool val) { pin_memory = val; return *this; }
};

/// Like DataLoaderOptions, but without unconfigured state.
/// Matches PyTorch torch::data::FullDataLoaderOptions.
struct FullDataLoaderOptions {
    explicit FullDataLoaderOptions(DataLoaderOptions options)
        : batch_size(options.batch_size),
          workers(options.workers),
          max_jobs(options.max_jobs.value_or(options.workers > 0 ? options.workers * options.prefetch_factor : 2)),
          timeout(options.timeout),
          enforce_ordering(options.enforce_ordering),
          drop_last(options.drop_last),
          shuffle(options.shuffle),
          pin_memory(options.pin_memory) {}

    size_t batch_size{1};
    size_t workers{0};
    size_t max_jobs{2};
    std::optional<std::chrono::milliseconds> timeout{std::nullopt};
    bool enforce_ordering{true};
    bool drop_last{false};
    bool shuffle{true};
    bool pin_memory{false};
};

} // namespace soar::data
