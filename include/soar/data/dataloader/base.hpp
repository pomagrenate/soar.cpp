#pragma once

#include <soar/data/dataloader_options.hpp>
#include <soar/data/detail/data_shuttle.hpp>
#include <soar/data/detail/sequencers.hpp>
#include <soar/data/iterator.hpp>
#include <soar/data/worker_exception.hpp>

#include <cassert>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace soar::data {

/// Base class for all DataLoaders.
/// Matches PyTorch torch::data::DataLoaderBase<Dataset, Batch, BatchRequest>.
template <typename Dataset, typename Batch, typename BatchRequest>
class DataLoaderBase {
public:
    using BatchType = Batch;
    using BatchRequestType = BatchRequest;

    DataLoaderBase(
        DataLoaderOptions options,
        std::unique_ptr<Dataset> main_thread_dataset = nullptr)
        : options_(options),
          main_thread_dataset_(std::move(main_thread_dataset)),
          sequencer_(new_sequencer()) {}

    DataLoaderBase(const DataLoaderBase&) = delete;
    DataLoaderBase(DataLoaderBase&&) = delete;
    DataLoaderBase& operator=(const DataLoaderBase&) = delete;
    DataLoaderBase& operator=(DataLoaderBase&&) = delete;

    virtual ~DataLoaderBase() {
        join();
    }

    /// Returns an input iterator into the DataLoader.
    Iterator<Batch> begin() {
        assert(shuttle_.in_flight_jobs() == 0 &&
               "Attempted to get a new DataLoader iterator while another is not yet exhausted");
        reset();
        return Iterator<Batch>(std::make_unique<detail::ValidIterator<Batch>>(
            [this] { return this->next(); }));
    }

    /// Returns a sentinel iterator.
    Iterator<Batch> end() {
        return Iterator<Batch>(std::make_unique<detail::SentinelIterator<Batch>>());
    }

    /// Joins the DataLoader's worker threads and drains internal queues.
    void join() {
        if (joined_) {
            return;
        }
        shuttle_.drain();
        for (size_t w = 0; w < options_.workers; ++w) {
            push_job(QuitWorker{});
        }
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        joined_ = true;
    }

    [[nodiscard]] const FullDataLoaderOptions& options() const noexcept {
        return options_;
    }

protected:
    struct Sequenced {
        Sequenced() = default;
        Sequenced(size_t sqn) : sequence_number(sqn) {}
        size_t sequence_number{0};
    };

    struct QuitWorker {};

    struct Job : Sequenced {
        Job() = default;
        Job(QuitWorker q, size_t sqn) : Sequenced(sqn), quit(q) {}
        Job(BatchRequest&& req, size_t sqn)
            : Sequenced(sqn), batch_request(std::move(req)) {}
        std::optional<QuitWorker> quit;
        std::optional<BatchRequest> batch_request;
    };

    struct Result : Sequenced {
        Result() = default;
        Result(std::optional<Batch>&& b, size_t sqn)
            : Sequenced(sqn), batch(std::move(b)) {}
        Result(std::exception_ptr exc, size_t sqn)
            : Sequenced(sqn), exception(std::move(exc)) {}
        std::optional<Batch> batch;
        std::exception_ptr exception;
    };

    virtual std::optional<BatchRequestType> get_batch_request() = 0;

    virtual void reset() {
        shuttle_.drain();
        sequence_number_ = 0;
        sequencer_ = new_sequencer();
        prefetch();
    }

    void prefetch(size_t requested_jobs) {
        for (size_t r = 0; r < requested_jobs; ++r) {
            if (auto batch_request = get_batch_request()) {
                this->push_job(std::move(*batch_request));
            } else {
                break;
            }
        }
    }

    void prefetch() {
        prefetch(options_.max_jobs);
    }

    std::optional<BatchType> next() {
        if (options_.workers > 0) {
            while (std::optional<Result> result = this->pop_result()) {
                if (result->exception) {
                    throw WorkerException(result->exception);
                } else if (result->batch) {
                    prefetch(1);
                    return std::move(result->batch);
                }
            }
        } else if (auto batch_request = get_batch_request()) {
            if (main_thread_dataset_) {
                return this->main_thread_dataset_->get_batch(std::move(*batch_request));
            }
        }
        return std::nullopt;
    }

    void worker_thread(Dataset& dataset) {
        while (true) {
            auto job = shuttle_.pop_job();
            if (job.quit) {
                break;
            }
            try {
                auto batch = dataset.get_batch(std::move(*job.batch_request));
                shuttle_.push_result(Result(std::move(batch), job.sequence_number));
            } catch (...) {
                shuttle_.push_result(Result(std::current_exception(), job.sequence_number));
            }
        }
    }

    template <typename T>
    void push_job(T value) {
        shuttle_.push_job(Job(std::move(value), sequence_number_++));
    }

    std::optional<Result> pop_result() {
        return sequencer_->next(
            [this] { return this->shuttle_.pop_result(this->options_.timeout); });
    }

    std::unique_ptr<detail::sequencers::Sequencer<Result>> new_sequencer() {
        if (options_.enforce_ordering) {
            return std::make_unique<detail::sequencers::OrderedSequencer<Result>>(
                options_.max_jobs);
        }
        return std::make_unique<detail::sequencers::NoSequencer<Result>>();
    }

    const FullDataLoaderOptions options_;
    std::unique_ptr<Dataset> main_thread_dataset_;
    size_t sequence_number_{0};
    std::vector<std::thread> workers_;
    detail::DataShuttle<Job, Result> shuttle_;
    std::unique_ptr<detail::sequencers::Sequencer<Result>> sequencer_;
    bool joined_{false};
};

} // namespace soar::data
