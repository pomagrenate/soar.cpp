#pragma once

#include <exception>
#include <string>
#include <utility>

namespace soar::data {

/// An exception thrown when a DataLoader's worker thread encounters an exception.
/// Matches PyTorch torch::data::WorkerException.
struct WorkerException : public std::exception {
    explicit WorkerException(std::exception_ptr original)
        : original_exception(std::move(original)),
          message("Caught exception in DataLoader worker thread.") {
        if (original_exception) {
            try {
                std::rethrow_exception(original_exception);
            } catch (const std::exception& e) {
                message += " Original message: ";
                message += e.what();
            } catch (...) {
                message += " Unknown exception caught.";
            }
        }
    }

    [[nodiscard]] const char* what() const noexcept override {
        return message.c_str();
    }

    std::exception_ptr original_exception;
    std::string message;
};

} // namespace soar::data
