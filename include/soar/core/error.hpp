#pragma once

#include <string>
#include <string_view>
#include <stdexcept>
#include <variant>
#include <utility>

namespace soar::core {

enum class StatusCode : uint8_t {
    Success = 0,
    InvalidArgument,
    OutOfMemory,
    DeviceLost,
    InitializationFailed,
    ShaderCompilationFailed,
    FileNotFound,
    IoError,
    ShapeMismatch,
    InternalError,
};

class Status {
public:
    Status() noexcept : code_(StatusCode::Success) {}
    Status(StatusCode code, std::string_view msg = "") : code_(code), message_(msg) {}

    [[nodiscard]] constexpr bool ok() const noexcept { return code_ == StatusCode::Success; }
    [[nodiscard]] constexpr StatusCode code() const noexcept { return code_; }
    [[nodiscard]] std::string_view message() const noexcept { return message_; }

    static Status success() noexcept { return Status(); }

private:
    StatusCode code_;
    std::string message_;
};

template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Status status) : data_(std::move(status)) {}
    Result(StatusCode code, std::string_view msg = "") : data_(Status(code, msg)) {}

    [[nodiscard]] bool ok() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    [[nodiscard]] const T& value() const & {
        return std::get<T>(data_);
    }

    [[nodiscard]] T& value() & {
        return std::get<T>(data_);
    }

    [[nodiscard]] T&& value() && {
        return std::get<T>(std::move(data_));
    }

    [[nodiscard]] Status status() const {
        if (ok()) return Status::success();
        return std::get<Status>(data_);
    }

private:
    std::variant<T, Status> data_;
};

} // namespace soar::core

namespace soar {
    using core::Status;
    using core::StatusCode;
    using core::Result;

    class SoarException : public std::runtime_error {
    public:
        explicit SoarException(const std::string& msg) : std::runtime_error(msg) {}
    };

    class DeviceError : public SoarException {
    public:
        explicit DeviceError(const std::string& msg) : SoarException("[DeviceError] " + msg) {}
    };

    class MemoryError : public SoarException {
    public:
        explicit MemoryError(const std::string& msg) : SoarException("[MemoryError] " + msg) {}
    };

    class ShaderError : public SoarException {
    public:
        explicit ShaderError(const std::string& msg) : SoarException("[ShaderError] " + msg) {}
    };

    class ShapeError : public SoarException {
    public:
        explicit ShapeError(const std::string& msg) : SoarException("[ShapeError] " + msg) {}
    };
} // namespace soar
