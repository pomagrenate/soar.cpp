#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <sstream>
#include <cstdint>

namespace soar::core {

enum class LogLevel : uint8_t {
    Debug = 0,
    Info,
    Warn,
    Error,
    Fatal,
};

void set_log_level(LogLevel level) noexcept;
LogLevel get_log_level() noexcept;

void log_message(LogLevel level, std::string_view msg);

namespace detail {
template <typename T>
inline void format_arg(std::ostringstream& oss, std::string_view fmt, size_t& pos, const T& arg) {
    size_t next = fmt.find("{}", pos);
    if (next != std::string_view::npos) {
        oss << fmt.substr(pos, next - pos);
        oss << arg;
        pos = next + 2;
    }
}
} // namespace detail

template <typename... Args>
inline void log(LogLevel level, std::string_view fmt, const Args&... args) {
    if (level < get_log_level()) return;
    if constexpr (sizeof...(Args) == 0) {
        log_message(level, fmt);
    } else {
        std::ostringstream oss;
        size_t pos = 0;
        (detail::format_arg(oss, fmt, pos, args), ...);
        if (pos < fmt.size()) {
            oss << fmt.substr(pos);
        }
        log_message(level, oss.str());
    }
}

} // namespace soar::core

#define SOAR_LOG_DEBUG(...) ::soar::core::log(::soar::core::LogLevel::Debug, __VA_ARGS__)
#define SOAR_LOG_INFO(...)  ::soar::core::log(::soar::core::LogLevel::Info, __VA_ARGS__)
#define SOAR_LOG_WARN(...)  ::soar::core::log(::soar::core::LogLevel::Warn, __VA_ARGS__)
#define SOAR_LOG_ERROR(...) ::soar::core::log(::soar::core::LogLevel::Error, __VA_ARGS__)
