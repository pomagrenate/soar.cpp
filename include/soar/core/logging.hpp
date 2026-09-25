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
inline void format_arg(std::string& out, std::string_view fmt, size_t& pos, const T& arg) {
    size_t next = fmt.find("{}", pos);
    if (next != std::string_view::npos) {
        out.append(fmt.substr(pos, next - pos));
        if constexpr (std::is_arithmetic_v<std::decay_t<T>>) {
            out.append(std::to_string(arg));
        } else if constexpr (std::is_convertible_v<T, std::string_view>) {
            out.append(std::string_view(arg));
        } else {
            out.append(arg);
        }
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
        std::string out;
        out.reserve(fmt.size() + 32 * sizeof...(Args));
        size_t pos = 0;
        (detail::format_arg(out, fmt, pos, args), ...);
        if (pos < fmt.size()) {
            out.append(fmt.substr(pos));
        }
        log_message(level, out);
    }
}

} // namespace soar::core

#define SOAR_LOG_DEBUG(...) ::soar::core::log(::soar::core::LogLevel::Debug, __VA_ARGS__)
#define SOAR_LOG_INFO(...)  ::soar::core::log(::soar::core::LogLevel::Info, __VA_ARGS__)
#define SOAR_LOG_WARN(...)  ::soar::core::log(::soar::core::LogLevel::Warn, __VA_ARGS__)
#define SOAR_LOG_ERROR(...) ::soar::core::log(::soar::core::LogLevel::Error, __VA_ARGS__)
