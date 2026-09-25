#pragma once

#include <iostream>
#include <string_view>
#include <format>

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

template <typename... Args>
void log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) {
    if (level < get_log_level()) return;
    std::string s = std::format(fmt, std::forward<Args>(args)...);
    log_message(level, s);
}

} // namespace soar::core

#define SOAR_LOG_DEBUG(...) ::soar::core::log(::soar::core::LogLevel::Debug, __VA_ARGS__)
#define SOAR_LOG_INFO(...)  ::soar::core::log(::soar::core::LogLevel::Info, __VA_ARGS__)
#define SOAR_LOG_WARN(...)  ::soar::core::log(::soar::core::LogLevel::Warn, __VA_ARGS__)
#define SOAR_LOG_ERROR(...) ::soar::core::log(::soar::core::LogLevel::Error, __VA_ARGS__)
