#include "soar/core/logging.hpp"
#include <iostream>
#include <chrono>

namespace soar::core {

static LogLevel g_log_level = LogLevel::Info;

void set_log_level(LogLevel level) noexcept {
    g_log_level = level;
}

LogLevel get_log_level() noexcept {
    return g_log_level;
}

void log_message(LogLevel level, std::string_view msg) {
    const char* prefix = "[INFO]";
    switch (level) {
        case LogLevel::Debug: prefix = "[DEBUG]"; break;
        case LogLevel::Info:  prefix = "[INFO] "; break;
        case LogLevel::Warn:  prefix = "[WARN] "; break;
        case LogLevel::Error: prefix = "[ERROR]"; break;
        case LogLevel::Fatal: prefix = "[FATAL]"; break;
    }
    std::cerr << prefix << " " << msg << "\n";
}

} // namespace soar::core
