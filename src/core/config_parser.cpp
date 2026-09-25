#include <soar/core/config_parser.hpp>
#include <soar/core/logging.hpp>

#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

namespace soar::core {

namespace {

std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

} // anonymous namespace

ModelConfig ConfigParser::load_model_config(const std::string& config_path) {
    std::string path_to_open = config_path;
    FILE* in = std::fopen(path_to_open.c_str(), "r");
    if (!in && (path_to_open.rfind("../", 0) != 0)) {
        path_to_open = "../" + config_path;
        in = std::fopen(path_to_open.c_str(), "r");
    }
    if (!in) {
        throw DeviceError("Failed to open model configuration file: " + config_path);
    }

    ModelConfig cfg;
    char buffer[4096];

    while (std::fgets(buffer, sizeof(buffer), in)) {
        std::string line(buffer);
        // Strip comments
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        line = trim(line);
        if (line.empty()) continue;

        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, colon_pos));
        std::string val = trim(line.substr(colon_pos + 1));

        if (key == "nc") {
            cfg.num_classes = std::stoul(val);
        } else if (key == "channels") {
            cfg.in_channels = std::stoul(val);
        } else if (key == "scale") {
            std::string s = val;
            for (auto& c : s) c = static_cast<char>(std::tolower(c));
            if (s == "n" || s == "nano") {
                cfg.variant = nn::ModelVariant::Nano;
                cfg.variant_str = "nano";
            } else if (s == "s" || s == "small") {
                cfg.variant = nn::ModelVariant::Small;
                cfg.variant_str = "small";
            } else if (s == "m" || s == "medium") {
                cfg.variant = nn::ModelVariant::Medium;
                cfg.variant_str = "medium";
            } else if (s == "l" || s == "large") {
                cfg.variant = nn::ModelVariant::Large;
                cfg.variant_str = "large";
            } else if (s == "x" || s == "xlarge") {
                cfg.variant = nn::ModelVariant::XLarge;
                cfg.variant_str = "xlarge";
            }
        }
    }
    std::fclose(in);

    SOAR_LOG_INFO("Parsed model configuration from {}: scale='{}', in_channels={}, nc={}",
                  config_path, cfg.variant_str, cfg.in_channels, cfg.num_classes);
    return cfg;
}

} // namespace soar::core
