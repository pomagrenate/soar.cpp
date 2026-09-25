#pragma once

#include <soar/nn/soar_model.hpp>
#include <string>

namespace soar::core {

struct ModelConfig {
    size_t in_channels{1};
    size_t num_classes{1};
    nn::ModelVariant variant{nn::ModelVariant::Nano};
    std::string variant_str{"nano"};
};

class ConfigParser {
public:
    /**
     * @brief Parse model configuration from a YAML or JSON file.
     * @param config_path Path to model config file (e.g. configs/models/soar_nano1.yaml).
     */
    static ModelConfig load_model_config(const std::string& config_path);
};

} // namespace soar::core
