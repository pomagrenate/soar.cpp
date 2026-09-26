#pragma once

#include <soar/tensor/tensor.hpp>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <utility>

namespace soar::nn {

/**
 * @brief Base class for all neural network modules in C++20.
 */
class Module : public std::enable_shared_from_this<Module> {
public:
    explicit Module(std::string name = "") : module_name_(std::move(name)) {}
    virtual ~Module() = default;

    // Single-input forward
    virtual TensorPtr forward(const TensorPtr& input) {
        return forward(std::vector<TensorPtr>{input});
    }

    // Multi-input forward
    virtual TensorPtr forward(const std::vector<TensorPtr>& inputs) {
        if (inputs.empty()) throw ShapeError("Forward called with empty input vector");
        return forward(inputs[0]);
    }

    TensorPtr operator()(const TensorPtr& input) {
        return forward(input);
    }

    TensorPtr operator()(const std::vector<TensorPtr>& inputs) {
        return forward(inputs);
    }

    void register_parameter(const std::string& name, TensorPtr param) {
        parameters_[name] = std::move(param);
    }

    void register_submodule(const std::string& name, std::shared_ptr<Module> submodule) {
        submodules_[name] = std::move(submodule);
    }

    [[nodiscard]] std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> result;
        for (const auto& [_, p] : parameters_) {
            result.push_back(p);
        }
        for (const auto& [_, m] : submodules_) {
            auto sub_params = m->parameters();
            result.insert(result.end(), sub_params.begin(), sub_params.end());
        }
        return result;
    }

    [[nodiscard]] std::vector<std::pair<std::string, TensorPtr>> named_parameters(const std::string& prefix = "") const {
        std::vector<std::pair<std::string, TensorPtr>> result;
        for (const auto& [name, p] : parameters_) {
            result.emplace_back(prefix.empty() ? name : prefix + "." + name, p);
        }
        for (const auto& [name, m] : submodules_) {
            std::string sub_prefix = prefix.empty() ? name : prefix + "." + name;
            auto sub_params = m->named_parameters(sub_prefix);
            result.insert(result.end(), sub_params.begin(), sub_params.end());
        }
        return result;
    }

    void zero_grad() {
        for (auto& p : parameters()) {
            p->zero_grad();
        }
    }

    virtual void train(bool mode = true) {
        training_ = mode;
        for (auto& [_, m] : submodules_) {
            m->train(mode);
        }
    }

    void eval() {
        train(false);
    }

    [[nodiscard]] bool is_training() const noexcept { return training_; }

    virtual void to_device(vk::VulkanContext& ctx) {
        for (auto& [_, p] : parameters_) {
            p->to_device(ctx);
        }
        for (auto& [_, m] : submodules_) {
            m->to_device(ctx);
        }
    }

    virtual void to_cuda(int device_id = 0) {
        for (auto& [_, p] : parameters_) {
            p->to_cuda(device_id);
        }
        for (auto& [_, m] : submodules_) {
            m->to_cuda(device_id);
        }
    }

    virtual void to_host() {
        for (auto& [_, p] : parameters_) {
            p->to_host();
        }
        for (auto& [_, m] : submodules_) {
            m->to_host();
        }
    }

    [[nodiscard]] bool is_cuda() const {
        for (const auto& [_, p] : parameters_) {
            if (p && p->is_cuda()) return true;
        }
        for (const auto& [_, m] : submodules_) {
            if (m && m->is_cuda()) return true;
        }
        return false;
    }

    [[nodiscard]] const std::string& name() const noexcept { return module_name_; }

protected:
    std::string module_name_;
    bool training_{true};
    std::map<std::string, TensorPtr> parameters_;
    std::map<std::string, std::shared_ptr<Module>> submodules_;
};

} // namespace soar::nn
