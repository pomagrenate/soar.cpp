#include <soar/vulkan/context.hpp>
#include <soar/core/logging.hpp>
#include <soar/core/error.hpp>

#include <vector>
#include <cstring>
#include <algorithm>

namespace soar::vk {

VulkanContext::VulkanContext(bool enable_validation_layers) {
    init_instance(enable_validation_layers);
    pick_physical_device();
    init_logical_device();
    init_command_pool();
    SOAR_LOG_INFO("VulkanContext initialized successfully on device: {}", device_info_.device_name);
}

VulkanContext::~VulkanContext() {
    cleanup();
}

VulkanContext::VulkanContext(VulkanContext&& other) noexcept {
    *this = std::move(other);
}

VulkanContext& VulkanContext::operator=(VulkanContext&& other) noexcept {
    if (this != &other) {
        cleanup();
        loader_ = std::move(other.loader_);
        instance_ = other.instance_;
        physical_device_ = other.physical_device_;
        device_ = other.device_;
        compute_queue_ = other.compute_queue_;
        compute_queue_family_index_ = other.compute_queue_family_index_;
        command_pool_ = other.command_pool_;
        memory_properties_ = other.memory_properties_;
        device_info_ = other.device_info_;

        other.instance_ = VK_NULL_HANDLE;
        other.physical_device_ = VK_NULL_HANDLE;
        other.device_ = VK_NULL_HANDLE;
        other.compute_queue_ = VK_NULL_HANDLE;
        other.command_pool_ = VK_NULL_HANDLE;
    }
    return *this;
}

void VulkanContext::cleanup() noexcept {
    if (device_ != VK_NULL_HANDLE) {
        if (command_pool_ != VK_NULL_HANDLE) {
            loader_.vkDestroyCommandPool(device_, command_pool_, nullptr);
            command_pool_ = VK_NULL_HANDLE;
        }
        loader_.vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        compute_queue_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE) {
        loader_.vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

void VulkanContext::init_instance(bool enable_validation) {
    auto attempt_create = [this, enable_validation]() -> VkResult {
        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "SOAR C++ Engine";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "SOAR";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        std::vector<const char*> layers;
        if (enable_validation) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        }

        std::vector<const char*> extensions;

        VkInstanceCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;
        create_info.enabledLayerCount = static_cast<uint32_t>(layers.size());
        create_info.ppEnabledLayerNames = layers.empty() ? nullptr : layers.data();
        create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        create_info.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();

        return loader_.vkCreateInstance(&create_info, nullptr, &instance_);
    };

    VkResult res = attempt_create();
    if (res != VK_SUCCESS) {
        SOAR_LOG_WARN("Standard Vulkan instance creation returned {}. Attempting SwiftShader fallback...", static_cast<int>(res));
        DynamicLoader::configure_swiftshader_fallback();
        loader_.load_library();
        res = attempt_create();
    }

    if (res != VK_SUCCESS) {
        throw DeviceError(std::string("Failed to create Vulkan instance. VkResult: ") + std::to_string(res));
    }

    loader_.load_instance_functions(instance_);
}

void VulkanContext::pick_physical_device() {
    uint32_t device_count = 0;
    loader_.vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0) {
        SOAR_LOG_WARN("0 Vulkan physical devices found. Trying SwiftShader fallback...");
        DynamicLoader::configure_swiftshader_fallback();
        loader_.load_library();
        init_instance(false);
        loader_.vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    }

    if (device_count == 0) {
        throw DeviceError("Failed to find any GPU or compute device with Vulkan support.");
    }

    std::vector<VkPhysicalDevice> devices(device_count);
    loader_.vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    int best_score = -1;
    VkPhysicalDevice best_dev = VK_NULL_HANDLE;

    for (const auto& dev : devices) {
        VkPhysicalDeviceProperties props;
        loader_.vkGetPhysicalDeviceProperties(dev, &props);

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score = 10000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
            score = 5000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
            score = 2000;
        } else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
            score = 1000;
        }

        if (score > best_score) {
            best_score = score;
            best_dev = dev;
        }
    }

    physical_device_ = best_dev;

    // Collect device info
    VkPhysicalDeviceProperties props;
    loader_.vkGetPhysicalDeviceProperties(physical_device_, &props);
    device_info_.device_name = props.deviceName;
    device_info_.vendor_id = props.vendorID;
    device_info_.device_id = props.deviceID;
    device_info_.device_type = props.deviceType;
    device_info_.api_version = props.apiVersion;
    device_info_.driver_version = props.driverVersion;
    device_info_.max_workgroup_invocations = props.limits.maxComputeWorkGroupInvocations;
    device_info_.max_workgroup_size[0] = props.limits.maxComputeWorkGroupSize[0];
    device_info_.max_workgroup_size[1] = props.limits.maxComputeWorkGroupSize[1];
    device_info_.max_workgroup_size[2] = props.limits.maxComputeWorkGroupSize[2];
    device_info_.max_compute_shared_memory_size = props.limits.maxComputeSharedMemorySize;

    loader_.vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties_);

    for (uint32_t i = 0; i < memory_properties_.memoryHeapCount; ++i) {
        if (memory_properties_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            device_info_.total_device_memory += memory_properties_.memoryHeaps[i].size;
        } else {
            device_info_.total_host_memory += memory_properties_.memoryHeaps[i].size;
        }
    }

    SOAR_LOG_INFO("Selected Vulkan Device: '{}' [Type: {}, API: {}.{}.{}]",
                  device_info_.device_name,
                  static_cast<int>(device_info_.device_type),
                  VK_VERSION_MAJOR(device_info_.api_version),
                  VK_VERSION_MINOR(device_info_.api_version),
                  VK_VERSION_PATCH(device_info_.api_version));
}

void VulkanContext::init_logical_device() {
    uint32_t queue_family_count = 0;
    loader_.vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    loader_.vkGetPhysicalDeviceQueueFamilyProperties(physical_device_, &queue_family_count, queue_families.data());

    bool found_compute = false;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            compute_queue_family_index_ = i;
            found_compute = true;
            break;
        }
    }

    if (!found_compute) {
        throw DeviceError("Failed to find a compute-capable queue family on physical device.");
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = compute_queue_family_index_;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceFeatures device_features{};

    VkDeviceCreateInfo device_create_info{};
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.pEnabledFeatures = &device_features;

    VkResult res = loader_.vkCreateDevice(physical_device_, &device_create_info, nullptr, &device_);
    if (res != VK_SUCCESS) {
        throw DeviceError(std::string("Failed to create Vulkan logical device. VkResult: ") + std::to_string(res));
    }

    loader_.load_device_functions(device_);
    loader_.vkGetDeviceQueue(device_, compute_queue_family_index_, 0, &compute_queue_);
}

void VulkanContext::init_command_pool() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = compute_queue_family_index_;

    VkResult res = loader_.vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_);
    if (res != VK_SUCCESS) {
        throw DeviceError(std::string("Failed to create Vulkan compute command pool. VkResult: ") + std::to_string(res));
    }
}

uint32_t VulkanContext::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) const {
    for (uint32_t i = 0; i < memory_properties_.memoryTypeCount; ++i) {
        if ((type_filter & (1 << i)) &&
            (memory_properties_.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw MemoryError("Failed to find suitable Vulkan memory type for required properties.");
}

void VulkanContext::wait_idle() const {
    if (device_ != VK_NULL_HANDLE) {
        loader_.vkDeviceWaitIdle(device_);
    }
}

} // namespace soar::vk
