#include <soar/vulkan/dynamic_loader.hpp>
#include <soar/core/logging.hpp>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#include <cstdlib>
#endif

#include <vector>
#include <fstream>
#include <filesystem>

namespace soar::vk {

namespace {

void* load_shared_library(const std::string& path) {
#if defined(_WIN32)
    return static_cast<void*>(LoadLibraryA(path.c_str()));
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* get_proc_symbol(void* handle, const char* symbol_name) {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), symbol_name));
#else
    return dlsym(handle, symbol_name);
#endif
}

void free_shared_library(void* handle) {
    if (!handle) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

void set_env_variable(const std::string& name, const std::string& value) {
#if defined(_WIN32)
    SetEnvironmentVariableA(name.c_str(), value.c_str());
    _putenv_s(name.c_str(), value.c_str());
#else
    setenv(name.c_str(), value.c_str(), 1);
#endif
}

} // anonymous namespace

DynamicLoader::DynamicLoader() {
    load_library();
}

DynamicLoader::~DynamicLoader() {
    unload();
}

DynamicLoader::DynamicLoader(DynamicLoader&& other) noexcept {
    *this = std::move(other);
}

DynamicLoader& DynamicLoader::operator=(DynamicLoader&& other) noexcept {
    if (this != &other) {
        unload();
        library_handle_ = other.library_handle_;
        other.library_handle_ = nullptr;

        vkGetInstanceProcAddr = other.vkGetInstanceProcAddr;
        vkCreateInstance = other.vkCreateInstance;
        vkEnumerateInstanceExtensionProperties = other.vkEnumerateInstanceExtensionProperties;
        vkEnumerateInstanceLayerProperties = other.vkEnumerateInstanceLayerProperties;
        vkEnumerateInstanceVersion = other.vkEnumerateInstanceVersion;

        vkDestroyInstance = other.vkDestroyInstance;
        vkEnumeratePhysicalDevices = other.vkEnumeratePhysicalDevices;
        vkGetPhysicalDeviceProperties = other.vkGetPhysicalDeviceProperties;
        vkGetPhysicalDeviceFeatures = other.vkGetPhysicalDeviceFeatures;
        vkGetPhysicalDeviceMemoryProperties = other.vkGetPhysicalDeviceMemoryProperties;
        vkGetPhysicalDeviceQueueFamilyProperties = other.vkGetPhysicalDeviceQueueFamilyProperties;
        vkCreateDevice = other.vkCreateDevice;
        vkGetDeviceProcAddr = other.vkGetDeviceProcAddr;
        vkEnumerateDeviceExtensionProperties = other.vkEnumerateDeviceExtensionProperties;

        vkDestroyDevice = other.vkDestroyDevice;
        vkGetDeviceQueue = other.vkGetDeviceQueue;
        vkQueueSubmit = other.vkQueueSubmit;
        vkQueueWaitIdle = other.vkQueueWaitIdle;
        vkDeviceWaitIdle = other.vkDeviceWaitIdle;

        vkCreateBuffer = other.vkCreateBuffer;
        vkDestroyBuffer = other.vkDestroyBuffer;
        vkGetBufferMemoryRequirements = other.vkGetBufferMemoryRequirements;
        vkAllocateMemory = other.vkAllocateMemory;
        vkFreeMemory = other.vkFreeMemory;
        vkBindBufferMemory = other.vkBindBufferMemory;
        vkMapMemory = other.vkMapMemory;
        vkUnmapMemory = other.vkUnmapMemory;
        vkFlushMappedMemoryRanges = other.vkFlushMappedMemoryRanges;
        vkInvalidateMappedMemoryRanges = other.vkInvalidateMappedMemoryRanges;

        vkCreateShaderModule = other.vkCreateShaderModule;
        vkDestroyShaderModule = other.vkDestroyShaderModule;
        vkCreateDescriptorSetLayout = other.vkCreateDescriptorSetLayout;
        vkDestroyDescriptorSetLayout = other.vkDestroyDescriptorSetLayout;
        vkCreatePipelineLayout = other.vkCreatePipelineLayout;
        vkDestroyPipelineLayout = other.vkDestroyPipelineLayout;
        vkCreateComputePipelines = other.vkCreateComputePipelines;
        vkDestroyPipeline = other.vkDestroyPipeline;

        vkCreateDescriptorPool = other.vkCreateDescriptorPool;
        vkDestroyDescriptorPool = other.vkDestroyDescriptorPool;
        vkAllocateDescriptorSets = other.vkAllocateDescriptorSets;
        vkFreeDescriptorSets = other.vkFreeDescriptorSets;
        vkUpdateDescriptorSets = other.vkUpdateDescriptorSets;

        vkCreateCommandPool = other.vkCreateCommandPool;
        vkDestroyCommandPool = other.vkDestroyCommandPool;
        vkAllocateCommandBuffers = other.vkAllocateCommandBuffers;
        vkFreeCommandBuffers = other.vkFreeCommandBuffers;
        vkBeginCommandBuffer = other.vkBeginCommandBuffer;
        vkEndCommandBuffer = other.vkEndCommandBuffer;
        vkResetCommandBuffer = other.vkResetCommandBuffer;

        vkCmdBindPipeline = other.vkCmdBindPipeline;
        vkCmdBindDescriptorSets = other.vkCmdBindDescriptorSets;
        vkCmdPushConstants = other.vkCmdPushConstants;
        vkCmdDispatch = other.vkCmdDispatch;
        vkCmdPipelineBarrier = other.vkCmdPipelineBarrier;
        vkCmdCopyBuffer = other.vkCmdCopyBuffer;

        vkCreateFence = other.vkCreateFence;
        vkDestroyFence = other.vkDestroyFence;
        vkWaitForFences = other.vkWaitForFences;
        vkResetFences = other.vkResetFences;
    }
    return *this;
}

void DynamicLoader::unload() {
    if (library_handle_) {
        free_shared_library(library_handle_);
        library_handle_ = nullptr;
    }
    vkGetInstanceProcAddr = nullptr;
    vkCreateInstance = nullptr;
}

void DynamicLoader::configure_swiftshader_fallback() {
    std::vector<std::string> candidates = {
        "E:\\Program Filess\\Antigravity\\vk_swiftshader.dll",
        "C:\\Program Files\\Google\\Chrome\\Application\\153.0.8010.53\\vk_swiftshader.dll",
        "vk_swiftshader.dll",
        "libvk_swiftshader.so"
    };

    std::string found_path;
    for (const auto& candidate : candidates) {
        FILE* test_f = std::fopen(candidate.c_str(), "rb");
        if (test_f) {
            std::fclose(test_f);
            found_path = candidate;
            break;
        }
    }

    if (found_path.empty()) {
        SOAR_LOG_WARN("Vulkan fallback: vk_swiftshader not found in known search paths.");
        return;
    }

    std::string temp_dir = ".";
#if defined(_WIN32)
    char tmp_buf[MAX_PATH];
    DWORD len = GetTempPathA(MAX_PATH, tmp_buf);
    if (len > 0 && len < MAX_PATH) {
        temp_dir = tmp_buf;
    }
#else
    const char* t = std::getenv("TMPDIR");
    if (!t) t = std::getenv("TMP");
    if (!t) t = "/tmp";
    temp_dir = t;
#endif
    if (!temp_dir.empty() && (temp_dir.back() == '/' || temp_dir.back() == '\\')) {
        temp_dir.pop_back();
    }
    std::string icd_json_path = temp_dir + "/soar_swiftshader_icd.json";

    FILE* out = std::fopen(icd_json_path.c_str(), "w");
    if (out) {
        std::string json_escaped_path;
        for (char c : found_path) {
            if (c == '\\') {
                json_escaped_path += "\\\\";
            } else {
                json_escaped_path += c;
            }
        }
        std::fprintf(out, "{\n  \"file_format_version\": \"1.0.0\",\n  \"ICD\": {\n    \"library_path\": \"%s\",\n    \"api_version\": \"1.3.0\"\n  }\n}\n", json_escaped_path.c_str());
        std::fclose(out);

        set_env_variable("VK_ICD_FILENAMES", icd_json_path);
        set_env_variable("VK_DRIVER_FILES", icd_json_path);
        SOAR_LOG_INFO("Configured Vulkan SwiftShader ICD fallback: {}", icd_json_path);
    }
}

void DynamicLoader::load_library(const std::string& custom_lib_path) {
    if (library_handle_) {
        unload();
    }

    std::vector<std::string> search_names;
    if (!custom_lib_path.empty()) {
        search_names.push_back(custom_lib_path);
    }
#if defined(_WIN32)
    search_names.push_back("vulkan-1.dll");
    search_names.push_back("E:\\Program Filess\\Antigravity\\vulkan-1.dll");
    search_names.push_back("C:\\Windows\\System32\\vulkan-1.dll");
#elif defined(__APPLE__)
    search_names.push_back("libvulkan.dylib");
    search_names.push_back("libvulkan.1.dylib");
    search_names.push_back("libMoltenVK.dylib");
#else
    search_names.push_back("libvulkan.so.1");
    search_names.push_back("libvulkan.so");
#endif

    for (const auto& name : search_names) {
        library_handle_ = load_shared_library(name);
        if (library_handle_) {
            SOAR_LOG_INFO("Loaded Vulkan shared library: {}", name);
            break;
        }
    }

    if (!library_handle_) {
        throw DeviceError("Failed to dynamically load Vulkan runtime library (vulkan-1.dll / libvulkan.so.1)");
    }

    vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        get_proc_symbol(library_handle_, "vkGetInstanceProcAddr")
    );

    if (!vkGetInstanceProcAddr) {
        unload();
        throw DeviceError("Vulkan shared library missing vkGetInstanceProcAddr symbol");
    }

    // Load global symbols
    vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
        vkGetInstanceProcAddr(nullptr, "vkCreateInstance")
    );
    vkEnumerateInstanceExtensionProperties = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceExtensionProperties")
    );
    vkEnumerateInstanceLayerProperties = reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceLayerProperties")
    );
    vkEnumerateInstanceVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion")
    );
}

void DynamicLoader::load_instance_functions(VkInstance instance) {
    if (!instance || !vkGetInstanceProcAddr) {
        throw DeviceError("Cannot load instance functions: invalid VkInstance or loader");
    }

    #define LOAD_INST(fn) fn = reinterpret_cast<PFN_##fn>(vkGetInstanceProcAddr(instance, #fn))
    LOAD_INST(vkDestroyInstance);
    LOAD_INST(vkEnumeratePhysicalDevices);
    LOAD_INST(vkGetPhysicalDeviceProperties);
    LOAD_INST(vkGetPhysicalDeviceFeatures);
    LOAD_INST(vkGetPhysicalDeviceMemoryProperties);
    LOAD_INST(vkGetPhysicalDeviceQueueFamilyProperties);
    LOAD_INST(vkCreateDevice);
    LOAD_INST(vkGetDeviceProcAddr);
    LOAD_INST(vkEnumerateDeviceExtensionProperties);
    #undef LOAD_INST
}

void DynamicLoader::load_device_functions(VkDevice device) {
    if (!device) {
        throw DeviceError("Cannot load device functions: invalid VkDevice");
    }

    auto get_dev_proc = [this, device](const char* name) -> PFN_vkVoidFunction {
        if (vkGetDeviceProcAddr) {
            PFN_vkVoidFunction fn = vkGetDeviceProcAddr(device, name);
            if (fn) return fn;
        }
        return vkGetInstanceProcAddr(nullptr, name);
    };

    #define LOAD_DEV(fn) fn = reinterpret_cast<PFN_##fn>(get_dev_proc(#fn))
    LOAD_DEV(vkDestroyDevice);
    LOAD_DEV(vkGetDeviceQueue);
    LOAD_DEV(vkQueueSubmit);
    LOAD_DEV(vkQueueWaitIdle);
    LOAD_DEV(vkDeviceWaitIdle);

    LOAD_DEV(vkCreateBuffer);
    LOAD_DEV(vkDestroyBuffer);
    LOAD_DEV(vkGetBufferMemoryRequirements);
    LOAD_DEV(vkAllocateMemory);
    LOAD_DEV(vkFreeMemory);
    LOAD_DEV(vkBindBufferMemory);
    LOAD_DEV(vkMapMemory);
    LOAD_DEV(vkUnmapMemory);
    LOAD_DEV(vkFlushMappedMemoryRanges);
    LOAD_DEV(vkInvalidateMappedMemoryRanges);

    LOAD_DEV(vkCreateShaderModule);
    LOAD_DEV(vkDestroyShaderModule);
    LOAD_DEV(vkCreateDescriptorSetLayout);
    LOAD_DEV(vkDestroyDescriptorSetLayout);
    LOAD_DEV(vkCreatePipelineLayout);
    LOAD_DEV(vkDestroyPipelineLayout);
    LOAD_DEV(vkCreateComputePipelines);
    LOAD_DEV(vkDestroyPipeline);

    LOAD_DEV(vkCreateDescriptorPool);
    LOAD_DEV(vkDestroyDescriptorPool);
    LOAD_DEV(vkAllocateDescriptorSets);
    LOAD_DEV(vkFreeDescriptorSets);
    LOAD_DEV(vkUpdateDescriptorSets);

    LOAD_DEV(vkCreateCommandPool);
    LOAD_DEV(vkDestroyCommandPool);
    LOAD_DEV(vkAllocateCommandBuffers);
    LOAD_DEV(vkFreeCommandBuffers);
    LOAD_DEV(vkBeginCommandBuffer);
    LOAD_DEV(vkEndCommandBuffer);
    LOAD_DEV(vkResetCommandBuffer);

    LOAD_DEV(vkCmdBindPipeline);
    LOAD_DEV(vkCmdBindDescriptorSets);
    LOAD_DEV(vkCmdPushConstants);
    LOAD_DEV(vkCmdDispatch);
    LOAD_DEV(vkCmdPipelineBarrier);
    LOAD_DEV(vkCmdCopyBuffer);

    LOAD_DEV(vkCreateFence);
    LOAD_DEV(vkDestroyFence);
    LOAD_DEV(vkWaitForFences);
    LOAD_DEV(vkResetFences);
    #undef LOAD_DEV
}

} // namespace soar::vk
