#include <soar/nn/soar_model.hpp>
#include <soar/engine/trainer.hpp>
#include <soar/engine/validator.hpp>
#include <soar/engine/predictor.hpp>
#include <soar/losses/losses.hpp>
#include <soar/optim/adamw.hpp>
#include <soar/core/logging.hpp>
#include <soar/core/error.hpp>
#include <soar/core/config_parser.hpp>
#include <soar/data/image_io.hpp>
#include <soar/data/coco_dataset.hpp>
#include <soar/data/yolo_dataset.hpp>
#include <soar/data/dataloader.hpp>
#include <soar/vulkan/context.hpp>
#include <soar/vulkan/buffer.hpp>
#include <soar/vulkan/pipeline.hpp>
#include <soar/vulkan/command_queue.hpp>
#include <soar/shaders/spv_shaders.hpp>
#include <soar/cuda/cuda_runtime.hpp>
#include "palloc.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <numeric>
#include <random>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

static std::string path_join(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/' || a.back() == '\\') return a + b;
    return a + "/" + b;
}

static bool file_exists(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f) {
        std::fclose(f);
        return true;
    }
#if defined(_WIN32)
    DWORD attrs = GetFileAttributesA(path.c_str());
    return (attrs != INVALID_FILE_ATTRIBUTES);
#else
    struct stat st;
    return (stat(path.c_str(), &st) == 0);
#endif
}

static void ensure_directory(const std::string& dir) {
    if (dir.empty()) return;
#if defined(_WIN32)
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
}

void print_usage() {
    std::cout << "=================================================================\n"
              << "  SOAR: High-Performance C++20 Resolution-Preserving Engine    \n"
              << "=================================================================\n"
              << "Usage: soar_engine <command> [options]\n\n"
              << "Commands:\n"
              << "  train       Train model with full epoch lifecycle & validation period\n"
              << "  val         Validate trained checkpoint on dataset (O(1) memory)\n"
              << "  predict     Run inference on single image or directory and export masks/RLE\n"
              << "  test        Evaluate model on test dataset and export predictions\n"
              << "  benchmark   Benchmark latency on high-resolution synthetic image\n"
              << "  help        Display this help message\n\n"
              << "General Options:\n"
              << "  --config, --model <path>      Path to model config YAML (default: configs/models/soar_nano1.yaml)\n"
              << "  --data <path>                 Path to dataset root directory\n"
              << "  --images-dir <path>           Path to images directory\n"
              << "  --annotation-file <path>      Path to COCO JSON annotation file\n"
              << "  --labels-dir <path>           Path to YOLO labels directory\n"
              << "  --data-format <coco|yolo>     Dataset format (default: coco)\n"
              << "  --weights <path>              Path to model weights (.soar)\n"
              << "  --device <auto|gpu|cpu>       Hardware acceleration device (default: auto)\n"
              << "  --img-size <H> [W]            Target image resolution (default: 1024 1024)\n"
              << "  --threshold <float>           Sigmoid classification threshold (default: 0.5)\n\n"
              << "Options for 'train':\n"
              << "  --epochs <N>                  Total training epochs (default: 50)\n"
              << "  --lr <float>                  Initial learning rate (default: 1e-4)\n"
              << "  --weight-decay <float>        Weight decay (default: 1e-5)\n"
              << "  --accumulate-grad-batches <K> Virtual gradient accumulation steps (default: 1)\n"
              << "  --val-split <float>           Validation split ratio (default: 0.1)\n"
              << "  --grad-clip <float>           Gradient clipping max norm (default: 2.0)\n"
              << "  --pos-weight <float>          BCE positive class weight for imbalance (default: 1.0)\n"
              << "  --bce-weight <float>          BCE loss component weight (default: 1.0)\n"
              << "  --dice-weight <float>         Dice loss component weight (default: 1.0)\n"
              << "  --dice-smooth <float>         Dice loss smoothing epsilon (default: 1.0)\n"
              << "  --save-interval <N>           Checkpoint saving interval in epochs (default: 5)\n"
              << "  --checkpoint-dir <path>       Directory to save checkpoints (default: checkpoints)\n"
              << "  --output <path>               Final checkpoint path (default: checkpoints/best.soar)\n\n"
              << "Options for 'val' & 'test':\n"
              << "  --save-dir <path>             Directory to save qualitative comparison visualizations\n\n"
              << "Options for 'predict':\n"
              << "  --input <path>                Input image file path or directory\n"
              << "  --output <path>               Output mask BMP file path or RLE CSV path\n"
              << "  --output-dir <path>           Directory for batch mask predictions (default: predictions)\n"
              << "  --output-format <image|rle>   Prediction output format (default: image)\n"
              << std::endl;
}

static bool is_img_ext(const std::string& ext) {
    return (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
            ext == ".tiff" || ext == ".tif" || ext == ".npy" || ext == ".fits");
}

static bool dir_contains_images(const std::string& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return false;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
                if (is_img_ext(ext)) return true;
            }
        }
    } catch (...) {}
    return false;
}

static bool dir_contains_yolo_labels(const std::string& dir) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) return false;
    try {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
                if (ext == ".txt") return true;
            }
        }
    } catch (...) {}
    return false;
}

// Resolve dataset directory helpers
void auto_detect_dataset_paths(const std::string& data_root,
                               std::string& images_dir,
                               std::string& annotation_file,
                               std::string& labels_dir,
                               std::string& data_format) {
    if (data_root.empty() && images_dir.empty()) return;

    // 1. Resolve images_dir
    if (images_dir.empty() && !data_root.empty() && fs::exists(data_root)) {
        std::vector<std::string> candidate_img_dirs = {
            path_join(path_join(data_root, "train"), "train_images"),
            path_join(data_root, "train_images"),
            path_join(path_join(data_root, "train"), "images"),
            path_join(data_root, "images"),
            path_join(data_root, "train"),
            data_root
        };
        for (const auto& cand : candidate_img_dirs) {
            if (dir_contains_images(cand)) {
                images_dir = cand;
                break;
            }
        }
        // If not found in primary candidates, recursive search (up to 3 levels deep)
        if (images_dir.empty() && fs::is_directory(data_root)) {
            try {
                for (const auto& entry : fs::recursive_directory_iterator(data_root, fs::directory_options::skip_permission_denied)) {
                    if (entry.is_directory() && dir_contains_images(entry.path().generic_string())) {
                        images_dir = entry.path().generic_string();
                        break;
                    }
                }
            } catch (...) {}
        }
    }

    // 2. Resolve annotation_file (COCO)
    if (annotation_file.empty()) {
        std::vector<std::string> search_roots;
        if (!data_root.empty() && fs::exists(data_root)) search_roots.push_back(data_root);
        if (!images_dir.empty() && fs::exists(images_dir)) {
            fs::path p(images_dir);
            if (p.has_parent_path()) {
                search_roots.push_back(p.parent_path().generic_string());
                if (p.parent_path().has_parent_path()) {
                    search_roots.push_back(p.parent_path().parent_path().generic_string());
                }
            }
        }

        // Check specific candidate filenames first
        for (const auto& root : search_roots) {
            std::vector<std::string> specific_cand_files = {
                path_join(path_join(root, "train"), "MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
                path_join(root, "MAGFiLO_1.0_Annotations_kaggle2026_train.json"),
                path_join(path_join(root, "train"), "annotations.json"),
                path_join(root, "annotations.json"),
                path_join(root, "train.json")
            };
            for (const auto& f : specific_cand_files) {
                if (file_exists(f)) {
                    annotation_file = f;
                    data_format = "coco";
                    break;
                }
            }
            if (!annotation_file.empty()) break;
        }

        // If not found, search candidate directories for any *.json
        if (annotation_file.empty()) {
            for (const auto& root : search_roots) {
                std::vector<std::string> search_dirs = {
                    path_join(root, "train"),
                    path_join(root, "annotations"),
                    root
                };
                for (const auto& sdir : search_dirs) {
                    if (fs::exists(sdir) && fs::is_directory(sdir)) {
                        for (const auto& entry : fs::directory_iterator(sdir)) {
                            if (entry.is_regular_file()) {
                                std::string ext = entry.path().extension().string();
                                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
                                if (ext == ".json") {
                                    annotation_file = entry.path().generic_string();
                                    data_format = "coco";
                                    break;
                                }
                            }
                        }
                    }
                    if (!annotation_file.empty()) break;
                }
                if (!annotation_file.empty()) break;
            }
        }
    }

    // 3. Resolve labels_dir (YOLO)
    if (labels_dir.empty() && !data_root.empty() && fs::exists(data_root)) {
        std::vector<std::string> candidate_label_dirs = {
            path_join(path_join(data_root, "train"), "labels"),
            path_join(data_root, "labels"),
            path_join(path_join(data_root, "train"), "train_labels"),
            path_join(data_root, "train_labels")
        };
        for (const auto& cand : candidate_label_dirs) {
            if (dir_contains_yolo_labels(cand)) {
                labels_dir = cand;
                if (annotation_file.empty()) data_format = "yolo";
                break;
            }
        }
    }

    if (!images_dir.empty() || !annotation_file.empty()) {
        std::cout << "[SOAR Engine] Dataset auto-detection result:\n"
                  << "  Images Directory: " << (images_dir.empty() ? "(none)" : images_dir) << "\n"
                  << "  Annotations:      " << (annotation_file.empty() ? "(none)" : annotation_file) << "\n"
                  << "  Labels Directory: " << (labels_dir.empty() ? "(none)" : labels_dir) << "\n"
                  << "  Data Format:      " << data_format << std::endl;
    }
}

static bool is_terminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout));
#else
    return isatty(fileno(stdout));
#endif
}

static void render_progress_bar(const std::string& prefix, size_t current, size_t total,
                                double elapsed_sec, float loss, float dice, float iou, float lr) {
    if (total == 0) return;
    float pct = static_cast<float>(current) / static_cast<float>(total);
    int bar_width = 15;
    int filled = static_cast<int>(std::round(pct * bar_width));
    filled = std::max(0, std::min(bar_width, filled));

    std::string bar;
    bar.reserve(bar_width);
    for (int i = 0; i < filled; ++i) bar += "=";
    if (filled < bar_width) bar += ">";
    while (bar.size() < static_cast<size_t>(bar_width)) bar += " ";

    double it_per_sec = (elapsed_sec > 0.0) ? (static_cast<double>(current) / elapsed_sec) : 0.0;
    double sec_per_it = (current > 0) ? (elapsed_sec / static_cast<double>(current)) : 0.0;
    double eta_sec = (it_per_sec > 0.0) ? (static_cast<double>(total - current) / it_per_sec) : 0.0;

    int el_m = static_cast<int>(elapsed_sec) / 60;
    int el_s = static_cast<int>(elapsed_sec) % 60;
    int eta_m = static_cast<int>(eta_sec) / 60;
    int eta_s = static_cast<int>(eta_sec) % 60;

    if (!is_terminal()) {
        std::cout << prefix << " " << std::setw(3) << static_cast<int>(pct * 100.0f) << "%|"
                  << bar << "| " << current << "/" << total
                  << " [" << std::setfill('0') << std::setw(2) << el_m << ":" << std::setw(2) << el_s
                  << "<" << std::setw(2) << eta_m << ":" << std::setw(2) << eta_s;
        if (it_per_sec >= 1.0) {
            std::cout << ", " << std::setfill(' ') << std::fixed << std::setprecision(1) << it_per_sec << "it/s";
        } else {
            std::cout << ", " << std::setfill(' ') << std::fixed << std::setprecision(1) << sec_per_it << "s/it";
        }
        std::cout << ", loss: " << std::setprecision(4) << loss
                  << ", dice: " << std::setprecision(4) << dice
                  << ", lr: " << std::scientific << std::setprecision(1) << lr << std::defaultfloat
                  << "]\n" << std::flush;
        return;
    }

    std::cout << "\r" << prefix << " " << std::setw(3) << static_cast<int>(pct * 100.0f) << "%|"
              << bar << "| " << current << "/" << total
              << " [" << std::setfill('0') << std::setw(2) << el_m << ":" << std::setw(2) << el_s
              << "<" << std::setw(2) << eta_m << ":" << std::setw(2) << eta_s;
    if (it_per_sec >= 1.0) {
        std::cout << ", " << std::setfill(' ') << std::fixed << std::setprecision(1) << it_per_sec << "it/s";
    } else {
        std::cout << ", " << std::setfill(' ') << std::fixed << std::setprecision(1) << sec_per_it << "s/it";
    }
    std::cout << ", loss: " << std::setprecision(4) << loss
              << ", dice: " << std::setprecision(4) << dice
              << ", lr: " << std::scientific << std::setprecision(1) << lr << std::defaultfloat
              << "]   " << std::flush;
}

static void run_vulkan_hardware_telemetry_and_sanity_probe(soar::vk::VulkanContext& vk_ctx) {
    const auto& info = vk_ctx.device_info();

    std::string dev_type_str = "Unknown";
    switch (info.device_type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: dev_type_str = "Discrete GPU"; break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: dev_type_str = "Integrated GPU"; break;
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: dev_type_str = "Virtual GPU"; break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU: dev_type_str = "Software / CPU Emulator"; break;
        default: dev_type_str = "Other / Accelerator"; break;
    }

    double vram_mb = static_cast<double>(info.total_device_memory) / (1024.0 * 1024.0);
    double vram_gb = vram_mb / 1024.0;

    std::cout << "\n=================================================================\n"
              << "  SOAR Universal GPU Hardware Telemetry & Sanity Probe           \n"
              << "=================================================================\n"
              << "  Physical Device Name:  " << info.device_name << "\n"
              << "  Vendor ID:             0x" << std::hex << info.vendor_id << std::dec << "\n"
              << "  Device Type:           " << dev_type_str << "\n"
              << "  API Version:           " << VK_VERSION_MAJOR(info.api_version) << "."
                                         << VK_VERSION_MINOR(info.api_version) << "."
                                         << VK_VERSION_PATCH(info.api_version) << "\n"
              << "  Dedicated VRAM:        " << std::fixed << std::setprecision(1) << vram_mb << " MB";
    if (vram_gb >= 1.0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << vram_gb << " GB)";
    }
    std::cout << "\n"
              << "  Compute Queue Family:  " << vk_ctx.compute_queue_family_index() << "\n"
              << "-----------------------------------------------------------------\n"
              << "  Executing 1MB DEVICE_LOCAL Compute Shader Sanity Probe...\n";

    constexpr size_t PROBE_ELEMENTS = 256 * 1024; // 256K floats = 1 MB
    constexpr size_t PROBE_BYTES = PROBE_ELEMENTS * sizeof(float);

    std::vector<float> host_in(PROBE_ELEMENTS);
    for (size_t i = 0; i < PROBE_ELEMENTS; ++i) {
        host_in[i] = static_cast<float>(i % 1000) * 0.005f - 2.5f;
    }

    // Allocate host staging buffers and device local storage buffers
    soar::vk::VulkanBuffer staging_in(vk_ctx, PROBE_BYTES, soar::vk::BufferUsageType::StagingHost);
    soar::vk::VulkanBuffer dev_in(vk_ctx, PROBE_BYTES, soar::vk::BufferUsageType::DeviceStorage);
    soar::vk::VulkanBuffer dev_out(vk_ctx, PROBE_BYTES, soar::vk::BufferUsageType::DeviceStorage);
    soar::vk::VulkanBuffer staging_out(vk_ctx, PROBE_BYTES, soar::vk::BufferUsageType::StagingHost);

    staging_in.upload_host(host_in.data(), PROBE_BYTES);

    struct SigmoidPC {
        uint32_t total_elements;
    } pc{ static_cast<uint32_t>(PROBE_ELEMENTS) };

    soar::vk::ComputePipeline pipeline(vk_ctx, soar::shaders::get_sigmoid(), 2, sizeof(SigmoidPC));
    const soar::vk::VulkanBuffer* bufs[] = { &dev_in, &dev_out };
    pipeline.bind_buffers(bufs);

    soar::vk::CommandQueue queue(vk_ctx);

    auto start_time = std::chrono::high_resolution_clock::now();

    queue.execute_sync([&](VkCommandBuffer cmd) {
        // Upload staging -> device local
        dev_in.copy_from(cmd, staging_in, PROBE_BYTES);

        // Memory barrier: transfer write -> compute shader read
        soar::vk::CommandQueue::memory_barrier(
            vk_ctx, cmd, dev_in,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT
        );

        // Compute dispatch
        uint32_t group_x = (static_cast<uint32_t>(PROBE_ELEMENTS) + 255) / 256;
        pipeline.record_dispatch(cmd, group_x, 1, 1, &pc, sizeof(pc));

        // Memory barrier: compute shader write -> transfer read
        soar::vk::CommandQueue::memory_barrier(
            vk_ctx, cmd, dev_out,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_TRANSFER_READ_BIT
        );

        // Download device local -> staging
        staging_out.copy_from(cmd, dev_out, PROBE_BYTES);
    });

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::vector<float> host_out(PROBE_ELEMENTS, 0.0f);
    staging_out.download_host(host_out.data(), PROBE_BYTES);

    float max_delta = 0.0f;
    for (size_t i = 0; i < PROBE_ELEMENTS; ++i) {
        float expected = 1.0f / (1.0f + std::exp(-host_in[i]));
        float delta = std::fabs(host_out[i] - expected);
        if (delta > max_delta) max_delta = delta;
    }

    if (max_delta > 1e-4f) {
        throw soar::DeviceError("Hardware compute probe numerical sanity check failed! Max delta: " + std::to_string(max_delta));
    }

    std::cout << "  -> Hardware Sanity Probe PASSED!\n"
              << "     Transfer + Compute Latency: " << duration_us << " us\n"
              << "     Numerical Precision Delta:  " << std::scientific << std::setprecision(3) << max_delta << std::defaultfloat << "\n"
              << "=================================================================\n\n";
}

int main(int argc, char* argv[]) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;

    if (argc < 2) {
        print_usage();
        return 0;
    }

    std::string cmd = argv[1];
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_usage();
        return 0;
    }

    std::string config_path = "configs/models/soar_nano1.yaml";
    std::string data_root = "";
    std::string images_dir = "";
    std::string annotation_file = "";
    std::string labels_dir = "";
    std::string data_format = "coco";
    std::string weights_path = "";
    std::string input_path = "";
    std::string output_path = "";
    std::string output_dir = "predictions";
    std::string save_dir = "";
    std::string output_format = "image";
    std::string checkpoint_dir = "checkpoints";

    size_t epochs = 50;
    float learning_rate = 1e-4f;
    float weight_decay = 1e-5f;
    size_t accumulate_grad_batches = 1;
    float val_split = 0.1f;
    float grad_clip = 2.0f;
    float pos_weight = 1.0f;
    float bce_weight = 1.0f;
    float dice_weight = 1.0f;
    float dice_smooth = 1.0f;
    size_t save_interval = 5;
    float threshold = 0.5f;
    size_t benchmark_h = 512;
    size_t benchmark_w = 512;
    std::string device_str = "auto";
    size_t batch_size = 1;
    size_t img_h = 1024;
    size_t img_w = 1024;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "--model") && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--data" && i + 1 < argc) data_root = argv[++i];
        else if (arg == "--images-dir" && i + 1 < argc) images_dir = argv[++i];
        else if (arg == "--annotation-file" && i + 1 < argc) annotation_file = argv[++i];
        else if (arg == "--labels-dir" && i + 1 < argc) labels_dir = argv[++i];
        else if (arg == "--data-format" && i + 1 < argc) data_format = argv[++i];
        else if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
        else if ((arg == "--batch-size" || arg == "-b") && i + 1 < argc) batch_size = std::stoul(argv[++i]);
        else if (arg == "--img-size" && i + 1 < argc) {
            img_h = std::stoul(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                img_w = std::stoul(argv[++i]);
            } else {
                img_w = img_h;
            }
        }
        else if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (arg == "--save-dir" && i + 1 < argc) save_dir = argv[++i];
        else if (arg == "--output-format" && i + 1 < argc) output_format = argv[++i];
        else if (arg == "--checkpoint-dir" && i + 1 < argc) checkpoint_dir = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoul(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) learning_rate = std::stof(argv[++i]);
        else if (arg == "--weight-decay" && i + 1 < argc) weight_decay = std::stof(argv[++i]);
        else if (arg == "--accumulate-grad-batches" && i + 1 < argc) accumulate_grad_batches = std::stoul(argv[++i]);
        else if (arg == "--val-split" && i + 1 < argc) val_split = std::stof(argv[++i]);
        else if (arg == "--grad-clip" && i + 1 < argc) grad_clip = std::stof(argv[++i]);
        else if (arg == "--pos-weight" && i + 1 < argc) pos_weight = std::stof(argv[++i]);
        else if (arg == "--bce-weight" && i + 1 < argc) bce_weight = std::stof(argv[++i]);
        else if (arg == "--dice-weight" && i + 1 < argc) dice_weight = std::stof(argv[++i]);
        else if (arg == "--dice-smooth" && i + 1 < argc) dice_smooth = std::stof(argv[++i]);
        else if (arg == "--save-interval" && i + 1 < argc) save_interval = std::stoul(argv[++i]);
        else if (arg == "--threshold" && i + 1 < argc) threshold = std::stof(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) benchmark_h = std::stoul(argv[++i]);
        else if (arg == "--width" && i + 1 < argc) benchmark_w = std::stoul(argv[++i]);
    }

    auto_detect_dataset_paths(data_root, images_dir, annotation_file, labels_dir, data_format);

    // Resolve model config path (fallback between configs/ and cfg/)
    if (!fs::exists(config_path)) {
        if (config_path.find("cfg/") != std::string::npos) {
            std::string alt = config_path;
            alt.replace(alt.find("cfg/"), 4, "configs/");
            if (fs::exists(alt)) config_path = alt;
        } else if (config_path.find("configs/") != std::string::npos) {
            std::string alt = config_path;
            alt.replace(alt.find("configs/"), 8, "cfg/");
            if (fs::exists(alt)) config_path = alt;
        }
    }

    // Load Model Config
    soar::core::ModelConfig m_cfg;
    if (fs::exists(config_path)) {
        m_cfg = soar::core::ConfigParser::load_model_config(config_path);
    } else {
        std::cout << "[WARN] Config file '" << config_path << "' not found, using default Nano configuration." << std::endl;
    }

    [[maybe_unused]] bool use_cuda = false;
    std::unique_ptr<soar::vk::VulkanContext> vk_ctx;

    if (device_str == "cuda") {
        int cuda_count = 0;
        if (soar::cuda::cudaGetDeviceCount(&cuda_count) == soar::cuda::cudaSuccess && cuda_count > 0) {
            std::cout << "[SOAR Engine] Native CUDA device initialized: "
                      << cuda_count << " GPU device(s) active." << std::endl;
            use_cuda = true;
        } else {
            std::cerr << "[FATAL] CUDA requested but no active device detected." << std::endl;
            return 1;
        }
    } else if (device_str == "vulkan" || device_str == "vk" || device_str == "gpu") {
        try {
            vk_ctx = std::make_unique<soar::vk::VulkanContext>(false, /*allow_cpu_fallback=*/false);
            run_vulkan_hardware_telemetry_and_sanity_probe(*vk_ctx);
        } catch (const soar::HardwareNotFoundError& e) {
            std::cerr << "[FATAL] Physical GPU hardware acceleration requested ('" << device_str << "'), "
                      << "but no physical GPU hardware is available: " << e.what() << "\n"
                      << "Execution terminated. Software/CPU emulation fallbacks are strictly prohibited." << std::endl;
            return 1;
        } catch (const std::exception& e) {
            std::cerr << "[FATAL] Physical GPU initialization failed: " << e.what() << std::endl;
            return 1;
        }
    } else if (device_str == "auto") {
        // Attempt physical Vulkan GPU first
        try {
            vk_ctx = std::make_unique<soar::vk::VulkanContext>(false, /*allow_cpu_fallback=*/false);
            run_vulkan_hardware_telemetry_and_sanity_probe(*vk_ctx);
        } catch (const std::exception& e) {
            std::cout << "[SOAR Engine] Auto-detection: No compatible physical Vulkan GPU detected ("
                      << e.what() << ").\n"
                      << "[SOAR Engine] Selecting multi-threaded CPU OpenMP engine ("
                      << std::thread::hardware_concurrency() << " concurrent threads active)." << std::endl;
            vk_ctx.reset();
        }
    } else if (device_str == "cpu") {
        std::cout << "[SOAR Engine] Hardware accelerator: Multi-threaded CPU OpenMP engine ("
                  << std::thread::hardware_concurrency() << " concurrent threads active)." << std::endl;
    } else {
        std::cerr << "[ERROR] Unknown device specified: '" << device_str
                  << "'. Valid options are: auto, gpu, vulkan, cuda, cpu." << std::endl;
        return 1;
    }

    auto model = std::make_shared<soar::nn::SOARModel>(m_cfg.in_channels, m_cfg.num_classes, m_cfg.variant);
    std::cout << "[SOAR Engine] Architecture: " << m_cfg.variant_str
              << " (" << model->parameter_count() << " parameters)" << std::endl;

    if (!weights_path.empty()) {
        std::cout << "[SOAR Engine] Loading checkpoint weights from: " << weights_path << std::endl;
        model->load_weights(weights_path);
    }

    if (vk_ctx) {
        model->to_device(*vk_ctx);
    }

    // --------------------------------------------------------------------------------------------------
    // BENCHMARK COMMAND
    // --------------------------------------------------------------------------------------------------
    if (cmd == "benchmark") {
        size_t H = benchmark_h;
        size_t W = benchmark_w;
        std::cout << "[SOAR Engine] Benchmarking " << H << "x" << W << " native full-resolution inference..." << std::endl;
        auto input = soar::Tensor::randn({static_cast<int64_t>(m_cfg.in_channels), static_cast<int64_t>(H), static_cast<int64_t>(W)}, 0.5f, 0.2f);
        soar::engine::Predictor predictor(model, threshold);

        for (size_t s = 0; s < 3; ++s) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = predictor.predict(input);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  Pass " << (s + 1) << "/3: Latency = " << std::fixed << std::setprecision(2)
                      << ms << " ms (" << H << "x" << W << ")" << std::endl;
        }
    }
    // --------------------------------------------------------------------------------------------------
    // TRAIN COMMAND (Full Training Period & Validation Lifecycle)
    // --------------------------------------------------------------------------------------------------
    else if (cmd == "train") {
        std::cout << "[SOAR Engine] Initializing Training Pipeline..." << std::endl;
        std::cout << "  Config:                  " << config_path << "\n"
                  << "  Epochs:                  " << epochs << "\n"
                  << "  Learning Rate:           " << learning_rate << "\n"
                  << "  Weight Decay:            " << weight_decay << "\n"
                  << "  Physical Batch Size:     1 (Strict Native Resolution)\n"
                  << "  Image Resolution:        " << img_h << "x" << img_w << "\n"
                  << "  Accumulate Grad Batches: " << accumulate_grad_batches << "\n"
                  << "  Validation Split:        " << val_split << "\n"
                  << "  Loss Composition:        BCE(pw=" << pos_weight << ", w=" << bce_weight
                  << ") + Dice(w=" << dice_weight << ", smooth=" << dice_smooth << ")\n"
                  << "  Checkpoint Directory:    " << checkpoint_dir << std::endl;

        ensure_directory(checkpoint_dir);
        std::string val_vis_base = path_join(checkpoint_dir, "val_visualizations");
        ensure_directory(val_vis_base);

        soar::losses::CompositeLoss loss_fn(bce_weight, dice_weight, pos_weight, dice_smooth);
        soar::optim::AdamWOptions opt_opts;
        opt_opts.lr = learning_rate;
        opt_opts.weight_decay = weight_decay;
        soar::optim::AdamW optimizer(model->parameters(), opt_opts);

        // Learning rate scheduler stepped per EPOCH (the correct training period)
        auto scheduler = std::make_unique<soar::optim::CosineAnnealingLR>(
            optimizer, epochs, 1e-6f, std::min(size_t(3), epochs / 10));

        soar::engine::Trainer trainer(model, loss_fn, std::move(optimizer), std::move(scheduler),
                                      accumulate_grad_batches, grad_clip);
        soar::engine::Validator validator(model, loss_fn, threshold);

        float best_val_loss = std::numeric_limits<float>::infinity();
        float best_val_dice = 0.0f;

        // Lambda to execute training period on dataset
        auto run_dataset_training = [&](const auto& ds) {
            size_t total_samples = ds.size();
            if (total_samples == 0) {
                std::cerr << "[ERROR] Dataset is empty!" << std::endl;
                return;
            }

            size_t val_count = static_cast<size_t>(std::round(static_cast<float>(total_samples) * val_split));
            if (val_split > 0.0f && val_count == 0 && total_samples > 1) val_count = 1;
            size_t train_count = total_samples - val_count;

            std::vector<size_t> indices(total_samples);
            std::iota(indices.begin(), indices.end(), 0);
            std::mt19937 g(42);
            std::shuffle(indices.begin(), indices.end(), g);

            std::vector<size_t> train_indices(indices.begin(), indices.begin() + train_count);
            std::vector<size_t> val_indices(indices.begin() + train_count, indices.end());

            size_t n_workers = std::thread::hardware_concurrency() > 0 ? std::min(size_t(8), (size_t)std::thread::hardware_concurrency()) : 4;
            soar::data::DataLoaderOptions loader_opts;
            loader_opts.batch_size = batch_size;
            loader_opts.workers = n_workers;
            loader_opts.prefetch_factor = 4;
            loader_opts.shuffle = true;
            loader_opts.pin_memory = true;

            soar::data::DataLoader train_loader(ds, loader_opts, train_indices);
            size_t total_train_batches = train_loader.total_batches();

            std::cout << "[SOAR Engine] Dataset loaded: " << total_samples << " total ("
                      << train_count << " train, " << val_count << " val) | Physical batch size: " << batch_size
                      << " | Asynchronous DataLoader workers: " << n_workers << std::endl;

            for (size_t ep = 0; ep < epochs; ++ep) {
                auto t0_ep = std::chrono::high_resolution_clock::now();

                // 1. Training Phase (Asynchronous Multi-Threaded Prefetched Pipeline)
                double ep_train_loss = 0.0;
                double ep_train_dice = 0.0;
                double ep_train_iou = 0.0;
                size_t step_idx = 0;

                std::string ep_prefix = "Epoch " + std::to_string(ep + 1) + "/" + std::to_string(epochs) + ":";

                for (const auto& batch : train_loader) {
                    bool is_accumulating = ((step_idx + 1) % accumulate_grad_batches != 0) && ((step_idx + 1) != total_train_batches);
                    auto m = trainer.train_step(batch.data, batch.target, is_accumulating);

                    ep_train_loss += m.loss;
                    ep_train_dice += m.dice_score;
                    ep_train_iou += m.iou_score;
                    step_idx++;

                    auto t_curr = std::chrono::high_resolution_clock::now();
                    double elapsed_curr = std::chrono::duration<double>(t_curr - t0_ep).count();
                    float curr_avg_dice = static_cast<float>(ep_train_dice / static_cast<double>(step_idx));
                    float curr_avg_iou = static_cast<float>(ep_train_iou / static_cast<double>(step_idx));

                    render_progress_bar(ep_prefix, step_idx, total_train_batches, elapsed_curr,
                                        m.loss, curr_avg_dice, curr_avg_iou, m.lr);

                    if (!is_accumulating) {
                        ::pa_collect(false);
                    }
                }
                std::cout << std::endl;

                float avg_train_loss = static_cast<float>(ep_train_loss / static_cast<double>(train_count));
                float avg_train_dice = static_cast<float>(ep_train_dice / static_cast<double>(train_count));
                float avg_train_iou = static_cast<float>(ep_train_iou / static_cast<double>(train_count));

                // 2. Validation Phase (Evaluation Period)
                std::string ep_vis_dir = path_join(val_vis_base, "epoch_" + std::to_string(ep + 1));
                soar::engine::ValidationMetrics v_metrics{};
                if (val_count > 0) {
                    v_metrics = validator.validate(ds, val_indices, ep_vis_dir, 4);
                } else {
                    v_metrics.loss = avg_train_loss;
                    v_metrics.dice = avg_train_dice;
                    v_metrics.iou = avg_train_iou;
                    v_metrics.samples = train_count;
                }

                // 3. Learning Rate Scheduler Period (Stepped once per epoch)
                trainer.step_scheduler();
                float current_lr = trainer.optimizer().get_lr();

                auto t1_ep = std::chrono::high_resolution_clock::now();
                double ep_time_sec = std::chrono::duration<double>(t1_ep - t0_ep).count();

                // 4. Progress Reporting
                std::cout << "Epoch " << std::setw(3) << (ep + 1) << "/" << epochs
                          << " [" << std::fixed << std::setprecision(1) << ep_time_sec << "s] "
                          << "| Train Loss: " << std::setprecision(4) << avg_train_loss
                          << " | Val Loss: " << v_metrics.loss
                          << " | IoU: " << v_metrics.iou
                          << " | Dice: " << v_metrics.dice
                          << " | LR: " << std::scientific << std::setprecision(2) << current_lr
                          << std::defaultfloat << std::endl;

                // 5. Periodic Tabular Summary every epoch (matching Python segres)
                soar::engine::Validator::print_results(v_metrics, ep + 1, epochs, "Validation");

                // 6. Checkpoint Management
                if (v_metrics.loss < best_val_loss) {
                    best_val_loss = v_metrics.loss;
                    best_val_dice = v_metrics.dice;
                    std::string best_path = path_join(checkpoint_dir, "best.soar");
                    model->save_weights(best_path);
                    std::cout << "  [*] Saved new best checkpoint (loss: " << best_val_loss << ") -> " << best_path << std::endl;
                }

                std::string last_path = path_join(checkpoint_dir, "last.soar");
                model->save_weights(last_path);

                if ((ep + 1) % save_interval == 0) {
                    std::string interval_path = path_join(checkpoint_dir, "epoch_" + std::to_string(ep + 1) + ".soar");
                    model->save_weights(interval_path);
                }

                ::pa_collect(true);
            }

            std::cout << "\n[SOAR Engine] Training completed! Best Val Loss: " << best_val_loss
                      << " | Best Val Dice: " << best_val_dice << std::endl;
        };

        if (!images_dir.empty() && !annotation_file.empty()) {
            std::cout << "[SOAR Engine] Loading COCO Dataset from: " << images_dir
                      << " (Resolution: " << img_h << "x" << img_w << ")" << std::endl;
            soar::data::COCODataset ds(images_dir, annotation_file, static_cast<int>(m_cfg.in_channels), img_h, img_w);
            run_dataset_training(ds);
        } else if (!images_dir.empty() && !labels_dir.empty()) {
            std::cout << "[SOAR Engine] Loading YOLO Dataset from: " << images_dir
                      << " (Resolution: " << img_h << "x" << img_w << ")" << std::endl;
            soar::data::YOLODataset ds(images_dir, labels_dir, static_cast<int>(m_cfg.in_channels), img_h, img_w);
            run_dataset_training(ds);
        } else {
            std::cout << "[SOAR Engine] No dataset path provided. Running synthetic training cycle..." << std::endl;
            constexpr size_t H = 64;
            constexpr size_t W = 64;
            auto img = soar::Tensor::randn({static_cast<int64_t>(m_cfg.in_channels), H, W}, 0.5f, 0.2f);
            auto msk = soar::Tensor::zeros({1, H, W});
            for (size_t i = 0; i < msk->numel(); i += 2) msk->data()[i] = 1.0f;

            for (size_t ep = 0; ep < epochs; ++ep) {
                auto m = trainer.train_step(img, msk, false);
                trainer.step_scheduler();
                std::cout << "Epoch " << (ep + 1) << "/" << epochs
                          << ": Loss = " << m.loss
                          << " | Dice = " << m.dice_score
                          << " | IoU = " << m.iou_score << std::endl;
            }
            std::string out_w = output_path.empty() ? path_join(checkpoint_dir, "checkpoint.soar") : output_path;
            model->save_weights(out_w);
        }
    }
    // --------------------------------------------------------------------------------------------------
    // VAL COMMAND (Dedicated Validation Pipeline)
    // --------------------------------------------------------------------------------------------------
    else if (cmd == "val") {
        std::cout << "[SOAR Engine] Starting Standalone Validation Pipeline..." << std::endl;
        soar::losses::CompositeLoss loss_fn(bce_weight, dice_weight, pos_weight, dice_smooth);
        soar::engine::Validator validator(model, loss_fn, threshold);

        soar::engine::ValidationMetrics v_metrics{};

        if (!images_dir.empty() && !annotation_file.empty()) {
            soar::data::COCODataset ds(images_dir, annotation_file, static_cast<int>(m_cfg.in_channels), img_h, img_w);
            std::cout << "[SOAR Engine] Loaded COCO Validation Dataset: " << ds.size() << " samples ("
                      << img_h << "x" << img_w << ")." << std::endl;
            v_metrics = validator.validate(ds, {}, save_dir, 10);
        } else if (!images_dir.empty() && !labels_dir.empty()) {
            soar::data::YOLODataset ds(images_dir, labels_dir, static_cast<int>(m_cfg.in_channels), img_h, img_w);
            std::cout << "[SOAR Engine] Loaded YOLO Validation Dataset: " << ds.size() << " samples ("
                      << img_h << "x" << img_w << ")." << std::endl;
            v_metrics = validator.validate(ds, {}, save_dir, 10);
        } else {
            std::cerr << "[ERROR] Validation requires a dataset (--images-dir and --annotation-file or --labels-dir)." << std::endl;
            return 1;
        }

        soar::engine::Validator::print_results(v_metrics, 0, 0, "Validation");
        if (!save_dir.empty()) {
            std::cout << "[SOAR Engine] Visual validation comparisons saved to: " << save_dir << std::endl;
        }
        ::pa_collect(true);
    }
    // --------------------------------------------------------------------------------------------------
    // PREDICT COMMAND (Inference & RLE / BMP Export)
    // --------------------------------------------------------------------------------------------------
    else if (cmd == "predict") {
        if (input_path.empty()) {
            std::cerr << "[ERROR] --input <path> (file or directory) is required for prediction." << std::endl;
            return 1;
        }

        soar::engine::Predictor predictor(model, threshold);

        // Check if single image or directory
        if (fs::is_directory(input_path)) {
            std::cout << "[SOAR Engine] Running batch inference on directory: " << input_path << std::endl;
            ensure_directory(output_dir);

            FILE* rle_csv = nullptr;
            if (output_format == "rle") {
                std::string csv_path = output_path.empty() ? path_join(output_dir, "predictions.csv") : output_path;
                rle_csv = std::fopen(csv_path.c_str(), "w");
                if (rle_csv) {
                    std::fputs("image_id,rle\n", rle_csv);
                }
            }

            size_t count = 0;
            for (const auto& entry : fs::directory_iterator(input_path)) {
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
                    auto img = soar::data::ImageIO::load(entry.path().string(), static_cast<int>(m_cfg.in_channels));
                    auto res = predictor.predict(img);

                    std::string stem = entry.path().stem().string();
                    if (output_format == "rle" && rle_csv) {
                        std::fprintf(rle_csv, "%s,%s\n", stem.c_str(), res.rle_string.c_str());
                    } else {
                        std::string out_bmp = path_join(output_dir, stem + "_mask.bmp");
                        soar::data::ImageIO::save_bmp(out_bmp, res.probabilities);
                    }
                    count++;
                }
            }
            if (rle_csv) {
                std::fclose(rle_csv);
            }
            std::cout << "[SOAR Engine] Processed " << count << " images." << std::endl;
        } else {
            // Single image
            std::cout << "[SOAR Engine] Loading input image: " << input_path << "..." << std::endl;
            auto input_img = soar::data::ImageIO::load(input_path, static_cast<int>(m_cfg.in_channels));
            auto res = predictor.predict(input_img);

            std::cout << "[SOAR Engine] Inference completed! Foreground pixels: " << res.foreground_pixels << std::endl;
            std::cout << "[SOAR Engine] Fortran RLE length: " << res.rle_string.size() << " characters." << std::endl;

            std::string out_mask = output_path.empty() ? "output_mask.bmp" : output_path;
            if (output_format == "rle") {
                std::cout << "\nRLE Encoding:\n" << res.rle_string << "\n" << std::endl;
            } else {
                soar::data::ImageIO::save_bmp(out_mask, res.probabilities);
                std::cout << "[SOAR Engine] Saved probability mask to: " << out_mask << std::endl;
            }
        }
        ::pa_collect(true);
    }
    // --------------------------------------------------------------------------------------------------
    // TEST COMMAND (Dataset Testing Pipeline)
    // --------------------------------------------------------------------------------------------------
    else if (cmd == "test") {
        std::cout << "[SOAR Engine] Starting Testing Pipeline..." << std::endl;
        soar::losses::CompositeLoss loss_fn(bce_weight, dice_weight, pos_weight, dice_smooth);
        soar::engine::Validator validator(model, loss_fn, threshold);

        soar::engine::ValidationMetrics t_metrics{};

        if (!images_dir.empty() && !annotation_file.empty()) {
            soar::data::COCODataset ds(images_dir, annotation_file, static_cast<int>(m_cfg.in_channels), img_h, img_w);
            std::cout << "[SOAR Engine] Loaded COCO Test Dataset: " << ds.size() << " samples ("
                      << img_h << "x" << img_w << ")." << std::endl;
            t_metrics = validator.validate(ds, {}, save_dir, 20);
        } else if (!images_dir.empty() && !labels_dir.empty()) {
            soar::data::YOLODataset ds(images_dir, labels_dir, static_cast<int>(m_cfg.in_channels), img_h, img_w);
            std::cout << "[SOAR Engine] Loaded YOLO Test Dataset: " << ds.size() << " samples ("
                      << img_h << "x" << img_w << ")." << std::endl;
            t_metrics = validator.validate(ds, {}, save_dir, 20);
        } else {
            std::cerr << "[ERROR] Test command requires a dataset." << std::endl;
            return 1;
        }

        soar::engine::Validator::print_results(t_metrics, 0, 0, "Test");
        if (!save_dir.empty()) {
            std::cout << "[SOAR Engine] Test qualitative comparison visualizations saved to: " << save_dir << std::endl;
        }
        ::pa_collect(true);
    } else {
        std::cerr << "Unknown command: " << cmd << std::endl;
        print_usage();
        return 1;
    }

    model.reset();
    vk_ctx.reset();
    return 0;
}
