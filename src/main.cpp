#include <soar/nn/soar_model.hpp>
#include <soar/engine/trainer.hpp>
#include <soar/engine/predictor.hpp>
#include <soar/losses/losses.hpp>
#include <soar/optim/adamw.hpp>
#include <soar/core/logging.hpp>
#include <soar/core/config_parser.hpp>
#include <soar/data/image_io.hpp>
#include <soar/data/coco_dataset.hpp>
#include <soar/data/yolo_dataset.hpp>

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

void print_usage() {
    std::cout << "=================================================================\n"
              << "  SOAR: High-Performance C++20 Resolution-Preserving Engine    \n"
              << "=================================================================\n"
              << "Usage: soar_engine <command> [options]\n\n"
              << "Commands:\n"
              << "  train       Train model using native C++ dataset & autograd engine\n"
              << "  predict     Run inference on image and generate mask/RLE\n"
              << "  benchmark   Benchmark latency on high-resolution synthetic image\n"
              << "  help        Display this help message\n\n"
              << "Options for 'train':\n"
              << "  --config <path>                             Path to model config (e.g. configs/models/soar_nano1.yaml)\n"
              << "  --data-format <coco|yolo>                   Dataset format (default: coco)\n"
              << "  --images-dir <path>                         Path to images directory\n"
              << "  --annotation-file <path>                    Path to COCO JSON annotation file\n"
              << "  --labels-dir <path>                         Path to YOLO labels directory\n"
              << "  --epochs <N>                                Number of training epochs (default: 5)\n"
              << "  --lr <float>                                Learning rate (default: 0.001)\n"
              << "  --output <path>                             Output checkpoint path (.soarbinary)\n\n"
              << "Options for 'predict':\n"
              << "  --config <path>                             Path to model config YAML\n"
              << "  --weights <path>                            Path to trained .soarbinary weights\n"
              << "  --input <path>                              Input image file path\n"
              << "  --output <path>                             Output mask BMP file path\n"
              << "  --threshold <float>                         Sigmoid classification threshold (default: 0.5)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
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
    std::string data_format = "coco";
    std::string images_dir = "";
    std::string annotation_file = "";
    std::string labels_dir = "";
    std::string weights_path = "";
    std::string input_image_path = "";
    std::string output_mask_path = "output_mask.bmp";
    std::string output_weights_path = "checkpoint.soar";

    size_t epochs = 3;
    float learning_rate = 1e-3f;
    float threshold = 0.5f;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (arg == "--data-format" && i + 1 < argc) data_format = argv[++i];
        else if (arg == "--images-dir" && i + 1 < argc) images_dir = argv[++i];
        else if (arg == "--annotation-file" && i + 1 < argc) annotation_file = argv[++i];
        else if (arg == "--labels-dir" && i + 1 < argc) labels_dir = argv[++i];
        else if (arg == "--weights" && i + 1 < argc) weights_path = argv[++i];
        else if (arg == "--input" && i + 1 < argc) input_image_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) {
            if (cmd == "train") output_weights_path = argv[++i];
            else output_mask_path = argv[++i];
        }
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoul(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) learning_rate = std::stof(argv[++i]);
        else if (arg == "--threshold" && i + 1 < argc) threshold = std::stof(argv[++i]);
    }

    // Load Model Config
    soar::core::ModelConfig m_cfg;
    if (std::filesystem::exists(config_path)) {
        m_cfg = soar::core::ConfigParser::load_model_config(config_path);
    } else {
        std::cout << "[WARN] Config file not found, using default Nano configuration." << std::endl;
    }

    auto model = std::make_shared<soar::nn::SOARModel>(m_cfg.in_channels, m_cfg.num_classes, m_cfg.variant);
    std::cout << "[SOAR Engine] Model initialized (" << m_cfg.variant_str << ") with "
              << model->parameter_count() << " parameters." << std::endl;

    if (!weights_path.empty()) {
        std::cout << "[SOAR Engine] Loading weights from " << weights_path << "..." << std::endl;
        model->load_weights(weights_path);
    }

    if (cmd == "benchmark") {
        constexpr size_t H = 2048;
        constexpr size_t W = 2048;
        std::cout << "[SOAR Engine] Benchmarking 2048x2048 resolution-preserving inference..." << std::endl;
        auto input = soar::Tensor::randn({static_cast<int64_t>(m_cfg.in_channels), static_cast<int64_t>(H), static_cast<int64_t>(W)}, 0.5f, 0.2f);
        soar::engine::Predictor predictor(model, threshold);

        for (size_t s = 0; s < 3; ++s) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = predictor.predict(input);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "  Pass " << (s + 1) << "/3: Latency = " << ms << " ms (2048x2048)" << std::endl;
        }
    } else if (cmd == "train") {
        std::cout << "[SOAR Engine] Starting C++ Native Training Pipeline..." << std::endl;
        soar::losses::CompositeLoss loss_fn(0.5f, 0.5f);
        soar::optim::AdamWOptions opt_opts;
        opt_opts.lr = learning_rate;
        soar::optim::AdamW optimizer(model->parameters(), opt_opts);
        soar::engine::Trainer trainer(model, loss_fn, optimizer);

        // Check if dataset is provided
        if (!images_dir.empty() && (!annotation_file.empty() || !labels_dir.empty())) {
            size_t sample_count = 0;
            if (data_format == "coco") {
                soar::data::COCODataset ds(images_dir, annotation_file, static_cast<int>(m_cfg.in_channels));
                sample_count = ds.size();
                std::cout << "[SOAR Engine] Loaded COCO dataset: " << sample_count << " samples." << std::endl;
                for (size_t ep = 0; ep < epochs; ++ep) {
                    for (size_t i = 0; i < sample_count; ++i) {
                        auto sample = ds.get_sample(i);
                        auto metrics = trainer.train_step(sample.image, sample.mask);
                        std::cout << "Epoch " << (ep + 1) << "/" << epochs
                                  << " [Sample " << (i + 1) << "/" << sample_count << "] "
                                  << "Loss: " << metrics.loss << " | Dice: " << metrics.dice_score
                                  << " | IoU: " << metrics.iou_score << std::endl;
                    }
                }
            } else {
                soar::data::YOLODataset ds(images_dir, labels_dir, static_cast<int>(m_cfg.in_channels));
                sample_count = ds.size();
                std::cout << "[SOAR Engine] Loaded YOLO dataset: " << sample_count << " samples." << std::endl;
                for (size_t ep = 0; ep < epochs; ++ep) {
                    for (size_t i = 0; i < sample_count; ++i) {
                        auto sample = ds.get_sample(i);
                        auto metrics = trainer.train_step(sample.image, sample.mask);
                        std::cout << "Epoch " << (ep + 1) << "/" << epochs
                                  << " [Sample " << (i + 1) << "/" << sample_count << "] "
                                  << "Loss: " << metrics.loss << " | Dice: " << metrics.dice_score
                                  << " | IoU: " << metrics.iou_score << std::endl;
                    }
                }
            }
        } else {
            std::cout << "[SOAR Engine] No dataset path passed. Running synthetic training steps..." << std::endl;
            auto img = soar::Tensor::randn({static_cast<int64_t>(m_cfg.in_channels), 64, 64}, 0.5f, 0.2f);
            auto msk = soar::Tensor::zeros({1, 64, 64});
            for (size_t i = 0; i < msk->numel(); i += 2) msk->data()[i] = 1.0f;

            for (size_t ep = 0; ep < epochs; ++ep) {
                auto metrics = trainer.train_step(img, msk);
                std::cout << "Epoch " << (ep + 1) << "/" << epochs
                          << ": Loss = " << metrics.loss
                          << " | Dice = " << metrics.dice_score
                          << " | IoU = " << metrics.iou_score << std::endl;
            }
        }

        std::cout << "[SOAR Engine] Saving model weights to " << output_weights_path << "..." << std::endl;
        model->save_weights(output_weights_path);
        std::cout << "[SOAR Engine] Training completed successfully!" << std::endl;
    } else if (cmd == "predict") {
        if (input_image_path.empty()) {
            std::cerr << "Error: --input <path> is required for prediction." << std::endl;
            return 1;
        }

        std::cout << "[SOAR Engine] Loading input image: " << input_image_path << "..." << std::endl;
        auto input_img = soar::data::ImageIO::load(input_image_path, static_cast<int>(m_cfg.in_channels));

        soar::engine::Predictor predictor(model, threshold);
        auto res = predictor.predict(input_img);

        std::cout << "[SOAR Engine] Inference completed! Foreground pixels: " << res.foreground_pixels << std::endl;
        std::cout << "[SOAR Engine] Kaggle Fortran RLE length: " << res.rle_string.size() << " characters." << std::endl;

        if (!output_mask_path.empty()) {
            soar::data::ImageIO::save_bmp(output_mask_path, res.probabilities);
            std::cout << "[SOAR Engine] Saved probability mask to: " << output_mask_path << std::endl;
        }
    } else {
        std::cerr << "Unknown command: " << cmd << std::endl;
        print_usage();
        return 1;
    }

    return 0;
}
