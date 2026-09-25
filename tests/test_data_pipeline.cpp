#include <soar/data/image_io.hpp>
#include <soar/data/polygon_rasterizer.hpp>
#include <soar/data/coco_dataset.hpp>
#include <soar/data/yolo_dataset.hpp>
#include <soar/core/config_parser.hpp>

#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>

void test_image_io() {
    std::cout << "[TEST] Running test_image_io..." << std::endl;
    constexpr size_t H = 64;
    constexpr size_t W = 64;
    auto img = soar::Tensor::zeros({1, H, W});
    for (size_t y = 10; y < 30; ++y) {
        for (size_t x = 10; x < 30; ++x) {
            img->data()[y * W + x] = 1.0f;
        }
    }

    std::string tmp_bmp = (std::filesystem::temp_directory_path() / "test_io.bmp").string();
    bool saved = soar::data::ImageIO::save_bmp(tmp_bmp, img);
    assert(saved);

    auto loaded = soar::data::ImageIO::load(tmp_bmp, 1);
    assert(loaded->dim(0) == 1);
    assert(loaded->dim(1) == H);
    assert(loaded->dim(2) == W);

    // Verify center pixel is white (1.0f)
    assert(loaded->data()[20 * W + 20] > 0.99f);
    // Verify border pixel is black (0.0f)
    assert(loaded->data()[0] < 0.01f);

    std::filesystem::remove(tmp_bmp);
    std::cout << "  -> test_image_io PASSED" << std::endl;
}

void test_polygon_rasterizer() {
    std::cout << "[TEST] Running test_polygon_rasterizer..." << std::endl;
    constexpr size_t H = 64;
    constexpr size_t W = 64;
    std::vector<float> mask(H * W, 0.0f);

    std::vector<soar::data::Point2D> poly = {
        {10.0f, 10.0f},
        {30.0f, 10.0f},
        {30.0f, 30.0f},
        {10.0f, 30.0f}
    };

    soar::data::PolygonRasterizer::rasterize(mask.data(), H, W, poly, 1.0f);

    // Verify points inside polygon are filled
    assert(mask[15 * W + 15] == 1.0f);
    assert(mask[25 * W + 25] == 1.0f);
    // Verify outside is 0.0f
    assert(mask[5 * W + 5] == 0.0f);
    assert(mask[40 * W + 40] == 0.0f);

    std::cout << "  -> test_polygon_rasterizer PASSED" << std::endl;
}

void test_coco_dataset() {
    std::cout << "[TEST] Running test_coco_dataset..." << std::endl;
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "soar_coco_test";
    std::filesystem::create_directories(tmp_dir);

    // Create a dummy image
    auto img = soar::Tensor::ones({1, 64, 64});
    std::string img_path = (tmp_dir / "sample0.bmp").string();
    soar::data::ImageIO::save_bmp(img_path, img);

    // Create a dummy COCO JSON
    std::string json_content = R"({
        "images": [{"id": 1, "file_name": "sample0.bmp", "width": 64, "height": 64}],
        "annotations": [{
            "id": 101, "image_id": 1, "category_id": 1,
            "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]]
        }]
    })";

    std::string json_path = (tmp_dir / "annotations.json").string();
    std::ofstream out(json_path);
    out << json_content;
    out.close();

    soar::data::COCODataset ds(tmp_dir.string(), json_path, 1);
    assert(ds.size() == 1);

    auto sample = ds.get_sample(0);
    assert(sample.filename == "sample0.bmp");
    assert(sample.image->dim(1) == 64 && sample.image->dim(2) == 64);
    assert(sample.mask->dim(1) == 64 && sample.mask->dim(2) == 64);
    assert(sample.mask->data()[20 * 64 + 20] == 1.0f);

    std::filesystem::remove_all(tmp_dir);
    std::cout << "  -> test_coco_dataset PASSED" << std::endl;
}

void test_yolo_dataset() {
    std::cout << "[TEST] Running test_yolo_dataset..." << std::endl;
    std::filesystem::path tmp_dir = std::filesystem::temp_directory_path() / "soar_yolo_test";
    std::filesystem::path img_dir = tmp_dir / "images";
    std::filesystem::path lbl_dir = tmp_dir / "labels";
    std::filesystem::create_directories(img_dir);
    std::filesystem::create_directories(lbl_dir);

    auto img = soar::Tensor::ones({1, 100, 100});
    std::string img_path = (img_dir / "yolo_sample.bmp").string();
    soar::data::ImageIO::save_bmp(img_path, img);

    // Create a normalized polygon in labels: class 0, box from (0.1, 0.1) to (0.5, 0.5)
    std::string lbl_path = (lbl_dir / "yolo_sample.txt").string();
    std::ofstream out(lbl_path);
    out << "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n";
    out.close();

    soar::data::YOLODataset ds(img_dir.string(), lbl_dir.string(), 1);
    assert(ds.size() == 1);

    auto sample = ds.get_sample(0);
    assert(sample.image->dim(1) == 100 && sample.image->dim(2) == 100);
    assert(sample.mask->data()[30 * 100 + 30] == 1.0f);
    assert(sample.mask->data()[80 * 100 + 80] == 0.0f);

    std::filesystem::remove_all(tmp_dir);
    std::cout << "  -> test_yolo_dataset PASSED" << std::endl;
}

void test_config_parser() {
    std::cout << "[TEST] Running test_config_parser..." << std::endl;
    auto cfg = soar::core::ConfigParser::load_model_config("configs/models/soar_nano1.yaml");
    assert(cfg.variant == soar::nn::ModelVariant::Nano);
    assert(cfg.num_classes == 1);
    std::cout << "  Loaded config: variant=" << cfg.variant_str
              << ", in_channels=" << cfg.in_channels
              << ", num_classes=" << cfg.num_classes << std::endl;
    std::cout << "  -> test_config_parser PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "=========================================================" << std::endl;
        std::cout << "  SOAR Data Pipeline & Config Parser Verification       " << std::endl;
        std::cout << "=========================================================" << std::endl;

        test_image_io();
        test_polygon_rasterizer();
        test_coco_dataset();
        test_yolo_dataset();
        test_config_parser();

        std::cout << std::endl;
        std::cout << ">>> ALL DATA PIPELINE & CONFIG TESTS PASSED! <<<" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
