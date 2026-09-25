#include <soar/data/yolo_dataset.hpp>
#include <soar/data/image_io.hpp>
#include <soar/data/polygon_rasterizer.hpp>
#include <soar/nn/blocks.hpp>
#include <soar/core/logging.hpp>

#include <fstream>
#include <sstream>
#include <filesystem>

namespace soar::data {

YOLODataset::YOLODataset(const std::string& images_dir,
                         const std::string& labels_dir,
                         int desired_channels,
                         size_t target_height,
                         size_t target_width)
    : images_dir_(images_dir), labels_dir_(labels_dir),
      desired_channels_(desired_channels),
      target_height_(target_height), target_width_(target_width) {
    scan_directory();
    SOAR_LOG_INFO("Loaded YOLO dataset with {} images from {}.", image_files_.size(), images_dir_);
}

void YOLODataset::scan_directory() {
    if (!std::filesystem::exists(images_dir_)) {
        throw DeviceError("Images directory does not exist: " + images_dir_);
    }

    for (const auto& entry : std::filesystem::directory_iterator(images_dir_)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        for (auto& c : ext) c = static_cast<char>(std::tolower(c));
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
            image_files_.push_back(entry.path().filename().string());
        }
    }
    std::sort(image_files_.begin(), image_files_.end());
}

DatasetSample YOLODataset::get_sample(size_t index) const {
    if (index >= image_files_.size()) {
        throw ShapeError("YOLODataset sample index out of range: " + std::to_string(index));
    }

    std::string filename = image_files_[index];
    std::filesystem::path img_path = std::filesystem::path(images_dir_) / filename;

    DatasetSample sample;
    sample.filename = filename;
    sample.image_id = index;
    sample.image = ImageIO::load(img_path.string(), desired_channels_);

    size_t H = sample.image->dim(1);
    size_t W = sample.image->dim(2);
    sample.orig_height = H;
    sample.orig_width = W;

    sample.mask = Tensor::zeros({1, H, W});
    float* mask_data = sample.mask->data();

    // Find label txt file with matching stem
    std::filesystem::path stem = std::filesystem::path(filename).stem();
    std::filesystem::path txt_path = std::filesystem::path(labels_dir_) / (stem.string() + ".txt");

    if (std::filesystem::exists(txt_path)) {
        std::ifstream in(txt_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            int class_id = 0;
            if (!(iss >> class_id)) continue;

            std::vector<Point2D> pts;
            float norm_x = 0, norm_y = 0;
            while (iss >> norm_x >> norm_y) {
                pts.push_back(Point2D{norm_x * static_cast<float>(W), norm_y * static_cast<float>(H)});
            }
            if (pts.size() >= 3) {
                PolygonRasterizer::rasterize(mask_data, H, W, pts, 1.0f);
            }
        }
    }

    if (target_height_ > 0 && target_width_ > 0 && (target_height_ != H || target_width_ != W)) {
        sample.image = nn::resize_bilinear(sample.image, target_height_, target_width_);
        sample.mask = nn::resize_bilinear(sample.mask, target_height_, target_width_);
    }

    return sample;
}

} // namespace soar::data
