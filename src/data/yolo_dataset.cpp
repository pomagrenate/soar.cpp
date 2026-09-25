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

static std::string join_paths(const std::string& base, const std::string& sub) {
    if (base.empty() || base == ".") return sub;
    if (base.back() == '/' || base.back() == '\\') return base + sub;
    return base + "/" + sub;
}

static std::string get_stem(const std::string& filename) {
    size_t last_slash = filename.find_last_of("/\\");
    std::string base = (last_slash == std::string::npos) ? filename : filename.substr(last_slash + 1);
    size_t last_dot = base.rfind('.');
    if (last_dot != std::string::npos) {
        return base.substr(0, last_dot);
    }
    return base;
}

DatasetSample YOLODataset::get_sample(size_t index) const {
    if (index >= image_files_.size()) {
        throw ShapeError("YOLODataset sample index out of range: " + std::to_string(index));
    }

    std::string filename = image_files_[index];
    std::string img_path_str = join_paths(images_dir_, filename);

    DatasetSample sample;
    sample.filename = filename;
    sample.image_id = index;
    sample.image = ImageIO::load(img_path_str, desired_channels_);

    size_t H = sample.image->dim(1);
    size_t W = sample.image->dim(2);
    sample.orig_height = H;
    sample.orig_width = W;

    sample.mask = Tensor::zeros({1, static_cast<int64_t>(H), static_cast<int64_t>(W)});
    float* mask_data = sample.mask->data();

    // Find label txt file with matching stem
    std::string stem = get_stem(filename);
    std::string txt_path_str = join_paths(labels_dir_, stem + ".txt");

    FILE* in = std::fopen(txt_path_str.c_str(), "r");
    if (in) {
        char line[4096];
        while (std::fgets(line, sizeof(line), in)) {
            char* ptr = line;
            char* end = nullptr;
            [[maybe_unused]] long class_id = std::strtol(ptr, &end, 10);
            if (end == ptr) continue;
            ptr = end;

            std::vector<Point2D> pts;
            while (*ptr != '\0' && *ptr != '\n' && *ptr != '\r') {
                float norm_x = std::strtof(ptr, &end);
                if (end == ptr) break;
                ptr = end;
                float norm_y = std::strtof(ptr, &end);
                if (end == ptr) break;
                ptr = end;
                pts.push_back(Point2D{norm_x * static_cast<float>(W), norm_y * static_cast<float>(H)});
            }
            if (pts.size() >= 3) {
                PolygonRasterizer::rasterize(mask_data, H, W, pts, 1.0f);
            }
        }
        std::fclose(in);
    }

    if (target_height_ > 0 && target_width_ > 0 && (target_height_ != H || target_width_ != W)) {
        sample.image = nn::resize_bilinear(sample.image, target_height_, target_width_);
        sample.mask = nn::resize_bilinear(sample.mask, target_height_, target_width_);
    }

    return sample;
}

} // namespace soar::data
