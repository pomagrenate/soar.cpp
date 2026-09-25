#include <soar/data/coco_dataset.hpp>
#include <soar/data/image_io.hpp>
#include <soar/data/polygon_rasterizer.hpp>
#include <soar/nn/blocks.hpp>
#include <soar/core/logging.hpp>

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace soar::data {

COCODataset::COCODataset(const std::string& images_dir,
                         const std::string& annotation_json_path,
                         int desired_channels,
                         size_t target_height,
                         size_t target_width)
    : images_dir_(images_dir), desired_channels_(desired_channels),
      target_height_(target_height), target_width_(target_width) {
    parse_json(annotation_json_path);
    SOAR_LOG_INFO("Loaded COCO dataset with {} images and {} annotated entries.",
                  image_records_.size(), annotations_by_image_.size());
}

void COCODataset::parse_json(const std::string& json_path) {
    std::ifstream in(json_path);
    if (!in.is_open()) {
        throw DeviceError("Failed to open COCO annotation JSON file: " + json_path);
    }

    nlohmann::json j;
    in >> j;

    if (j.contains("images") && j["images"].is_array()) {
        for (const auto& img : j["images"]) {
            ImageRecord rec;
            rec.id = img.value("id", uint64_t(0));
            rec.file_name = img.value("file_name", "");
            rec.height = img.value("height", size_t(0));
            rec.width = img.value("width", size_t(0));
            image_records_.push_back(std::move(rec));
        }
    }

    if (j.contains("annotations") && j["annotations"].is_array()) {
        for (const auto& ann : j["annotations"]) {
            AnnotationRecord rec;
            rec.image_id = ann.value("image_id", uint64_t(0));
            rec.category_id = ann.value("category_id", 0);

            if (ann.contains("segmentation") && ann["segmentation"].is_array()) {
                for (const auto& poly : ann["segmentation"]) {
                    if (poly.is_array()) {
                        std::vector<float> coords;
                        for (const auto& coord : poly) {
                            coords.push_back(coord.get<float>());
                        }
                        rec.polygons.push_back(std::move(coords));
                    }
                }
            }
            annotations_by_image_[rec.image_id].push_back(std::move(rec));
        }
    }
}

DatasetSample COCODataset::get_sample(size_t index) const {
    if (index >= image_records_.size()) {
        throw ShapeError("COCODataset sample index out of range: " + std::to_string(index));
    }

    const auto& rec = image_records_[index];
    std::filesystem::path img_path = std::filesystem::path(images_dir_) / rec.file_name;

    DatasetSample sample;
    sample.filename = rec.file_name;
    sample.image_id = rec.id;
    sample.image = ImageIO::load(img_path.string(), desired_channels_);

    size_t H = sample.image->dim(1);
    size_t W = sample.image->dim(2);
    sample.orig_height = H;
    sample.orig_width = W;

    sample.mask = Tensor::zeros({1, static_cast<int64_t>(H), static_cast<int64_t>(W)});
    float* mask_data = sample.mask->data();

    auto it = annotations_by_image_.find(rec.id);
    if (it != annotations_by_image_.end()) {
        for (const auto& ann : it->second) {
            for (const auto& poly_coords : ann.polygons) {
                if (poly_coords.size() < 6) continue;
                std::vector<Point2D> pts;
                pts.reserve(poly_coords.size() / 2);
                for (size_t i = 0; i + 1 < poly_coords.size(); i += 2) {
                    pts.push_back(Point2D{poly_coords[i], poly_coords[i + 1]});
                }
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
