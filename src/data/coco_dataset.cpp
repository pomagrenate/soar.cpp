#include <soar/data/coco_dataset.hpp>
#include <soar/data/image_io.hpp>
#include <soar/data/polygon_rasterizer.hpp>
#include <soar/nn/blocks.hpp>
#include <soar/core/logging.hpp>

#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace soar::data {

static bool is_image_file_ext(const std::string& ext) {
    return (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" ||
            ext == ".tiff" || ext == ".tif" || ext == ".npy" || ext == ".fits");
}

void COCODataset::index_images_dir(const std::string& dir) {
    if (dir.empty() || !fs::exists(dir)) return;
    try {
        if (fs::is_directory(dir)) {
            auto add_file = [&](const fs::path& p) {
                std::string ext = p.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c){ return std::tolower(c); });
                if (is_image_file_ext(ext)) {
                    std::string full_p = p.generic_string();
                    std::string fname = p.filename().generic_string();
                    std::string stem = p.stem().generic_string();
                    file_map_[fname] = full_p;
                    file_map_[stem] = full_p;
                }
            };

            for (const auto& entry : fs::directory_iterator(dir)) {
                if (entry.is_regular_file()) {
                    add_file(entry.path());
                } else if (entry.is_directory() && dir != "." && dir != "./") {
                    std::string dname = entry.path().filename().string();
                    if (dname == "train_images" || dname == "images" || dname == "train") {
                        try {
                            for (const auto& sub_entry : fs::directory_iterator(entry.path())) {
                                if (sub_entry.is_regular_file()) {
                                    add_file(sub_entry.path());
                                    std::string sub_rel = dname + "/" + sub_entry.path().filename().generic_string();
                                    file_map_[sub_rel] = sub_entry.path().generic_string();
                                }
                            }
                        } catch (...) {}
                    }
                }
            }
        }
    } catch (...) {
        // Fallback gracefully
    }
}

COCODataset::COCODataset(const std::string& images_dir,
                         const std::string& annotation_json_path,
                         int desired_channels,
                         size_t target_height,
                         size_t target_width)
    : images_dir_(images_dir), desired_channels_(desired_channels),
      target_height_(target_height), target_width_(target_width) {
    index_images_dir(images_dir_);
    parse_json(annotation_json_path);
    SOAR_LOG_INFO("Loaded COCO dataset with {} images, {} annotated entries, and {} indexed files.",
                  image_records_.size(), annotations_by_image_.size(), file_map_.size());
}

static std::string json_str_id(const nlohmann::json& obj, const std::string& key) {
    if (!obj.contains(key)) return "";
    const auto& v = obj[key];
    if (v.is_string()) return v.get<std::string>();
    if (v.is_number_integer()) return std::to_string(v.get<int64_t>());
    if (v.is_number_unsigned()) return std::to_string(v.get<uint64_t>());
    if (v.is_number_float()) return std::to_string(static_cast<int64_t>(v.get<double>()));
    return "";
}

static uint64_t json_uint_id(const nlohmann::json& obj, const std::string& key, uint64_t def = 0) {
    if (!obj.contains(key)) return def;
    const auto& v = obj[key];
    if (v.is_number_unsigned()) return v.get<uint64_t>();
    if (v.is_number_integer()) return static_cast<uint64_t>(std::max(int64_t(0), v.get<int64_t>()));
    if (v.is_number_float()) return static_cast<uint64_t>(std::max(0.0, v.get<double>()));
    if (v.is_string()) {
        try {
            return std::stoull(v.get<std::string>());
        } catch (...) {
            return def;
        }
    }
    return def;
}

static size_t json_size_val(const nlohmann::json& obj, const std::string& key, size_t def = 0) {
    if (!obj.contains(key)) return def;
    const auto& v = obj[key];
    if (v.is_number_unsigned()) return v.get<size_t>();
    if (v.is_number_integer()) return static_cast<size_t>(std::max(int64_t(0), v.get<int64_t>()));
    if (v.is_number_float()) return static_cast<size_t>(std::max(0.0, v.get<double>()));
    if (v.is_string()) {
        try {
            return std::stoull(v.get<std::string>());
        } catch (...) {
            return def;
        }
    }
    return def;
}

static int json_int_val(const nlohmann::json& obj, const std::string& key, int def = 0) {
    if (!obj.contains(key)) return def;
    const auto& v = obj[key];
    if (v.is_number_integer()) return v.get<int>();
    if (v.is_number_unsigned()) return static_cast<int>(v.get<unsigned int>());
    if (v.is_string()) {
        try {
            return std::stoi(v.get<std::string>());
        } catch (...) {
            return def;
        }
    }
    return def;
}

static float json_to_float(const nlohmann::json& v, float def = 0.0f) {
    if (v.is_number()) return v.get<float>();
    if (v.is_string()) {
        try {
            return std::stof(v.get<std::string>());
        } catch (...) {
            return def;
        }
    }
    return def;
}

void COCODataset::parse_json(const std::string& json_path) {
    FILE* in = std::fopen(json_path.c_str(), "rb");
    if (!in) {
        throw DeviceError("Failed to open COCO annotation JSON file: " + json_path);
    }
    std::fseek(in, 0, SEEK_END);
    long sz = std::ftell(in);
    std::fseek(in, 0, SEEK_SET);

    std::string content;
    if (sz > 0) {
        content.resize(static_cast<size_t>(sz));
        size_t read_bytes = std::fread(content.data(), 1, sz, in);
        content.resize(read_bytes);
    }
    std::fclose(in);

    nlohmann::json j = nlohmann::json::parse(content, nullptr, false);
    if (j.is_discarded()) {
        throw DeviceError("Failed to parse COCO annotation JSON file: " + json_path);
    }

    std::unordered_map<std::string, std::string> id_to_filename;

    if (j.contains("images") && j["images"].is_array()) {
        for (const auto& img : j["images"]) {
            ImageRecord rec;
            rec.str_id = json_str_id(img, "id");
            rec.id = json_uint_id(img, "id");
            rec.file_name = img.value("file_name", "");
            rec.height = json_size_val(img, "height");
            rec.width = json_size_val(img, "width");

            if (!rec.str_id.empty()) {
                id_to_filename[rec.str_id] = rec.file_name;
            }
            if (rec.id != 0) {
                id_to_filename[std::to_string(rec.id)] = rec.file_name;
            }

            image_records_.push_back(std::move(rec));
        }
    }

    if (j.contains("annotations") && j["annotations"].is_array()) {
        for (const auto& ann : j["annotations"]) {
            AnnotationRecord rec;
            rec.image_id_str = json_str_id(ann, "image_id");
            rec.image_id = json_uint_id(ann, "image_id");
            rec.category_id = json_int_val(ann, "category_id", 0);

            if (ann.contains("segmentation") && ann["segmentation"].is_array()) {
                for (const auto& poly : ann["segmentation"]) {
                    if (poly.is_array()) {
                        std::vector<float> coords;
                        coords.reserve(poly.size());
                        for (const auto& coord : poly) {
                            coords.push_back(json_to_float(coord));
                        }
                        if (coords.size() >= 6) {
                            rec.polygons.push_back(std::move(coords));
                        }
                    }
                }
            }

            if (rec.image_id != 0) {
                annotations_by_image_[rec.image_id].push_back(rec);
            }

            std::string resolved_fname;
            if (!rec.image_id_str.empty()) {
                annotations_by_key_[rec.image_id_str].push_back(rec);
                auto it = id_to_filename.find(rec.image_id_str);
                if (it != id_to_filename.end()) resolved_fname = it->second;
            }
            if (resolved_fname.empty() && rec.image_id != 0) {
                auto it = id_to_filename.find(std::to_string(rec.image_id));
                if (it != id_to_filename.end()) resolved_fname = it->second;
            }

            if (!resolved_fname.empty()) {
                annotations_by_key_[resolved_fname].push_back(rec);
                std::string bname = fs::path(resolved_fname).filename().generic_string();
                std::string stem = fs::path(resolved_fname).stem().generic_string();
                if (bname != resolved_fname) annotations_by_key_[bname].push_back(rec);
                if (stem != resolved_fname && stem != bname) annotations_by_key_[stem].push_back(rec);
            }
        }
    }
}

DatasetSample COCODataset::get_sample(size_t index) const {
    if (index >= image_records_.size()) {
        throw ShapeError("COCODataset sample index out of range: " + std::to_string(index));
    }

    const auto& rec = image_records_[index];
    std::string img_path;
    
    // Check direct path first
    std::string candidate;
    if (images_dir_.empty() || images_dir_ == ".") {
        candidate = rec.file_name;
    } else if (images_dir_.back() == '/' || images_dir_.back() == '\\') {
        candidate = images_dir_ + rec.file_name;
    } else {
        candidate = images_dir_ + "/" + rec.file_name;
    }

    if (fs::exists(candidate)) {
        img_path = candidate;
    } else if (fs::exists(rec.file_name)) {
        img_path = rec.file_name;
    } else {
        // Search in pre-indexed file_map_
        auto it = file_map_.find(rec.file_name);
        if (it != file_map_.end()) {
            img_path = it->second;
        } else {
            std::string fname = fs::path(rec.file_name).filename().generic_string();
            auto it2 = file_map_.find(fname);
            if (it2 != file_map_.end()) {
                img_path = it2->second;
            } else {
                std::string stem = fs::path(rec.file_name).stem().generic_string();
                auto it3 = file_map_.find(stem);
                if (it3 != file_map_.end()) {
                    img_path = it3->second;
                } else {
                    img_path = candidate;
                }
            }
        }
    }

    DatasetSample sample;
    sample.filename = rec.file_name;
    sample.image_id = rec.id;
    sample.image = ImageIO::load(img_path, desired_channels_);

    size_t H = sample.image->dim(1);
    size_t W = sample.image->dim(2);
    sample.orig_height = H;
    sample.orig_width = W;

    sample.mask = Tensor::zeros({1, static_cast<int64_t>(H), static_cast<int64_t>(W)});
    float* mask_data = sample.mask->data();

    const std::vector<AnnotationRecord>* matched_anns = nullptr;
    auto it_key = annotations_by_key_.find(rec.file_name);
    if (it_key != annotations_by_key_.end()) {
        matched_anns = &it_key->second;
    } else {
        std::string bname = fs::path(rec.file_name).filename().generic_string();
        it_key = annotations_by_key_.find(bname);
        if (it_key != annotations_by_key_.end()) {
            matched_anns = &it_key->second;
        } else {
            std::string stem = fs::path(rec.file_name).stem().generic_string();
            it_key = annotations_by_key_.find(stem);
            if (it_key != annotations_by_key_.end()) {
                matched_anns = &it_key->second;
            } else if (!rec.str_id.empty()) {
                it_key = annotations_by_key_.find(rec.str_id);
                if (it_key != annotations_by_key_.end()) {
                    matched_anns = &it_key->second;
                }
            }
        }
    }

    if (matched_anns) {
        for (const auto& ann : *matched_anns) {
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
    } else {
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
    }

    if (target_height_ > 0 && target_width_ > 0 && (target_height_ != H || target_width_ != W)) {
        sample.image = nn::resize_bilinear(sample.image, target_height_, target_width_);
        sample.mask = nn::resize_bilinear(sample.mask, target_height_, target_width_);
    }

    return sample;
}

} // namespace soar::data
