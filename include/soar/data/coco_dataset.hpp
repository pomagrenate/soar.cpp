#pragma once

#include <soar/tensor/tensor.hpp>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace soar::data {

struct DatasetSample {
    TensorPtr image;
    TensorPtr mask;
    std::string filename;
    uint64_t image_id{0};
    size_t orig_height{0};
    size_t orig_width{0};
};

/**
 * @brief Native C++ COCO format dataset parser and loader.
 */
class COCODataset {
public:
    COCODataset(const std::string& images_dir,
                const std::string& annotation_json_path,
                int desired_channels = 1,
                size_t target_height = 0,
                size_t target_width = 0);

    [[nodiscard]] size_t size() const noexcept { return image_records_.size(); }
    [[nodiscard]] DatasetSample get_sample(size_t index) const;

private:
    struct ImageRecord {
        uint64_t id{0};
        std::string str_id;
        std::string file_name;
        size_t height{0};
        size_t width{0};
    };

    struct AnnotationRecord {
        uint64_t image_id{0};
        std::string image_id_str;
        int category_id{0};
        std::vector<std::vector<float>> polygons;
    };

    std::string images_dir_;
    int desired_channels_;
    size_t target_height_;
    size_t target_width_;

    std::vector<ImageRecord> image_records_;
    std::map<uint64_t, std::vector<AnnotationRecord>> annotations_by_image_;
    std::unordered_map<std::string, std::vector<AnnotationRecord>> annotations_by_key_;
    std::unordered_map<std::string, std::string> file_map_;

    void parse_json(const std::string& json_path);
    void index_images_dir(const std::string& dir);
};

} // namespace soar::data
