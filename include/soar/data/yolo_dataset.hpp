#pragma once

#include <soar/data/coco_dataset.hpp>
#include <string>
#include <vector>

namespace soar::data {

/**
 * @brief Native C++ YOLO segmentation format dataset parser and loader.
 */
class YOLODataset {
public:
    using BatchType = Batch;
    using BatchRequestType = std::vector<size_t>;

    YOLODataset(const std::string& images_dir,
                const std::string& labels_dir,
                int desired_channels = 1,
                size_t target_height = 0,
                size_t target_width = 0);

    [[nodiscard]] size_t size() const noexcept { return image_files_.size(); }
    [[nodiscard]] DatasetSample get_sample(size_t index) const;
    [[nodiscard]] Batch get_batch(const std::vector<size_t>& indices) const;

private:
    std::string images_dir_;
    std::string labels_dir_;
    int desired_channels_;
    size_t target_height_;
    size_t target_width_;

    std::vector<std::string> image_files_;
    void scan_directory();
};

} // namespace soar::data
