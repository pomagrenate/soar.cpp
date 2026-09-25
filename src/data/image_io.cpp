#include <soar/data/image_io.hpp>
#include <soar/core/logging.hpp>

#include "palloc.h"
#define STBI_MALLOC(sz) ::pa_malloc(sz)
#define STBI_REALLOC(p,newsz) ::pa_realloc(p,newsz)
#define STBI_FREE(p) ::pa_free(p)
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace soar::data {

TensorPtr ImageIO::load(const std::string& path, int desired_channels) {
    int w = 0;
    int h = 0;
    int orig_channels = 0;

    unsigned char* raw_pixels = stbi_load(path.c_str(), &w, &h, &orig_channels, desired_channels);
    if (!raw_pixels) {
        throw DeviceError("Failed to load image at: " + path + " (Reason: " + stbi_failure_reason() + ")");
    }

    size_t C = static_cast<size_t>(desired_channels);
    size_t H = static_cast<size_t>(h);
    size_t W = static_cast<size_t>(w);

    TensorPtr tensor = Tensor::create({static_cast<int64_t>(C), static_cast<int64_t>(H), static_cast<int64_t>(W)});
    float* tensor_data = tensor->data();

    // Convert interleaved HWC unsigned char to planar CHW float normalized to [0, 1]
    for (size_t c = 0; c < C; ++c) {
        for (size_t y = 0; y < H; ++y) {
            for (size_t x = 0; x < W; ++x) {
                size_t raw_idx = (y * W + x) * C + c;
                float normalized = static_cast<float>(raw_pixels[raw_idx]) / 255.0f;
                tensor_data[c * (H * W) + y * W + x] = normalized;
            }
        }
    }

    stbi_image_free(raw_pixels);
    return tensor;
}

bool ImageIO::save_bmp(const std::string& path, const TensorPtr& tensor) {
    size_t C = 1;
    size_t H = 0;
    size_t W = 0;

    if (tensor->ndim() == 3) {
        C = tensor->dim(0);
        H = tensor->dim(1);
        W = tensor->dim(2);
    } else if (tensor->ndim() == 2) {
        C = 1;
        H = tensor->dim(0);
        W = tensor->dim(1);
    } else {
        throw ShapeError("save_bmp expects 2D or 3D tensor");
    }

    // BMP rows are padded to multiples of 4 bytes
    size_t row_padded = (W * 3 + 3) & (~3);
    uint32_t image_size = static_cast<uint32_t>(row_padded * H);
    uint32_t file_size = 54 + image_size;

    uint8_t file_header[14] = {
        'B', 'M',
        static_cast<uint8_t>(file_size & 0xFF),
        static_cast<uint8_t>((file_size >> 8) & 0xFF),
        static_cast<uint8_t>((file_size >> 16) & 0xFF),
        static_cast<uint8_t>((file_size >> 24) & 0xFF),
        0, 0, 0, 0,
        54, 0, 0, 0
    };

    uint8_t info_header[40] = {
        40, 0, 0, 0,
        static_cast<uint8_t>(W & 0xFF),
        static_cast<uint8_t>((W >> 8) & 0xFF),
        static_cast<uint8_t>((W >> 16) & 0xFF),
        static_cast<uint8_t>((W >> 24) & 0xFF),
        static_cast<uint8_t>(H & 0xFF),
        static_cast<uint8_t>((H >> 8) & 0xFF),
        static_cast<uint8_t>((H >> 16) & 0xFF),
        static_cast<uint8_t>((H >> 24) & 0xFF),
        1, 0,
        24, 0, // 24 bpp
        0, 0, 0, 0, // uncompressed BI_RGB
        static_cast<uint8_t>(image_size & 0xFF),
        static_cast<uint8_t>((image_size >> 8) & 0xFF),
        static_cast<uint8_t>((image_size >> 16) & 0xFF),
        static_cast<uint8_t>((image_size >> 24) & 0xFF),
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    FILE* out = std::fopen(path.c_str(), "wb");
    if (!out) return false;

    std::fwrite(file_header, 1, 14, out);
    std::fwrite(info_header, 1, 40, out);

    const float* tensor_data = tensor->data();
    std::vector<uint8_t> row(row_padded, 0);

    // BMP stores rows bottom-to-top
    for (int y = static_cast<int>(H) - 1; y >= 0; --y) {
        for (size_t x = 0; x < W; ++x) {
            uint8_t b, g, r;
            if (C == 1) {
                float val = std::clamp(tensor_data[y * W + x], 0.0f, 1.0f);
                uint8_t gray = static_cast<uint8_t>(std::round(val * 255.0f));
                b = g = r = gray;
            } else {
                float rv = std::clamp(tensor_data[0 * (H * W) + y * W + x], 0.0f, 1.0f);
                float gv = std::clamp(tensor_data[1 * (H * W) + y * W + x], 0.0f, 1.0f);
                float bv = std::clamp(tensor_data[2 * (H * W) + y * W + x], 0.0f, 1.0f);
                r = static_cast<uint8_t>(std::round(rv * 255.0f));
                g = static_cast<uint8_t>(std::round(gv * 255.0f));
                b = static_cast<uint8_t>(std::round(bv * 255.0f));
            }
            row[x * 3 + 0] = b; // BGR format
            row[x * 3 + 1] = g;
            row[x * 3 + 2] = r;
        }
        std::fwrite(row.data(), 1, row_padded, out);
    }
    std::fclose(out);

    return true;
}

bool ImageIO::save_comparison_bmp(const std::string& path,
                                  const TensorPtr& image,
                                  const TensorPtr& true_mask,
                                  const TensorPtr& pred_mask) {
    if (!image || !true_mask || !pred_mask) return false;

    size_t C = image->ndim() == 3 ? image->dim(0) : 1;
    size_t H = image->ndim() == 3 ? image->dim(1) : image->dim(0);
    size_t W = image->ndim() == 3 ? image->dim(2) : image->dim(1);

    size_t total_W = 4 * W;
    size_t row_padded = (total_W * 3 + 3) & (~3);
    uint32_t image_size = static_cast<uint32_t>(row_padded * H);
    uint32_t file_size = 54 + image_size;

    uint8_t file_header[14] = {
        'B', 'M',
        static_cast<uint8_t>(file_size & 0xFF),
        static_cast<uint8_t>((file_size >> 8) & 0xFF),
        static_cast<uint8_t>((file_size >> 16) & 0xFF),
        static_cast<uint8_t>((file_size >> 24) & 0xFF),
        0, 0, 0, 0,
        54, 0, 0, 0
    };

    uint8_t info_header[40] = {
        40, 0, 0, 0,
        static_cast<uint8_t>(total_W & 0xFF),
        static_cast<uint8_t>((total_W >> 8) & 0xFF),
        static_cast<uint8_t>((total_W >> 16) & 0xFF),
        static_cast<uint8_t>((total_W >> 24) & 0xFF),
        static_cast<uint8_t>(H & 0xFF),
        static_cast<uint8_t>((H >> 8) & 0xFF),
        static_cast<uint8_t>((H >> 16) & 0xFF),
        static_cast<uint8_t>((H >> 24) & 0xFF),
        1, 0,
        24, 0,
        0, 0, 0, 0,
        static_cast<uint8_t>(image_size & 0xFF),
        static_cast<uint8_t>((image_size >> 8) & 0xFF),
        static_cast<uint8_t>((image_size >> 16) & 0xFF),
        static_cast<uint8_t>((image_size >> 24) & 0xFF),
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    FILE* out = std::fopen(path.c_str(), "wb");
    if (!out) return false;

    std::fwrite(file_header, 1, 14, out);
    std::fwrite(info_header, 1, 40, out);

    const float* img_d = image->data();
    const float* tm_d = true_mask->data();
    const float* pm_d = pred_mask->data();

    std::vector<uint8_t> row(row_padded, 0);

    for (int y = static_cast<int>(H) - 1; y >= 0; --y) {
        for (size_t x = 0; x < W; ++x) {
            size_t idx = y * W + x;

            // Base raw pixel (R, G, B)
            float base_r, base_g, base_b;
            if (C == 1) {
                float v = std::clamp(img_d[idx], 0.0f, 1.0f);
                base_r = base_g = base_b = v;
            } else {
                base_r = std::clamp(img_d[0 * (H * W) + idx], 0.0f, 1.0f);
                base_g = std::clamp(img_d[1 * (H * W) + idx], 0.0f, 1.0f);
                base_b = std::clamp(img_d[2 * (H * W) + idx], 0.0f, 1.0f);
            }

            bool t_val = tm_d[idx] > 0.5f;
            bool p_val = pm_d[idx] > 0.5f;

            // Panel 0: Raw Image
            size_t p0_x = x;
            row[p0_x * 3 + 0] = static_cast<uint8_t>(std::round(base_b * 255.0f));
            row[p0_x * 3 + 1] = static_cast<uint8_t>(std::round(base_g * 255.0f));
            row[p0_x * 3 + 2] = static_cast<uint8_t>(std::round(base_r * 255.0f));

            // Panel 1: Ground Truth Overlay (Green tint)
            size_t p1_x = W + x;
            float r1 = t_val ? base_r * 0.4f : base_r;
            float g1 = t_val ? std::min(1.0f, base_g * 0.5f + 0.7f) : base_g;
            float b1 = t_val ? base_b * 0.4f : base_b;
            row[p1_x * 3 + 0] = static_cast<uint8_t>(std::round(b1 * 255.0f));
            row[p1_x * 3 + 1] = static_cast<uint8_t>(std::round(g1 * 255.0f));
            row[p1_x * 3 + 2] = static_cast<uint8_t>(std::round(r1 * 255.0f));

            // Panel 2: Prediction Overlay (Cyan tint)
            size_t p2_x = 2 * W + x;
            float r2 = p_val ? base_r * 0.3f : base_r;
            float g2 = p_val ? std::min(1.0f, base_g * 0.4f + 0.8f) : base_g;
            float b2 = p_val ? std::min(1.0f, base_b * 0.4f + 0.9f) : base_b;
            row[p2_x * 3 + 0] = static_cast<uint8_t>(std::round(b2 * 255.0f));
            row[p2_x * 3 + 1] = static_cast<uint8_t>(std::round(g2 * 255.0f));
            row[p2_x * 3 + 2] = static_cast<uint8_t>(std::round(r2 * 255.0f));

            // Panel 3: Error Difference Map (TP=Green, FP=Red, FN=Yellow)
            size_t p3_x = 3 * W + x;
            uint8_t er_b = 30, er_g = 30, er_r = 35;
            if (t_val && p_val) {
                // True Positive: Bright Green
                er_b = 30; er_g = 220; er_r = 30;
            } else if (!t_val && p_val) {
                // False Positive: Crimson Red
                er_b = 40; er_g = 40; er_r = 230;
            } else if (t_val && !p_val) {
                // False Negative: Amber Orange
                er_b = 20; er_g = 180; er_r = 255;
            } else {
                // True Negative: Dark muted image
                uint8_t bg = static_cast<uint8_t>(std::round(base_g * 60.0f));
                er_b = er_g = er_r = bg;
            }
            row[p3_x * 3 + 0] = er_b;
            row[p3_x * 3 + 1] = er_g;
            row[p3_x * 3 + 2] = er_r;
        }
        std::fwrite(row.data(), 1, row_padded, out);
    }
    std::fclose(out);

    return true;
}

} // namespace soar::data
