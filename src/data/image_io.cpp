#include <soar/data/image_io.hpp>
#include <soar/core/logging.hpp>

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

    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;

    out.write(reinterpret_cast<const char*>(file_header), 14);
    out.write(reinterpret_cast<const char*>(info_header), 40);

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
        out.write(reinterpret_cast<const char*>(row.data()), row_padded);
    }

    return true;
}

} // namespace soar::data
