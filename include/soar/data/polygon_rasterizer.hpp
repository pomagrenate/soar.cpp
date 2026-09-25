#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>

namespace soar::data {

struct Point2D {
    float x;
    float y;
};

class PolygonRasterizer {
public:
    /**
     * @brief Rasterize a polygon into an existing mask buffer using the standard even-odd scanline rule.
     * @param mask Pointer to row-major binary mask of size H x W.
     * @param H Height of mask.
     * @param W Width of mask.
     * @param poly List of 2D vertices in pixel coordinates.
     * @param value Value to fill (e.g. 1.0f).
     */
    static void rasterize(float* mask, size_t H, size_t W, const std::vector<Point2D>& poly, float value = 1.0f) {
        if (poly.size() < 3) return;

        float min_y = poly[0].y;
        float max_y = poly[0].y;
        for (const auto& pt : poly) {
            min_y = std::min(min_y, pt.y);
            max_y = std::max(max_y, pt.y);
        }

        int start_y = std::max(0, static_cast<int>(std::floor(min_y)));
        int end_y = std::min(static_cast<int>(H) - 1, static_cast<int>(std::ceil(max_y)));

        size_t num_vertices = poly.size();
        std::vector<float> node_x;
        node_x.reserve(num_vertices);

        for (int y = start_y; y <= end_y; ++y) {
            float scan_y = static_cast<float>(y) + 0.5f;
            node_x.clear();

            for (size_t i = 0, j = num_vertices - 1; i < num_vertices; j = i++) {
                float yi = poly[i].y;
                float yj = poly[j].y;
                float xi = poly[i].x;
                float xj = poly[j].x;

                if ((yi < scan_y && yj >= scan_y) || (yj < scan_y && yi >= scan_y)) {
                    float intersect_x = xi + (scan_y - yi) / (yj - yi) * (xj - xi);
                    node_x.push_back(intersect_x);
                }
            }

            std::sort(node_x.begin(), node_x.end());

            for (size_t k = 0; k + 1 < node_x.size(); k += 2) {
                int x_start = std::max(0, static_cast<int>(std::ceil(node_x[k])));
                int x_end = std::min(static_cast<int>(W) - 1, static_cast<int>(std::floor(node_x[k + 1])));

                for (int x = x_start; x <= x_end; ++x) {
                    mask[y * W + x] = value;
                }
            }
        }
    }
};

} // namespace soar::data
