#include <soar/data/preprocess.hpp>
#include <soar/memory/palloc_allocator.hpp>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_map>

namespace soar::data {

void normalize_01(TensorPtr& img, float p_low, float p_high) {
    size_t n = img->numel();
    if (n == 0) return;
    memory::PallocVector<float> copy_data(n);
    std::memcpy(copy_data.data(), img->data(), n * sizeof(float));

    size_t idx_low = static_cast<size_t>(std::clamp(std::round((p_low / 100.0f) * float(n - 1)), 0.0f, float(n - 1)));
    size_t idx_high = static_cast<size_t>(std::clamp(std::round((p_high / 100.0f) * float(n - 1)), 0.0f, float(n - 1)));

    std::nth_element(copy_data.begin(), copy_data.begin() + idx_low, copy_data.end());
    float lo = copy_data[idx_low];

    std::nth_element(copy_data.begin() + idx_low, copy_data.begin() + idx_high, copy_data.end());
    float hi = copy_data[idx_high];

    float denom = std::max(hi - lo, 1e-6f);
    float* d = img->data();
    for (size_t i = 0; i < n; ++i) {
        d[i] = std::clamp((d[i] - lo) / denom, 0.0f, 1.0f);
    }
}

void normalize_min_max(TensorPtr& img) {
    size_t n = img->numel();
    if (n == 0) return;
    float* d = img->data();
    float min_val = d[0];
    float max_val = d[0];
    for (size_t i = 1; i < n; ++i) {
        if (d[i] < min_val) min_val = d[i];
        if (d[i] > max_val) max_val = d[i];
    }
    float denom = std::max(max_val - min_val, 1e-6f);
    for (size_t i = 0; i < n; ++i) {
        d[i] = (d[i] - min_val) / denom;
    }
}

void normalize_zscore(TensorPtr& img) {
    size_t n = img->numel();
    if (n == 0) return;
    float* d = img->data();
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += d[i];
    float mean = static_cast<float>(sum / double(n));
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float diff = d[i] - mean;
        sum_sq += diff * diff;
    }
    float std_dev = static_cast<float>(std::sqrt(sum_sq / double(n)));
    float denom = std::max(std_dev, 1e-6f);
    for (size_t i = 0; i < n; ++i) {
        d[i] = (d[i] - mean) / denom;
    }
}

void morphological_close(const uint8_t* binary, uint8_t* out, size_t H, size_t W, int kernel_size) {
    std::vector<std::pair<int, int>> se;
    int rad = kernel_size / 2;
    float r_sq = float(rad * rad);
    for (int dy = -rad; dy <= rad; ++dy) {
        for (int dx = -rad; dx <= rad; ++dx) {
            if (float(dy * dy + dx * dx) <= r_sq + 0.5f) {
                se.emplace_back(dy, dx);
            }
        }
    }

    memory::PallocVector<uint8_t> dilated(H * W, 0);

    #pragma omp parallel for
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            bool hit = false;
            for (const auto& [dy, dx] : se) {
                int ny = static_cast<int>(y) + dy;
                int nx = static_cast<int>(x) + dx;
                if (ny >= 0 && ny < static_cast<int>(H) && nx >= 0 && nx < static_cast<int>(W)) {
                    if (binary[ny * W + nx] > 0) {
                        hit = true;
                        break;
                    }
                }
            }
            dilated[y * W + x] = hit ? 1 : 0;
        }
    }

    #pragma omp parallel for
    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            bool all_ones = true;
            for (const auto& [dy, dx] : se) {
                int ny = static_cast<int>(y) + dy;
                int nx = static_cast<int>(x) + dx;
                if (ny >= 0 && ny < static_cast<int>(H) && nx >= 0 && nx < static_cast<int>(W)) {
                    if (dilated[ny * W + nx] == 0) {
                        all_ones = false;
                        break;
                    }
                } else {
                    all_ones = false;
                    break;
                }
            }
            out[y * W + x] = all_ones ? 1 : 0;
        }
    }
}

std::vector<uint8_t> morphological_close(const std::vector<uint8_t>& binary, size_t H, size_t W, int kernel_size) {
    std::vector<uint8_t> result(H * W);
    morphological_close(binary.data(), result.data(), H, W, kernel_size);
    return result;
}

namespace {

struct DisjointSet {
    memory::PallocVector<int32_t> parent;

    explicit DisjointSet(size_t n) : parent(n, 0) {}

    int find(int i) {
        int root = i;
        while (parent[root] > 0) {
            root = parent[root];
        }
        int curr = i;
        while (curr != root && parent[curr] > 0) {
            int next = parent[curr];
            parent[curr] = root;
            curr = next;
        }
        return root;
    }

    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra != rb) {
            if (parent[ra] > parent[rb]) std::swap(ra, rb);
            parent[ra] += parent[rb];
            parent[rb] = ra;
        }
    }

    int size(int root) {
        return -parent[find(root)];
    }
};

} // anonymous namespace

void filter_connected_components(const uint8_t* binary, uint8_t* out, size_t H, size_t W, size_t min_area) {
    size_t total = H * W;
    DisjointSet ds(total);

    for (size_t i = 0; i < total; ++i) {
        if (binary[i] > 0) {
            ds.parent[i] = -1; // size 1
        }
    }

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            size_t idx = y * W + x;
            if (binary[idx] == 0) continue;

            const int dys[4] = {-1, -1, -1, 0};
            const int dxs[4] = {-1,  0,  1, -1};
            for (int k = 0; k < 4; ++k) {
                int ny = static_cast<int>(y) + dys[k];
                int nx = static_cast<int>(x) + dxs[k];
                if (ny >= 0 && ny < static_cast<int>(H) && nx >= 0 && nx < static_cast<int>(W)) {
                    size_t n_idx = static_cast<size_t>(ny) * W + static_cast<size_t>(nx);
                    if (binary[n_idx] > 0) {
                        ds.unite(static_cast<int>(idx), static_cast<int>(n_idx));
                    }
                }
            }
        }
    }

    for (size_t i = 0; i < total; ++i) {
        if (binary[i] > 0) {
            int root = ds.find(static_cast<int>(i));
            out[i] = (static_cast<size_t>(ds.size(root)) >= min_area) ? 1 : 0;
        } else {
            out[i] = 0;
        }
    }
}

std::vector<uint8_t> filter_connected_components(const std::vector<uint8_t>& binary, size_t H, size_t W, size_t min_area) {
    std::vector<uint8_t> result(H * W);
    filter_connected_components(binary.data(), result.data(), H, W, min_area);
    return result;
}

std::vector<std::vector<uint8_t>> extract_connected_components(const uint8_t* binary, size_t H, size_t W, size_t min_area) {
    size_t total = H * W;
    DisjointSet ds(total);

    for (size_t i = 0; i < total; ++i) {
        if (binary[i] > 0) {
            ds.parent[i] = -1;
        }
    }

    for (size_t y = 0; y < H; ++y) {
        for (size_t x = 0; x < W; ++x) {
            size_t idx = y * W + x;
            if (binary[idx] == 0) continue;

            const int dys[4] = {-1, -1, -1, 0};
            const int dxs[4] = {-1,  0,  1, -1};
            for (int k = 0; k < 4; ++k) {
                int ny = static_cast<int>(y) + dys[k];
                int nx = static_cast<int>(x) + dxs[k];
                if (ny >= 0 && ny < static_cast<int>(H) && nx >= 0 && nx < static_cast<int>(W)) {
                    size_t n_idx = static_cast<size_t>(ny) * W + static_cast<size_t>(nx);
                    if (binary[n_idx] > 0) {
                        ds.unite(static_cast<int>(idx), static_cast<int>(n_idx));
                    }
                }
            }
        }
    }

    std::unordered_map<int, size_t> root_to_comp_idx;
    std::vector<std::vector<uint8_t>> components;

    for (size_t i = 0; i < total; ++i) {
        if (binary[i] > 0) {
            int root = ds.find(static_cast<int>(i));
            if (static_cast<size_t>(ds.size(root)) >= min_area) {
                auto it = root_to_comp_idx.find(root);
                if (it == root_to_comp_idx.end()) {
                    size_t comp_idx = components.size();
                    components.emplace_back(total, 0);
                    root_to_comp_idx[root] = comp_idx;
                    components[comp_idx][i] = 1;
                } else {
                    components[it->second][i] = 1;
                }
            }
        }
    }

    return components;
}

} // namespace soar::data
