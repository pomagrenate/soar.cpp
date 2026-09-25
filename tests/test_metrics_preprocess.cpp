#include <soar/utils/metrics.hpp>
#include <soar/utils/ema.hpp>
#include <soar/data/preprocess.hpp>
#include <soar/nn/soar_model.hpp>

#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

void test_metrics() {
    std::cout << "[TEST] Testing Metrics (IoU, Dice, Panoptic Quality)..." << std::endl;

    std::vector<uint8_t> m1 = {1, 1, 1, 1, 0, 0, 0, 0};
    std::vector<uint8_t> m2 = {0, 1, 1, 1, 1, 0, 0, 0};

    // Intersection: 3, Union: 5 -> IoU = 3/5 = 0.6
    float iou = soar::utils::compute_iou(m1, m2);
    assert(std::abs(iou - 0.6f) < 1e-4f);

    // Dice: 2*3 / (4 + 4) = 6/8 = 0.75
    float dice = soar::utils::compute_dice(m1, m2);
    assert(std::abs(dice - 0.75f) < 1e-4f);

    // Panoptic Quality
    std::vector<std::vector<uint8_t>> preds = {m1, m2};
    std::vector<std::vector<uint8_t>> gts = {m1};
    auto pq = soar::utils::compute_panoptic_quality(preds, gts, 0.5f);
    assert(pq.tp == 1);
    assert(pq.fp == 1);
    assert(pq.fn == 0);
    assert(pq.sq >= 0.99f); // m1 matches m1 perfectly (IoU = 1.0)
    assert(pq.rq > 0.6f);
    assert(pq.pq > 0.6f);

    std::cout << "  -> Metrics tests PASSED (IoU=" << iou << ", Dice=" << dice
              << ", PQ=" << pq.pq << ", SQ=" << pq.sq << ", RQ=" << pq.rq << ")" << std::endl;
}

void test_preprocess() {
    std::cout << "[TEST] Testing Preprocessing & Morphology..." << std::endl;

    // 1. MinMax Normalization
    auto t1 = soar::Tensor::create({1, 4, 4});
    float* d1 = t1->data();
    for (size_t i = 0; i < 16; ++i) d1[i] = static_cast<float>(i) * 10.0f; // 0 to 150
    soar::data::normalize_min_max(t1);
    assert(std::abs(d1[0] - 0.0f) < 1e-5f);
    assert(std::abs(d1[15] - 1.0f) < 1e-5f);
    assert(std::abs(d1[7] - (70.0f / 150.0f)) < 1e-5f);

    // 2. ZScore Normalization
    auto t2 = soar::Tensor::create({1, 4, 4});
    float* d2 = t2->data();
    for (size_t i = 0; i < 16; ++i) d2[i] = static_cast<float>(i);
    soar::data::normalize_zscore(t2);
    float mean = 0.0f;
    for (size_t i = 0; i < 16; ++i) mean += d2[i];
    mean /= 16.0f;
    assert(std::abs(mean) < 1e-4f);

    // 3. Percentile [0, 1] Normalization
    auto t3 = soar::Tensor::create({1, 100, 1});
    float* d3 = t3->data();
    for (size_t i = 0; i < 100; ++i) d3[i] = static_cast<float>(i);
    soar::data::normalize_01(t3, 5.0f, 95.0f);
    assert(d3[0] == 0.0f);
    assert(d3[99] == 1.0f);

    // 4. Connected component filtering
    constexpr size_t H = 8, W = 8;
    std::vector<uint8_t> binary(H * W, 0);
    // Large component of size 9 (3x3)
    for (size_t y = 1; y <= 3; ++y) {
        for (size_t x = 1; x <= 3; ++x) {
            binary[y * W + x] = 1;
        }
    }
    // Small noise pixel at (6, 6) of size 1
    binary[6 * W + 6] = 1;

    auto filtered = soar::data::filter_connected_components(binary, H, W, 4);
    assert(filtered[6 * W + 6] == 0); // noise removed
    assert(filtered[2 * W + 2] == 1); // large component kept

    auto comps = soar::data::extract_connected_components(binary, H, W, 4);
    assert(comps.size() == 1); // only 1 component with area >= 4

    // 5. Morphological closing: close a 1-pixel hole inside a 5x5 block
    std::vector<uint8_t> hole_block(H * W, 0);
    for (size_t y = 1; y <= 5; ++y) {
        for (size_t x = 1; x <= 5; ++x) {
            hole_block[y * W + x] = 1;
        }
    }
    hole_block[3 * W + 3] = 0; // hole
    auto closed = soar::data::morphological_close(hole_block, H, W, 3);
    assert(closed[3 * W + 3] == 1); // hole filled!

    std::cout << "  -> Preprocessing & Morphology tests PASSED" << std::endl;
}

void test_model_ema() {
    std::cout << "[TEST] Testing ModelEMA..." << std::endl;
    auto model = std::make_shared<soar::nn::SOARModel>(1, 1, soar::nn::ModelVariant::Nano);
    soar::utils::ModelEMA ema(model, 0.9f);

    auto p0 = model->parameters()[0];
    float orig_val = p0->data()[0];
    p0->data()[0] = orig_val + 10.0f; // simulate parameter update

    ema.update(model);
    float expected_ema = 0.9f * orig_val + 0.1f * (orig_val + 10.0f);
    float shadow_val = ema.shadow_params()[0]->data()[0];
    assert(std::abs(shadow_val - expected_ema) < 1e-4f);

    ema.apply_to(model);
    assert(std::abs(p0->data()[0] - expected_ema) < 1e-4f);

    std::cout << "  -> ModelEMA tests PASSED" << std::endl;
}

int main() {
    std::cout << "=========================================================" << std::endl;
    std::cout << "  SOAR Metrics, Preprocessing & EMA Verification Suite   " << std::endl;
    std::cout << "=========================================================" << std::endl;

    test_metrics();
    test_preprocess();
    test_model_ema();

    std::cout << std::endl;
    std::cout << ">>> ALL METRICS, PREPROCESS & EMA TESTS PASSED! <<<" << std::endl;
    return 0;
}
