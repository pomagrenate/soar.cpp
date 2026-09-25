#include <soar/nn/soar_model.hpp>
#include <soar/engine/trainer.hpp>
#include <soar/engine/predictor.hpp>
#include <soar/losses/losses.hpp>
#include <soar/optim/adamw.hpp>
#include <soar/api/c_api.h>

#include <iostream>
#include <cassert>
#include <filesystem>

void test_soar_model_variants() {
    std::cout << "[TEST] Testing SOAR model variant construction..." << std::endl;
    soar::nn::SOARModel nano(1, 1, soar::nn::ModelVariant::Nano);
    std::cout << "  Nano parameters:   " << nano.parameter_count() << std::endl;
    assert(nano.parameter_count() > 200000);

    soar::nn::SOARModel small(1, 1, soar::nn::ModelVariant::Small);
    std::cout << "  Small parameters:  " << small.parameter_count() << std::endl;
    assert(small.parameter_count() > nano.parameter_count());

    soar::nn::SOARModel medium(1, 1, soar::nn::ModelVariant::Medium);
    std::cout << "  Medium parameters: " << medium.parameter_count() << std::endl;
    assert(medium.parameter_count() > small.parameter_count());

    std::cout << "  -> test_soar_model_variants PASSED" << std::endl;
}

void test_forward_backward_training() {
    std::cout << "[TEST] Testing forward pass, autograd, and AdamW training step..." << std::endl;
    auto model = std::make_shared<soar::nn::SOARModel>(1, 1, soar::nn::ModelVariant::Nano);

    // Use a spatial dimension that is divisible by 32 (e.g. 64x64 or 128x128 for rapid unit testing)
    constexpr size_t H = 64;
    constexpr size_t W = 64;

    auto img = soar::Tensor::randn({1, H, W}, 0.5f, 0.1f);
    auto target = soar::Tensor::zeros({1, H, W});
    for (size_t i = 0; i < target->numel(); i += 2) target->data()[i] = 1.0f;

    soar::losses::CompositeLoss loss_fn(0.5f, 0.5f);
    soar::optim::AdamWOptions opt_opts;
    opt_opts.lr = 1e-3f;
    soar::optim::AdamW optimizer(model->parameters(), opt_opts);

    soar::engine::Trainer trainer(model, loss_fn, optimizer);

    auto m1 = trainer.train_step(img, target);
    std::cout << "  Step 1: Loss = " << m1.loss << ", Dice = " << m1.dice_score << ", IoU = " << m1.iou_score << std::endl;

    auto m2 = trainer.train_step(img, target);
    std::cout << "  Step 2: Loss = " << m2.loss << ", Dice = " << m2.dice_score << ", IoU = " << m2.iou_score << std::endl;

    assert(!std::isnan(m1.loss) && !std::isinf(m1.loss));
    assert(!std::isnan(m2.loss) && !std::isinf(m2.loss));
    std::cout << "  -> test_forward_backward_training PASSED" << std::endl;
}

void test_predictor_and_rle() {
    std::cout << "[TEST] Testing Predictor and RLE generation..." << std::endl;
    auto model = std::make_shared<soar::nn::SOARModel>(1, 1, soar::nn::ModelVariant::Nano);
    soar::engine::Predictor predictor(model, 0.5f);

    constexpr size_t H = 64;
    constexpr size_t W = 64;
    auto img = soar::Tensor::randn({1, H, W}, 0.5f, 0.1f);

    auto res = predictor.predict(img);
    assert(res.probabilities->dim(1) == H);
    assert(res.probabilities->dim(2) == W);
    assert(res.binary_mask.size() == H * W);

    std::cout << "  Predicted foreground pixels: " << res.foreground_pixels << std::endl;
    std::cout << "  RLE length: " << res.rle_string.size() << " chars" << std::endl;
    std::cout << "  -> test_predictor_and_rle PASSED" << std::endl;
}

void test_weight_serialization() {
    std::cout << "[TEST] Testing weight serialization and loading..." << std::endl;
    auto m1 = std::make_shared<soar::nn::SOARModel>(1, 1, soar::nn::ModelVariant::Nano);

    std::string tmp_path = (std::filesystem::temp_directory_path() / "test_model.soar").string();
    m1->save_weights(tmp_path);

    auto m2 = std::make_shared<soar::nn::SOARModel>(1, 1, soar::nn::ModelVariant::Nano);
    m2->load_weights(tmp_path);

    // Verify parameter values match
    auto p1 = m1->named_parameters();
    auto p2 = m2->named_parameters();
    assert(p1.size() == p2.size());

    for (size_t i = 0; i < p1.size(); ++i) {
        assert(p1[i].first == p2[i].first);
        const float* d1 = p1[i].second->data();
        const float* d2 = p2[i].second->data();
        size_t n = p1[i].second->numel();
        for (size_t j = 0; j < n; ++j) {
            assert(d1[j] == d2[j]);
        }
    }

    std::filesystem::remove(tmp_path);
    std::cout << "  -> test_weight_serialization PASSED" << std::endl;
}

void test_c_api() {
    std::cout << "[TEST] Testing C API exports..." << std::endl;
    void* handle = soar_create_model(1, 1, 0); // Nano
    assert(handle != nullptr);

    size_t params = soar_get_parameter_count(handle);
    assert(params > 0);
    std::cout << "  C API Model parameter count: " << params << std::endl;

    std::vector<float> img(64 * 64, 0.5f);
    std::vector<float> msk(64 * 64, 1.0f);
    float loss = 0, dice = 0, iou = 0;

    int res = soar_train_step(handle, img.data(), msk.data(), 1, 64, 64, 1e-3f, &loss, &dice, &iou);
    assert(res == 0);
    std::cout << "  C API Train Step Loss: " << loss << ", Dice: " << dice << std::endl;

    std::vector<float> out_probs(64 * 64);
    std::vector<uint8_t> out_mask(64 * 64);
    char rle[1024]{0};
    res = soar_predict(handle, img.data(), 1, 64, 64, 0.5f, out_probs.data(), out_mask.data(), rle, sizeof(rle));
    assert(res == 0);

    soar_destroy_model(handle);
    std::cout << "  -> test_c_api PASSED" << std::endl;
}

int main() {
    try {
        std::cout << "=========================================================" << std::endl;
        std::cout << "  SOAR Full C++ Engine Verification (Train + Infer + API)" << std::endl;
        std::cout << "=========================================================" << std::endl;

        test_soar_model_variants();
        test_forward_backward_training();
        test_predictor_and_rle();
        test_weight_serialization();
        test_c_api();

        std::cout << std::endl;
        std::cout << ">>> ALL FULL C++ ENGINE TESTS PASSED! <<<" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
