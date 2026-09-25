#include <soar/losses/losses.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace soar;
using namespace soar::losses;

void test_bce_and_dice() {
    std::cout << "[TEST] Testing BCEWithLogitsLoss & DiceLoss...\n";
    auto logits = Tensor::randn({1, 64, 64}, 0.0f, 1.0f, true);
    auto targets = Tensor::zeros({1, 64, 64});
    for (size_t i = 0; i < targets->numel(); i += 2) targets->data()[i] = 1.0f;

    BCEWithLogitsLoss bce(1.0f, 3.0f);
    auto l_bce = bce.forward(logits, targets);
    assert(!std::isnan(l_bce->item()) && !std::isinf(l_bce->item()) && l_bce->item() > 0.0f);

    logits->zero_grad();
    l_bce->backward();
    assert(logits->grad() != nullptr);
    assert(!std::isnan(logits->grad()->data()[0]));

    DiceLoss dice(1.0f, 1.0f);
    logits->zero_grad();
    auto l_dice = dice.forward(logits, targets);
    assert(!std::isnan(l_dice->item()) && !std::isinf(l_dice->item()) && l_dice->item() > 0.0f);

    l_dice->backward();
    assert(!std::isnan(logits->grad()->data()[0]));

    std::cout << "  -> BCE & Dice passed\n";
}

void test_focal_and_tversky() {
    std::cout << "[TEST] Testing FocalLoss & TverskyLoss...\n";
    auto logits = Tensor::randn({1, 64, 64}, 0.0f, 1.0f, true);
    auto targets = Tensor::zeros({1, 64, 64});
    for (size_t i = 0; i < targets->numel(); i += 4) targets->data()[i] = 1.0f;

    FocalLoss focal(1.0f, 2.0f, 0.25f);
    logits->zero_grad();
    auto l_focal = focal.forward(logits, targets);
    assert(!std::isnan(l_focal->item()) && !std::isinf(l_focal->item()));
    l_focal->backward();
    assert(!std::isnan(logits->grad()->data()[0]));

    TverskyLoss tversky(1.0f, 0.5f, 0.5f, 1.0f);
    logits->zero_grad();
    auto l_tversky = tversky.forward(logits, targets);
    assert(!std::isnan(l_tversky->item()) && !std::isinf(l_tversky->item()));
    l_tversky->backward();
    assert(!std::isnan(logits->grad()->data()[0]));

    std::cout << "  -> Focal & Tversky passed\n";
}

void test_boundary_sdf_and_cldice() {
    std::cout << "[TEST] Testing BoundaryDistLoss (SDF) & CLDiceLoss...\n";
    auto logits = Tensor::randn({1, 64, 64}, 0.0f, 1.0f, true);
    auto targets = Tensor::zeros({1, 64, 64});
    // Draw a box foreground
    for (size_t y = 20; y < 44; ++y) {
        for (size_t x = 20; x < 44; ++x) {
            targets->data()[y * 64 + x] = 1.0f;
        }
    }

    // SDF verification
    auto sdf = compute_sdf_2d(targets);
    assert(sdf->dim(1) == 64 && sdf->dim(2) == 64);
    // Center of box should be negative distance (inside foreground)
    assert(sdf->data()[32 * 64 + 32] < 0.0f);
    // Corner outside box should be positive distance (in background)
    assert(sdf->data()[5 * 64 + 5] > 0.0f);

    BoundaryDistLoss bdist(1.0f);
    logits->zero_grad();
    auto l_bdist = bdist.forward(logits, targets);
    assert(!std::isnan(l_bdist->item()) && !std::isinf(l_bdist->item()));
    l_bdist->backward();
    assert(!std::isnan(logits->grad()->data()[0]));

    // Soft skeleton & CLDice verification
    auto skel = soft_skeletonize(targets, 3);
    assert(skel->dim(1) == 64 && skel->dim(2) == 64);

    CLDiceLoss cldice(1.0f, 3, 1.0f);
    logits->zero_grad();
    auto l_cldice = cldice.forward(logits, targets);
    assert(!std::isnan(l_cldice->item()) && !std::isinf(l_cldice->item()));
    l_cldice->backward();
    assert(!std::isnan(logits->grad()->data()[0]));

    std::cout << "  -> BoundaryDist & CLDice passed\n";
}

void test_composite_segmentation_loss() {
    std::cout << "[TEST] Testing full SegmentationLoss with warmup scheduling...\n";
    auto logits = Tensor::randn({1, 64, 64}, 0.0f, 1.0f, true);
    auto targets = Tensor::zeros({1, 64, 64});
    for (size_t y = 16; y < 48; ++y) {
        for (size_t x = 16; x < 48; ++x) {
            targets->data()[y * 64 + x] = 1.0f;
        }
    }

    auto region = std::make_shared<DiceBCELoss>(1.0f, 1.0f, 1.0f, 3.0f, 1.0f);
    auto boundary = std::make_shared<BoundaryDistLoss>(0.5f);
    auto structure = std::make_shared<CLDiceLoss>(0.2f, 3, 1.0f);

    SegmentationLoss loss_fn(region, boundary, structure, 1.0f, 1.0f, 3.0f, 0.2f, 5);

    // Epoch 0 (cldice warmup should be 0)
    LossBreakdown b0;
    auto l0 = loss_fn.forward(logits, targets, b0, 0);
    assert(b0.cldice == 0.0f);
    assert(b0.region > 0.0f);
    assert(b0.boundary != 0.0f);

    // Epoch 6 (cldice warmed up)
    LossBreakdown b6;
    logits->zero_grad();
    auto l6 = loss_fn.forward(logits, targets, b6, 6);
    assert(b6.cldice > 0.0f);
    assert(!std::isnan(l6->item()));

    l6->backward();
    assert(logits->grad() != nullptr);
    assert(!std::isnan(logits->grad()->data()[0]));

    std::cout << "  -> SegmentationLoss passed\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  SOAR Scientific Losses Verification   \n";
    std::cout << "========================================\n";

    test_bce_and_dice();
    test_focal_and_tversky();
    test_boundary_sdf_and_cldice();
    test_composite_segmentation_loss();

    std::cout << "\n>>> ALL SCIENTIFIC LOSS TESTS PASSED! <<<\n";
    return 0;
}
