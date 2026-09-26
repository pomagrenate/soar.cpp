#include <soar/optim/adamw.hpp>
#include <soar/optim/sgd.hpp>
#include <soar/utils/ema.hpp>
#include <soar/tensor/tensor.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace soar;
using namespace soar::optim;

void test_adamw_parity() {
    std::cout << "[Test] AdamW Mathematical Parity against PyTorch..." << std::endl;

    auto p = Tensor::create({1}, true);
    p->data()[0] = 1.0f;

    AdamWOptions opt;
    opt.lr = 1e-3f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.eps = 1e-8f;
    opt.weight_decay = 1e-2f;

    AdamW optimizer({p}, opt);

    // Provide gradient g = 0.5
    auto g = Tensor::create({1}, false);
    g->data()[0] = 0.5f;
    p->set_grad(g);

    optimizer.step();

    // PyTorch ground truth calculation:
    // step 1:
    // p_wd = 1.0 * (1 - 1e-3 * 1e-2) = 0.99999
    // m = 0.1 * 0.5 = 0.05
    // v = 0.001 * 0.25 = 0.00025
    // denom = sqrt(0.00025) / sqrt(0.001) + 1e-8 = 0.5 + 1e-8
    // step_size = 0.001 / 0.1 = 0.01
    // delta = 0.01 * (0.05 / 0.5) = 0.001
    // p = 0.99999 - 0.001 = 0.99899
    float expected = 0.99899f;
    float actual = p->data()[0];
    float diff = std::abs(actual - expected);

    std::cout << "  AdamW Step 1: actual = " << actual << ", expected = " << expected << ", delta = " << diff << std::endl;
    assert(diff < 1e-6f);
}

void test_sgd_parity() {
    std::cout << "[Test] SGD Momentum & Weight Decay Parity against PyTorch..." << std::endl;

    auto p = Tensor::create({1}, true);
    p->data()[0] = 2.0f;

    SGDOptions opt;
    opt.lr = 0.1f;
    opt.momentum = 0.9f;
    opt.weight_decay = 0.01f;

    SGD optimizer({p}, opt);

    // Step 1: grad = 0.4
    auto g1 = Tensor::create({1}, false);
    g1->data()[0] = 0.4f;
    p->set_grad(g1);
    optimizer.step();

    // PyTorch expected:
    // d_p = 0.4 + 0.01 * 2.0 = 0.42
    // buf = 0.42
    // p = 2.0 - 0.1 * 0.42 = 1.958
    float expected1 = 1.958f;
    float actual1 = p->data()[0];
    (void)expected1;
    (void)actual1;
    assert(std::abs(actual1 - expected1) < 1e-6f);

    // Step 2: grad = 0.2
    auto g2 = Tensor::create({1}, false);
    g2->data()[0] = 0.2f;
    p->set_grad(g2);
    optimizer.step();

    // PyTorch expected:
    // d_p = 0.2 + 0.01 * 1.958 = 0.21958
    // buf = 0.9 * 0.42 + 0.21958 = 0.378 + 0.21958 = 0.59758
    // p = 1.958 - 0.1 * 0.59758 = 1.898242
    float expected2 = 1.898242f;
    float actual2 = p->data()[0];
    float diff2 = std::abs(actual2 - expected2);
    std::cout << "  SGD Step 2: actual = " << actual2 << ", expected = " << expected2 << ", delta = " << diff2 << std::endl;
    assert(diff2 < 1e-6f);
}

void test_gradient_clipping() {
    std::cout << "[Test] Gradient Norm Clipping..." << std::endl;
    auto p1 = Tensor::create({2}, true);
    p1->data()[0] = 3.0f;
    p1->data()[1] = 4.0f;

    auto g = Tensor::create({2}, false);
    g->data()[0] = 3.0f;
    g->data()[1] = 4.0f; // norm = sqrt(3^2 + 4^2) = 5.0
    p1->set_grad(g);

    AdamW optimizer({p1});
    float norm = optimizer.clip_grad_norm(2.5f);
    (void)norm;
    assert(std::abs(norm - 5.0f) < 1e-5f);

    // After clipping with max_norm=2.5: scale = 2.5 / 5.0 = 0.5
    // g[0] = 1.5, g[1] = 2.0 -> new norm = 2.5
    float new_norm = std::sqrt(g->data()[0] * g->data()[0] + g->data()[1] * g->data()[1]);
    (void)new_norm;
    assert(std::abs(new_norm - 2.5f) < 1e-5f);
    std::cout << "  Passed gradient clipping by global L2 norm." << std::endl;
}

int main() {
    std::cout << "=== Running SOAR Optimizer Parity Tests ===" << std::endl;
    test_adamw_parity();
    test_sgd_parity();
    test_gradient_clipping();
    std::cout << "ALL OPTIMIZER TESTS PASSED!" << std::endl;
    return 0;
}
