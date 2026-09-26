#include <soar/autograd/engine.hpp>
#include <soar/tensor/tensor.hpp>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace soar;
using namespace soar::autograd;

// Custom AddNode for testing multi-branch DAG
struct AddNode : public AutogradNode {
    std::shared_ptr<AutogradNode> lhs;
    std::shared_ptr<AutogradNode> rhs;

    AddNode(std::shared_ptr<AutogradNode> l, std::shared_ptr<AutogradNode> r)
        : lhs(std::move(l)), rhs(std::move(r)) {}

    void backward(const TensorPtr& grad_output) override {
        if (lhs) lhs->add_grad_output(grad_output);
        if (rhs) rhs->add_grad_output(grad_output);
    }

    std::vector<std::shared_ptr<AutogradNode>> get_inputs() const override {
        return {lhs, rhs};
    }
};

void test_dag_dependency_counting() {
    std::cout << "[Test] DAG In-Degree Dependency Computation..." << std::endl;
    // Graph:
    // Root -> Add1 -> (LeafA, LeafB)
    //      -> Add2 -> (LeafB, LeafC)
    // LeafB is shared between Add1 and Add2!
    auto param_a = Tensor::create({2}, true);
    auto param_b = Tensor::create({2}, true);
    auto param_c = Tensor::create({2}, true);

    auto acc_a = std::make_shared<AccumulateGradNode>(param_a);
    auto acc_b = std::make_shared<AccumulateGradNode>(param_b);
    auto acc_c = std::make_shared<AccumulateGradNode>(param_c);

    auto add1 = std::make_shared<AddNode>(acc_a, acc_b);
    auto add2 = std::make_shared<AddNode>(acc_b, acc_c);
    auto root = std::make_shared<AddNode>(add1, add2);

    Engine engine;
    std::unordered_map<AutogradNode*, size_t> dependencies;
    std::unordered_map<AutogradNode*, std::vector<std::shared_ptr<AutogradNode>>> graph_edges;

    engine.compute_dependencies(root.get(), dependencies, graph_edges);

    // Assert in-degrees
    assert(dependencies[add1.get()] == 1);
    assert(dependencies[add2.get()] == 1);
    assert(dependencies[acc_a.get()] == 1);
    assert(dependencies[acc_b.get()] == 2); // acc_b is referenced by both add1 and add2
    assert(dependencies[acc_c.get()] == 1);

    std::cout << "  Passed in-degree calculation (acc_b has in-degree 2)." << std::endl;
}

void test_topological_execution_and_gradient_accumulation() {
    std::cout << "[Test] Engine Execution & In-Place AccumulateGrad..." << std::endl;

    auto w = Tensor::create({2}, true);
    w->data()[0] = 3.0f;
    w->data()[1] = -2.0f;

    auto acc_w = std::make_shared<AccumulateGradNode>(w);
    // Double dependency on w: Root -> (acc_w, acc_w)
    auto root = std::make_shared<AddNode>(acc_w, acc_w);

    auto root_grad = Tensor::create({2}, false);
    root_grad->data()[0] = 1.0f;
    root_grad->data()[1] = 1.0f;

    Engine engine;

    // Step 1: Backward pass 1
    engine.execute(root, root_grad);

    assert(w->grad() != nullptr);
    // Root contributes root_grad to both branches, so w->grad should be 2.0f
    assert(std::abs(w->grad()->data()[0] - 2.0f) < 1e-6f);
    assert(std::abs(w->grad()->data()[1] - 2.0f) < 1e-6f);

    float* grad_ptr_first_pass = w->grad()->data();

    // Step 2: Virtual gradient accumulation (micro-batch 2 without zeroing grad)
    auto root2 = std::make_shared<AddNode>(acc_w, acc_w);
    engine.execute(root2, root_grad);

    // Gradient buffer pointer should be PRESERVED (zero reallocation)
    assert(w->grad()->data() == grad_ptr_first_pass);

    // Gradient value accumulated in-place: 2.0 + 2.0 = 4.0
    assert(std::abs(w->grad()->data()[0] - 4.0f) < 1e-6f);
    assert(std::abs(w->grad()->data()[1] - 4.0f) < 1e-6f);

    std::cout << "  Passed topological execution and zero-allocation gradient accumulation." << std::endl;
}

int main() {
    std::cout << "=== Running SOAR Autograd Engine Tests ===" << std::endl;
    test_dag_dependency_counting();
    test_topological_execution_and_gradient_accumulation();
    std::cout << "ALL AUTOGRAD ENGINE TESTS PASSED!" << std::endl;
    return 0;
}
