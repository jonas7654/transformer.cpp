#include "value_matrix.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

const double EPSILON = 1e-5;

bool is_close(double a, double b, double tol = EPSILON) {
    return abs(a - b) < tol;
}

template<typename Func>
double numerical_gradient(Matrix* input, Func compute_output, size_t i, size_t j, double eps = 1e-5) {
    double original = input->at(i, j);
    
    input->at(i, j) = original + eps;
    Matrix* out_plus = compute_output();
    double loss_plus = out_plus->sum();
    delete out_plus;
    
    input->at(i, j) = original - eps;
    Matrix* out_minus = compute_output();
    double loss_minus = out_minus->sum();
    delete out_minus;
    
    input->at(i, j) = original;
    
    return (loss_plus - loss_minus) / (2 * eps);
}

void test_addition() {
    cout << "Testing Addition..." << endl;
    Matrix* a = new Matrix(2, 2, false);
    Matrix* b = new Matrix(2, 2, false);
    a->fill(2.0);
    b->fill(3.0);
    
    Matrix* c = *a + b;
    
    assert(c->at(0,0) == 5.0);
    assert(c->at(1,1) == 5.0);
    
    c->fill(1.0);
    c->backward();
    
    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<2; j++) {
            assert(is_close(a->grad_at(i,j), 1.0));
            assert(is_close(b->grad_at(i,j), 1.0));
        }
    }
    
    c->deleteGraph();
    cout << "Addition Test Passed!" << endl;
}

void test_subtraction() {
    cout << "Testing Subtraction..." << endl;
    Matrix* a = new Matrix(2, 2, false);
    Matrix* b = new Matrix(2, 2, false);
    a->fill(5.0);
    b->fill(3.0);
    
    Matrix* c = *a - b;
    
    assert(c->at(0,0) == 2.0);
    assert(c->at(1,1) == 2.0);
    
    c->fill(1.0);
    c->backward();
    
    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<2; j++) {
            assert(is_close(a->grad_at(i,j), 1.0));
            assert(is_close(b->grad_at(i,j), -1.0));
        }
    }
    
    c->deleteGraph();
    cout << "Subtraction Test Passed!" << endl;
}

void test_matmul() {
    cout << "Testing Matrix Multiplication..." << endl;
    Matrix* a = new Matrix(2, 3, false);
    Matrix* b = new Matrix(3, 2, false);
    a->fill(1.0);
    b->fill(2.0);
    
    Matrix* c = *a * b;
    
    assert(c->at(0,0) == 6.0);
    assert(c->at(1,1) == 6.0);
    
    c->backward();
    
    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<3; j++) {
            assert(is_close(a->grad_at(i,j), 4.0));
        }
    }
    for (size_t i=0; i<3; i++) {
        for (size_t j=0; j<2; j++) {
            assert(is_close(b->grad_at(i,j), 2.0));
        }
    }
    
    auto compute_output = [a, b]() { return (*a) * (b); };
    double num_grad = numerical_gradient(a, compute_output, 0, 0);
    assert(is_close(num_grad, 4.0));
    
    num_grad = numerical_gradient(b, compute_output, 0, 0);
    assert(is_close(num_grad, 2.0));
    
    c->deleteGraph();
    cout << "Matrix Multiplication Test Passed!" << endl;
}

void test_relu() {
    cout << "Testing ReLU..." << endl;
    Matrix* a = new Matrix(2, 2, false);
    a->at(0,0) = 2.0; a->at(0,1) = -1.0;
    a->at(1,0) = 0.0; a->at(1,1) = -3.0;
    
    Matrix* c = a->relu();
    
    assert(c->at(0,0) == 2.0);
    assert(c->at(0,1) == 0.0);
    assert(c->at(1,0) == 0.0);
    assert(c->at(1,1) == 0.0);
    
    c->backward();
    
    assert(is_close(a->grad_at(0,0), 1.0));
    assert(is_close(a->grad_at(0,1), 0.0));
    assert(is_close(a->grad_at(1,0), 0.0));
    assert(is_close(a->grad_at(1,1), 0.0));
    
    auto compute_output = [a]() { return a->relu(); };
    double num_grad = numerical_gradient(a, compute_output, 0, 0);
    assert(is_close(num_grad, 1.0));
    
    num_grad = numerical_gradient(a, compute_output, 0, 1);
    assert(is_close(num_grad, 0.0));
    
    c->deleteGraph();
    cout << "ReLU Test Passed!" << endl;
}

void test_sigmoid() {
    cout << "Testing Sigmoid..." << endl;
    Matrix* a = new Matrix(1, 1, false);
    a->at(0,0) = 0.0;
    
    Matrix* c = a->sigmoid();
    assert(is_close(c->at(0,0), 0.5));
    
    c->grad_at(0,0) = 1.0;
    c->backward();
    assert(is_close(a->grad_at(0,0), 0.25));
    
    auto compute_output = [a]() { return a->sigmoid(); };
    double num_grad = numerical_gradient(a, compute_output, 0, 0);
    assert(is_close(num_grad, 0.25));
    
    c->deleteGraph();
    cout << "Sigmoid Test Passed!" << endl;
}


void test_softmax() {
    cout << "Testing Row-wise Softmax..." << endl;
    
    // Test case 1: Basic row-wise softmax
    {
        Matrix* a = new Matrix(2, 3, false); // 2 rows, 3 columns
        a->at(0,0) = 1.0; a->at(0,1) = 2.0; a->at(0,2) = 3.0;
        a->at(1,0) = -1.0; a->at(1,1) = 0.0; a->at(1,2) = 1.0;

        Matrix* sm = a->softmax();
        
        // Verify each row sums to 1
        for (size_t i = 0; i < 2; i++) {
            double row_sum = 0.0;
            for (size_t j = 0; j < 3; j++) {
                row_sum += sm->at(i,j);
            }
            assert(is_close(row_sum, 1.0));
        }
        
        // Verify some known values
        double denom1 = exp(1.0) + exp(2.0) + exp(3.0);
        assert(is_close(sm->at(0,0), exp(1.0)/denom1));
        assert(is_close(sm->at(0,1), exp(2.0)/denom1));
        assert(is_close(sm->at(0,2), exp(3.0)/denom1));

        // Test backpropagation
        // Set some arbitrary output gradients
        sm->grad_at(0,0) = 0.1;
        sm->grad_at(0,1) = 0.2;
        sm->grad_at(0,2) = 0.3;
        sm->grad_at(1,0) = 0.4;
        sm->grad_at(1,1) = 0.5;
        sm->grad_at(1,2) = 0.6;
        
        sm->backward();
        
        // Verify gradients numerically
        auto compute_softmax = [a]() { return a->softmax(); };
        
        // Test gradient for a specific element
        size_t test_row = 0;
        size_t test_col = 1;
        double original = a->at(test_row, test_col);
        
        // Perturb positive
        a->at(test_row, test_col) = original + EPSILON;
        Matrix* sm_plus = compute_softmax();
        double sum_plus = sm_plus->sum();
        delete sm_plus;
        
        // Perturb negative
        a->at(test_row, test_col) = original - EPSILON;
        Matrix* sm_minus = compute_softmax();
        double sum_minus = sm_minus->sum();
        delete sm_minus;
        
        // Reset
        a->at(test_row, test_col) = original;
        
        // Numerical gradient
        double num_grad = (sum_plus - sum_minus) / (2 * EPSILON);
        
        // Analytical gradient
        double analy_grad = a->grad_at(test_row, test_col);
        
        assert(is_close(num_grad, analy_grad, 1e-4));
        
        sm->deleteGraph();
    }
    
    // Test case 2: Numerical stability (large values)
    {
        Matrix* a = new Matrix(1, 3, false);
        a->at(0,0) = 1000.0; a->at(0,1) = 1001.0; a->at(0,2) = 1002.0;
        
        Matrix* sm = a->softmax();
        
        // Shouldn't be NaN or Inf
        for (size_t j = 0; j < 3; j++) {
            assert(!isnan(sm->at(0,j)));
            assert(!isinf(sm->at(0,j)));
        }
        
        // Should still sum to 1
        double row_sum = 0.0;
        for (size_t j = 0; j < 3; j++) {
            row_sum += sm->at(0,j);
        }
        assert(is_close(row_sum, 1.0));
        
        sm->deleteGraph();
    }
    
    cout << "Softmax Tests Passed!" << endl;
}
void test_square() {
    cout << "Testing Square..." << endl;
    Matrix* a = new Matrix(2, 2, false);
    a->fill(3.0);
    
    Matrix* c = a->square();
    assert(c->at(0,0) == 9.0);
    
    c->backward();
   
    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<2; j++) {
            assert(is_close(a->grad_at(i,j), 6.0));
        }
    }
    
    auto compute_output = [a]() { return a->square(); };
    double num_grad = numerical_gradient(a, compute_output, 0, 0);
    assert(is_close(num_grad, 6.0));
    
    c->deleteGraph();
    cout << "Square Test Passed!" << endl;
}

void test_add_bias() {
    cout << "Testing Add Bias..." << endl;
    Matrix* a = new Matrix(2, 3, false);
    Matrix* b = new Matrix(3, 1, false);
    a->fill(1.0);
    b->fill(2.0);
    
    Matrix* c = a->add_bias(b);
    assert(c->at(0,0) == 3.0);
    assert(c->at(1,1) == 3.0);
    
    c->backward();
    
    for (size_t i=0; i<2; i++) {
        for (size_t j=0; j<3; j++) {
            assert(is_close(a->grad_at(i,j), 1.0));
        }
    }
    for (size_t j=0; j<3; j++) {
        assert(is_close(b->grad_at(j,0), 2.0));
    }
    
    auto compute_output = [a, b]() { return a->add_bias(b); };
    double num_grad = numerical_gradient(a, compute_output, 0, 0);
    assert(is_close(num_grad, 1.0));
    
    num_grad = numerical_gradient(b, compute_output, 0, 0);
    assert(is_close(num_grad, 2.0));
    
    c->deleteGraph();
    cout << "Add Bias Test Passed!" << endl;
}

int main() {
    test_addition();
    test_subtraction();
    test_matmul();
    test_relu();
    test_sigmoid();
    test_softmax();
    test_square();
    test_add_bias();
    return 0;


    x->setIsPersistent(true);
    y->setIsPersistent(true);
  
    // Manually create a small network
    // Input is 4 , 2
    Matrix* W1 = new Matrix(2, 1, true);
    Matrix* B1 = new Matrix(1,1, true);
  
    double lr = 0.1;
    for (int i = 0; i < 100*10000000000; i++) {
      Matrix* Z1 = *x *W1;
      Matrix* A1 = Z1->add_bias(B1)->sigmoid();
      Matrix* loss = mse_loss(A1, y, 4, 1);
  
      loss->backward();
      W1->gradDescent(lr);
      B1->gradDescent(lr);
      W1->zeroGrad();
      B1->zeroGrad();
      W1->visited = false;
      B1->visited = false;
      loss->print();
      loss->deleteGraph();
    }
  
    Matrix* Z1 = *x *W1;
    Matrix* A1 = Z1->add_bias(B1)->sigmoid();
    A1->print();
  
  
  
    return 0;
}

