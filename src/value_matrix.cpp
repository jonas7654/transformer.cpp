#include "../include/value_matrix.h"
#include <algorithm>
#include <cassert>
#include <cblas.h> 
#include <cmath>

Matrix::Matrix(size_t rows, size_t cols, bool isLearnable) { 
  this->n_rows = rows;
  this->n_cols = cols;
  this->isPersistent = false;
  double* _array = new double[n_rows * n_cols];
  double* _grad = new double[n_rows * n_cols];
  this->_data = _array;
  this->_gradient = _grad;
  this->n = n_rows * n_cols;
  this->op = "NONE";
  this->isLearnable = isLearnable;
  this->visited = false;

  std::unordered_set<Matrix*> childs;
  this->childs = childs;

  // fill _gradient with zeros
  cblas_dscal(this->n, 0.0, _gradient, 1);
  
  // Randomly initialize the matrix
  if (isLearnable) {
    rand();
  }
  else {
    cblas_dscal(this->n, 0.0, _data, 1);
  }

};

Matrix::Matrix(Matrix* other) {
  if(other == this) {
    return;
  }

  double* _array = new double[other->n_rows * other->n_cols];
  double* _grad = new double[other->n_rows * other->n_cols];
  for(size_t i = 0; i < other->n_cols * other->n_rows; i++) {
    _array[i] = other->_data[i];
  }
  for(size_t i = 0; i < other->n_cols * other->n_rows; i++) {
    _grad[i] = other->_gradient[i];
  }


  this->_data = _array;
  this->_gradient = _grad;

  this->n_rows = other->n_rows;
  this->n_cols = other->n_cols;
  this->n = other->n;
  this->op = other->op;
  this->isLearnable = other->isLearnable;
  this->visited = other->visited;
}


Matrix::~Matrix() {
  delete[] _data;
  delete[] _gradient;
}

double& Matrix::at(size_t i, size_t j) {
  assert(i < n_rows && j < n_cols);
  return _data[i * n_cols + j];
}

double& Matrix::grad_at(size_t i, size_t j) {
  assert(i < n_rows && j < n_cols);
  return _gradient[i * n_cols + j];
}

void Matrix::fill(double _value){
  for (size_t i = 0; i < n_cols * n_rows; i++) {
    _data[i] = _value;
  }
}

double& Matrix::_data_at(size_t i) {
  assert(i < n_rows * n_cols);
  return _data[i];
}

Matrix* Matrix::softmax() {
  // Apply the softmax row-wise: S = e^x / sum(e^x)
  // TODO: numerical stability
  Matrix* result = new Matrix(this->n_rows, this->n_cols, false);
  
   for (size_t i = 0; i < n_rows; i++) {
    // Find maximum value in the row for numerical stability
    double max_val = _data[i * n_cols];
    for (size_t j = 1; j < n_cols; j++) {
      if (_data[i * n_cols + j] > max_val) {
        max_val = _data[i * n_cols + j];
      }
    }

    // Compute exponents and sum
    double row_sum = 0.0;
    for (size_t j = 0; j < n_cols; j++) {
      double shifted_exp = std::exp(_data[i * n_cols + j] - max_val);
      result->_data[i * n_cols + j] = shifted_exp;
      row_sum += shifted_exp;
    }

    // Normalize the row
    double row_sum_inv = 1.0 / row_sum;
    cblas_dscal(n_cols, row_sum_inv, result->_data + (i * n_cols), 1);
  }

  result->op = "softmax";
  result->childs.insert(this);
  
  // Apply gradient row-wise
  // Jacobian: J = diag(S_i) - S_i^T * S_i => N x N matrix for each row
  result->_backward = [this, result]() {
  
    size_t N = result->n_cols; // Number of classes
    size_t B = result->n_rows;

    // Temporary storage for Jacobian (N x N)
    double* J = new double[N * N];

    for (size_t i = 0; i < B; i++) {
        double* s = &result->_data[i * N]; // Softmax output for i-th sample
        double* dL_dS = &result->_gradient[i * N]; // Incoming grad for i-th sample
        double* dL_dX = &this->_gradient[i * N]; // Output grad for i-th sample

        // Compute J = diag(s) - s^T s
        // 1. J = -s^T s (outer product)
        cblas_dger(
            CblasRowMajor,
            N, N,               // Dimensions
            -1.0,               // Scaling factor
            s, 1,               // s (row vector)
            s, 1,               // s (row vector)
            J, N                // Output J (C x C)
        );

        // 2. J += diag(s)
        for (size_t j = 0; j < N; j++) {
            J[j * N + j] += s[j];
        }

        // Compute dL_dX = dL_dS @ J (since J is symmetric)
        // result grad is then a 1 x N for each row to update
        cblas_dsymv(
            CblasRowMajor,
            CblasUpper,         // Use upper triangle (J is symmetric)
            N,                 // Dimension
            1.0,               // Scaling factor
            J, N,              // Jacobian matrix
            dL_dS, 1,          // Incoming gradient (row vector)
            1.0,               // Accumulate into dL_dX
            dL_dX, 1           // Output gradient
        );

    }

    delete[] J;
};
  return result;
}

void Matrix::rand() {
  // Initialize random number generator
  std::random_device rd;  // Seed for the random number engine
  std::mt19937 gen(rd()); // Mersenne Twister engine
  std::uniform_real_distribution<double> dist(-0.5, 0.5); // Distribution for random doubles
  for (size_t i = 0; i < n_cols * n_rows; i++) {
    _data[i] = dist(gen);
  }
}

void Matrix::setIsPersistent(bool b) {
  isPersistent = b;
}


void Matrix::print() {
  std::cout << "Operation: " << this->op << std::endl;

  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_cols; j++) {
      std::cout << this->at(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void Matrix::printGradient() {
  std::cout << "Gradients: "<< std::endl;

  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_cols; j++) {
      std::cout << this->grad_at(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void Matrix::scale(double& d) {
   cblas_dscal(n, d, _data, 1);
}

void Matrix::gradDescent(double &lr) {
// This calculates result = this - lr * _gradient  
  cblas_daxpy(n_rows * n_cols,
              -lr, // Minus here for gradient descent
              this->_gradient,
              1,
              this->_data,
              1);
}

Matrix* Matrix::relu() {
  Matrix* result = new Matrix(n_rows, n_cols, false);

  cblas_dcopy(n_rows * n_cols, this->_data, 1, result->_data, 1);

    
  // NOTE:
  // The CPU cache is crucial for performance. When we access memory in a linear, sequential fashion (which is what cblas_scopy does), the CPU is much more likely to keep the data in cache, which leads to faster access times.The CPU cache is crucial for performance. When we access memory in a linear, sequential fashion (which is what cblas_dcopy does), the CPU is much more likely to keep the data in cache, which leads to faster access times. cblas_dcopy(n, _data, 1, result->_data, 1);

  for (size_t i = 0; i < n; i++) {
    result->_data[i] = std::max(0.0, result->_data[i]);
  }

  result->childs.insert(this);
  result->op = "relu";
  // Note that relu() is an elementwise function (hadamard) so the derivative is also elementwise
  result->_backward = [result, this] () {
    #pragma omp parallel for
    for (size_t i = 0; i < result->n; i++) {
      if(result->_data[i] > 0){
        this->_gradient[i] += result->_gradient[i];
      }
    }
  };

  return result;
}

// TODO: use openBLAS
Matrix* Matrix::sigmoid() {
  Matrix* result = new Matrix(n_rows, n_cols, isPersistent);

  cblas_dcopy(n, this->_data, 1, result->_data, 1);

  for (size_t i = 0; i < n; i++) {
    result->_data[i] = 1 / (1 + std::exp(-result->_data[i]));
  }

  result->childs.insert(this);
  result->op = "sigmoid";
  
  // Elementwise gradient
  result->_backward = [this, result] () {
    #pragma omp parallel for
    for (size_t i = 0; i < result->n; i++) {
        double sigmoid_val = result->_data[i];  
        this->_gradient[i] += result->_gradient[i] * sigmoid_val * (1 - sigmoid_val);
    }
  };

  return result;
}

Matrix* Matrix::operator+ (Matrix* other){
  assert(this->n_rows == other->n_rows);
  assert(this->n_cols == other->n_cols);

  Matrix* result = new Matrix(n_rows, n_cols, false);
  // cblas works inplace => copy this->_data to result
  for (size_t i = 0; i < n_rows * n_cols; i++) {
    result->_data[i] = _data[i];
  }

  // This calculates result = a * x + y | x = this, y = other , a = scaling factor 
  cblas_daxpy(n_rows * n_cols,
              1.0,
              other->_data,
              1,
              result->_data,
              1);

  result->childs.insert(this);
  result->childs.insert(other);
  result->op = "+";
//  result->_backward = [this, other, result]() {
//    for (size_t i = 0; i < this->n; i++) {
//        this->_gradient[i] += result->_gradient[i];
//        other->_gradient[i] += result->_gradient[i];
//    }
//};
  result->_backward = [this, other, result] () {

    // Gradient w.r.t this
    cblas_daxpy(n_rows * n_cols,
                1.0,
                result->_gradient,
                1,
                this->_gradient,
                1);

    // Gradient w.r.t other
    cblas_daxpy(n_rows * n_cols,
                1.0,
                result->_gradient,
                1,
                other->_gradient,
                1);
  };

  return result;
};

Matrix* Matrix::operator- (Matrix* other){
  assert(this->n_rows == other->n_rows);
  assert(this->n_cols == other->n_cols);

  Matrix* result = new Matrix(n_rows, n_cols, false);

  // cblas works inplace => copy this->_data to result
  for (size_t i = 0; i < n_rows * n_cols; i++) {
    result->_data[i] = -(other->_data[i]); // take the negative of other
  }

  // This calculates result = a * x - y | x = this, y = other , a = scaling factor 
  cblas_daxpy(n_rows * n_cols,
              1.0,
              this->_data,
              1,
              result->_data,
              1);

  result->childs.insert(this);
  result->childs.insert(other);
  result->op = "-";

  result->_backward = [this, other, result] () {
    // Note: Addtition and Subtraction just pass the gradient to the childs
    // In order to accumulate gradients we need to sum the incoming gradients
    // This is why I use daxpy

    // Gradient w.r.t this
    cblas_daxpy(n_rows * n_cols,
                1.0,
                result->_gradient,
                1,
                this->_gradient,
                1);
    // Gradient w.r.t other
    cblas_daxpy(n_rows * n_cols,
                -1.0,// This is -(result->_gradient)
                result->_gradient,           
                1,
                other->_gradient,
                1);
  };

  return result;
};


Matrix* Matrix::operator* (Matrix* other){
  assert(this->n_cols == other->n_rows);
  assert(this->_data != nullptr && "Matrix A data is null!");
  assert(other->_data != nullptr && "Matrix B data is null!");

  Matrix* result = new Matrix(n_rows, other->n_cols, false);

  // BLAS MatMul
  cblas_dgemm(CblasRowMajor, // _data is row-major
              CblasNoTrans,  // Do not transpose this
              CblasNoTrans,  // Do not transpose other
              n_rows,        // rows of result
              other->n_cols, // columns of result 
              this->n_cols,  // columns of this 
              1.0,           // Scaling factor
              this->_data,   // lvalue 
              this->n_cols,  // leading dimension of this 
              other->_data,  // rvalue
              other->n_cols, // leading dimension of other
              1.0,           // Scaling for accumulation
              result->_data, // result data
              result->n_cols // leading dimension of result
              );

  result->childs.insert(this);
  result->childs.insert(other);
  result->op = "*";


result->_backward = [this, other, result]() {
  // Gradient for 'this' (X): dL/dX = dL/dZ * W^T
  // Dimensions:
  // - X (this): m x k
  // - W (other): k x n
  // - dL/dZ (result->_gradient): m x n
  // - dL/dX (this->_gradient): m x k
  
//  // Loop over X's gradient dimensions
//  for (int i = 0; i < this->n_rows; ++i) {         // m rows
//      for (int j = 0; j < this->n_cols; ++j) {     // k columns
//          double grad = 0.0;
//          
//          // Compute dot product between:
//          // - i-th row of dL/dZ (m x n)
//          // - j-th row of W (k x n)
//          for (int n = 0; n < result->n_cols; ++n) {  // n columns
//              grad += result->_gradient[i * result->n_cols + n] *  // dZ[i][n]
//                      other->_data[j * other->n_cols + n];          // W[j][n]
//          }
//          
//          this->_gradient[i * this->n_cols + j] += grad;
//      }
//  }
//
//  // Gradient for 'other' (W): dL/dW = X^T * dL/dZ
//  // Dimensions:
//  // - dL/dW (other->_gradient): k x n
//  
//  for (int j = 0; j < other->n_rows; ++j) {       // k rows
//      for (int l = 0; l < other->n_cols; ++l) {    // n columns
//          double grad = 0.0;
//          
//          // Compute dot product between:
//          // - j-th column of X (m x k)
//          // - l-th column of dL/dZ (m x n)
//          for (int i = 0; i < this->n_rows; ++i) {  // m rows
//              grad += this->_data[i * this->n_cols + j] *   // X[i][j]
//                      result->_gradient[i * result->n_cols + l];  // dZ[i][l]
//          }
//          
//          other->_gradient[j * other->n_cols + l] += grad;
//      }
//  }


  // For gradient w.r.t this (X): dL/dX = dL/dZ * W^T
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              this->n_rows, this->n_cols, result->n_cols,
              1.0,
              result->_gradient, result->n_cols,
              other->_data, other->n_cols,
              1.0,
              this->_gradient, this->n_cols);
  
  // For gradient w.r.t other (W): dL/dW = X^T * dL/dZ
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              other->n_rows, other->n_cols, this->n_rows,
              1.0,
              this->_data, this->n_cols,
              result->_gradient, result->n_cols,
              1.0,
              other->_gradient, other->n_cols);
};

  return result;
};

Matrix* Matrix::square() {
  
  Matrix* result = new Matrix(n_rows, n_cols, false);
  cblas_dcopy(n, _data, 1, result->_data, 1);
  for (size_t i = 0; i < n; i++) {
    result->_data[i] *= result->_data[i];
  }

  result->childs.insert(this);
  result->op = "square";

  result->_backward = [result, this] () {   
    //)(d(A^2) / dA = 2 * A) => out_grad * (scalar * this->_data)
    // Note that square() is an element-wise operation so the gradient applies also elementwise (hadamard product)
    double* copy_of_data = new double[result->n];
    cblas_dcopy(result->n, this->_data, 1, copy_of_data, 1);
    
    // multiply everything by 2
    cblas_dscal(this->n, 2.0, copy_of_data, 1);
    
    for (size_t i = 0; i < this->n; i++) {
      this->_gradient[i] += result->_gradient[i] * copy_of_data[i];
    }
    delete[] copy_of_data;
  };
   
  return result;
}

Matrix* Matrix::add_bias(Matrix* other) {
  assert(other->n_cols == 1 && other->n_rows == n_cols);

  Matrix* result = new Matrix(n_rows, n_cols, false);

  cblas_dcopy(n, _data, 1, result->_data, 1);
  for (size_t i = 0; i < result->n_rows; i++) {
    size_t ptr_shift = i * n_cols;
    cblas_daxpy(n_cols, 1.0, other->_data, 1, result->_data + ptr_shift, 1);
  }

  result->op = "add_bias";
  result->childs.insert(this);
  result->childs.insert(other);
    
  result->_backward = [this, other, result] () {
    cblas_daxpy(n, 1.0, result->_gradient, 1, this->_gradient, 1);


    
    const size_t cols = result->n_cols;
    for (size_t col = 0; col < cols; ++col) {
        double bias_grad = 0.0;
        // Sum gradients across all rows for this column
        for (size_t row = 0; row < result->n_rows; ++row) {
            bias_grad += result->_gradient[row * cols + col];
        }
        other->_gradient[col] += bias_grad;
    }
};

  return result;
}

// Creates a log transformed new Matrix with "this" as child 
Matrix* Matrix::log() {
  Matrix* result = new Matrix(n_rows, n_cols, isPersistent);
  cblas_dcopy(n, _data, 1, result->_data, 1);
  
  // Apply log
  for (size_t i = 0; i < n ;i++) {
    result->_data[i] = std::log(result->_data[i]);
  }

  result->childs.insert(this);
  result->op = "log";
  
  // Apply gradient elementwise
  result->_backward = [this, result] () {
    // this_grad = result_grad + this_grad;
    cblas_daxpy(n, 1.0, result->_gradient, 1, this->_gradient, 1);

    for (size_t i = 0; i < n; i++) {
          this->_gradient[i] *= 1 / result->_data[i];
    }
  };

  return result;
}

void Matrix::batch_subset(size_t start_row, size_t batch_size) {
  // Function only for the training Function
  // Allows subsetting batches using pointer arithmetic to work inplace
  _data = _data + start_row * n_cols;
  n_rows = batch_size;
}

Matrix* Matrix::select_row(size_t row) {
  return slice(row , row, 0, n_cols - 1);
}

Matrix* Matrix::select_col(size_t col) {
  return slice(0, n_rows - 1, col, col);
}

Matrix* Matrix::slice(size_t row_start_idx, size_t row_end_idx, size_t col_start_idx, size_t col_end_idx) {
  assert(row_start_idx < n_rows && row_end_idx < n_rows && row_start_idx <= row_end_idx);
  assert(col_start_idx < n_cols && col_end_idx < n_cols && col_start_idx <= col_end_idx);
  size_t new_n_rows = row_end_idx - row_start_idx + 1;
  size_t new_n_cols = col_end_idx - col_start_idx + 1;

  Matrix* sliced_matrix = new Matrix(new_n_rows, new_n_cols, false);

  // Copy sliced data row-wise
  for (size_t i = 0; i < new_n_rows; i++) {
    cblas_dcopy(new_n_cols, this->_data + (row_start_idx + i) * n_cols + col_start_idx, 1 , sliced_matrix->_data + i * new_n_cols, 1);
  }
  
  return sliced_matrix;
}

void Matrix::tranpose() {
  double* temp = new double[n];
  cblas_dcopy(n, _data, 1, temp, 1);

  for (size_t i = 0; i < n_rows; i++) {
    for (size_t j = 0; j < n_cols; j++) {
      _data[j * n_rows+ i] = temp[i * n_cols + j];
    }  
  }
  delete[] temp;
  size_t rows = n_rows;
  this->n_rows = this->n_cols;
  this->n_cols = rows;
}

void Matrix::topological_sort(std::vector<Matrix*> &topo_vector){
  // check if the current Node was already visited.
  // It would probably more efficient to store a bool _visited within Value :TODO
  if (this->visited) {
    return;
  }
  this->visited = true;
  for (Matrix* child : childs) {
    child->topological_sort(topo_vector);
  }
  // Add the first node to the vector (This will be called at at the first node)
  topo_vector.push_back(this);
}

void Matrix::backward() {
  // df/df = 1.0
  std::fill(_gradient, _gradient + n_rows * n_cols, 1.0);


  std::vector<Matrix*> topo_vector;
  topological_sort(topo_vector);
  
  // Traverse the collected nodes in reverse
  for (std::vector<Matrix*>::iterator it_end = topo_vector.end() ; it_end != topo_vector.begin();){
    it_end--;
    if ((*it_end)->_backward) {
      (*it_end)->_backward();
    }
  }
}


void Matrix::zeroGrad() {
  cblas_dscal(n_rows * n_cols, 0.0, _gradient, 1);
}

void Matrix::collect_nodes(std::vector<Matrix*>& collected) { 
    if (this->visited || isPersistent || isLearnable){
      return;
  } 
    this->visited = true;
  // Traverse children first
    for (Matrix* child : childs) {
        child->collect_nodes(collected);
    }
    collected.push_back(this);
}

void Matrix::deleteGraph() {
    std::vector<Matrix*> nodes;
    collect_nodes(nodes);

    // Delete in reverse order (children first, parents last)
    for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
        delete *it;  
  }
}


void Matrix::resetVisited() {
  if (this->visited) {
    this->visited = false;
    for (Matrix* child : childs) {
      child->resetVisited();
    }
  }
}

void Matrix::clear_children() {
  childs.clear(); // Remove all child references
}

double Matrix::sum() const {
    double total = 0.0;
    for (size_t i = 0; i < n; i++) {
        total += _data[i];
    }
    return total;
}



