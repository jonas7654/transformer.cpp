#ifndef VALUE_MATRIX_H
#define VALUE_MATRIX_H

#include <cassert>
#include <cstddef>
#include <unordered_set>
#include <string>
#include <functional>
#include <cblas.h> // OpenBLAS header
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <stack>

class Embedding;

class Matrix {
typedef std::mt19937 rng_type;
friend class mlp;
friend class Embedding;

private:
  double* _data;
  double* _gradient;

  
  std::string op;
  std::unordered_set<Matrix*> childs;
  std::function<void()> _backward;

  void topological_sort(std::vector<Matrix*>& topo_vector);
  void collect_nodes(std::vector<Matrix*>& collected);
  void batch_subset(size_t start_row, size_t end_row);

  bool isLearnable;
  bool isPersistent;
  bool visited;
public:
  size_t n_rows;
  size_t n_cols;
  size_t n;

  Matrix(size_t n_rows, size_t n_cols, bool isLearnable);
  Matrix(Matrix* other);
  ~Matrix();

  double& at(size_t i, size_t j);
  double& grad_at(size_t i, size_t j);
  double& _data_at(size_t i);
  double sum() const;
  
  void fill(double _value);
  void rand();
  void print();
  void printGradient();
  void backward();
  void zeroGrad();
  void resetVisited();
  void deleteGraph();
  void clear_children();
  
  Matrix* operator = (Matrix* other);

  // The idea is to only work with heap allocated instances of matrix
  Matrix* operator +(Matrix* other);
  Matrix* operator -(Matrix* other);
  Matrix* operator *(Matrix* other);
  Matrix* operator /(Matrix* other);

  Matrix* relu ();
  Matrix* sigmoid();
  Matrix* softmax();
  Matrix* add_bias(Matrix* other);
  Matrix* square();
  Matrix* log();

  void scale(double& d);
  void gradDescent(double& lr);
  void setIsPersistent(bool b);
  
  Matrix* select_row(size_t row);
  Matrix* select_col(size_t col);
  void tranpose();
  Matrix* slice(size_t row_start_idx, size_t row_end_idx, size_t col_start_idx, size_t col_end_idx);   
};


#endif // !VALUE_MATRIX_H
