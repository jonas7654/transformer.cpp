#ifndef TENSOR_H
#define TENSOR_H

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

class Tensor {
typedef std::mt19937 rng_type;
friend class mlp;
friend class Embedding;

private:
  double* _data;
  double* _gradient;
  
  std::string op;
  std::unordered_set<Tensor*> childs;
  std::function<void()> _backward;

  void topological_sort(std::vector<Tensor*>& topo_vector);
  void collect_nodes(std::vector<Tensor*>& collected);
  void batch_subset(size_t start_row, size_t end_row);

  bool isLearnable;
  bool isPersistent;
  bool visited;

public:
  size_t n;
  size_t* shape;

  Tensor(size_t dim, size_t* shape, bool isLearnable); // row * D1 * D2 + D2 * j + k
  Tensor(Tensor* other);
  ~Tensor();

  double& at(size_t* indices);
  double& grad_at(size_t* indices);
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
  
  Tensor* operator = (Tensor* other);

  // The idea is to only work with heap allocated instances of matrix
  Tensor* operator +(Tensor* other);
  Tensor* operator -(Tensor* other);
  Tensor* operator *(Tensor* other);
  Tensor* operator /(Tensor* other);

  Tensor* relu ();
  Tensor* sigmoid();
  Tensor* softmax();
  Tensor* add_bias(Tensor* other);
  Tensor* square();
  Tensor* log();

  void scale(double& d);
  void gradDescent(double& lr);
  void setIsPersistent(bool b);
  
  Tensor* select_row(size_t row);
  Tensor* select_col(size_t col);
  void tranpose();
  Tensor* slice(size_t row_start_idx, size_t row_end_idx, size_t col_start_idx, size_t col_end_idx);   
};


#endif
