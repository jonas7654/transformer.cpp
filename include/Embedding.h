#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "value_matrix.h"
#include <cstddef>

class Embedding {
  private:
  Matrix* embedding_matrix;
  Matrix* embedding_weight_matrix;

  size_t batch_size;
  size_t context_size;
  size_t n_embd;

  public:
  Embedding(size_t& vocab_size, size_t& n_embd, size_t& batch_size, size_t& context_size);
  Matrix* forward(Matrix* x);
  void update(float& lr);
};

#endif // !EMBEDDING_H
