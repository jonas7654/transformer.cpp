#include "../include/Embedding.h"

Embedding::Embedding(size_t& vocab_size, size_t& n_embd, size_t& batch_size, size_t& context_size) {
  this->embedding_matrix = new Matrix(vocab_size, n_embd, true);
  this->embedding_weight_matrix = new Matrix(n_embd, n_embd, true);
  this->batch_size = batch_size;
  this->context_size = context_size;
  this->n_embd = n_embd;
}

Matrix* Embedding::forward(Matrix *x) {
  // Input has shape (B, context_size)
  // Input consists of Tokens
  // Output has shape B, context_size, n_embd
  
  // Update the current Embedding Matrix 
  Matrix* temp = this->embedding_matrix;

  embedding_matrix = *embedding_matrix * embedding_weight_matrix;
  embedding_weight_matrix->isLearnable = true;
  delete temp;


  Matrix* selected_embeddings = new Matrix(context_size, n_embd, false);
  for (size_t batch = 0; batch < batch_size ; batch++) {
    for (size_t i = 0; i < context_size; i++) {
      size_t token = (size_t) x->_data[batch * context_size + i];
      selected_embeddings->_data[batch * context_size + token]  = embedding_matrix->_data[batch * context_size * i];
    }
  }

}
