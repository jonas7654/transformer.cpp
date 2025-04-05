#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "value_matrix.h"
#include "mlp.h"

class transformer {
  private:
   size_t vocab_size;
   size_t context_size;
   size_t batch_size;
   size_t n_layer;
   size_t n_embd;    
   size_t n_heads;    
   float dropout;    
   float lr;
  
  public:
  transformer(const size_t vocab_size,
              const size_t context_size,
              const size_t batch_size,
              const size_t n_layer,
              const size_t n_embd,
              const size_t n_heads,
              const float dropout,
              const float lr);

  ~transformer();
}; 
#endif // !TRANSFORMER_H
