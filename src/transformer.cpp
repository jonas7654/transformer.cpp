#include "../include/transformer.h"


transformer::transformer(const size_t vocab_size,
                         const size_t context_size,
                         const size_t batch_size,
                         const size_t n_layer,
                         const size_t n_embd,
                         const size_t n_heads,
                         const float dropout,
                         const float lr) 
{
  this->vocab_size = vocab_size;
  this->context_size = context_size;
  this->batch_size = batch_size;
  this->n_layer = n_layer;
  this->n_embd = n_embd;
  this->n_heads = n_heads;
  this->dropout = dropout;
  this->lr = lr;

  // Set the strides to convert n-dimensional indices to flat array indice
  
  // Initialize and mlp
  // In standard literature the hidden layers are often 4 * n_emd
  // Note that mlp takes input + hidden layers + output as layers_sizes but here I chose to 
    // only specify the hidden layers within n_layer.
  size_t layer_sizes = n_layer + 2;
  size_t mlp_layers[layer_sizes];
  mlp_layers[0] = context_size;
  for (size_t i = 1; i < n_layer; i++){
    mlp_layers[i] = 4 * n_embd;
  }
  mlp_layers[layer_sizes - 1] = vocab_size;
                  
  mlp* MLP = new mlp(layer_sizes, mlp_layers, batch_size, true);

  // ****** ******///

}

transformer::~transformer() {

}
