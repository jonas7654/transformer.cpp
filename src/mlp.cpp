#include "../include/mlp.h"

mlp::mlp(const size_t number_of_layers, const size_t layer_sizes[], const size_t  batch_size, const bool use_one_hot) {
    num_layers = number_of_layers - 1; // Exclude input layer
    this->context_size = layer_sizes[0]; // store the context_size for better code readabillity
    this->output_size = layer_sizes[num_layers];
    this->batch_size = batch_size;
    this->use_one_hot = use_one_hot;

    this->layer_weights = new Matrix*[num_layers]; this->layer_biases = new Matrix*[num_layers];
    for (size_t i = 1; i < number_of_layers ; i++) {
      // Allocate weights 
      Matrix* weights = new Matrix(layer_sizes[i - 1], layer_sizes[i], true);
      layer_weights[i - 1] = weights;

      Matrix* biases = new Matrix(layer_sizes[i], 1, true); // TODO: Dimensions
      layer_biases[i - 1] = biases;
    }
}

mlp::~mlp() {
  for (size_t i = 0; i < num_layers; i++) {
    delete layer_weights[i];
    delete layer_biases[i];
  }
  delete[] layer_weights;
  delete[] layer_biases;
}


void mlp::print() const {
    for (size_t i = 0; i < num_layers; i++) {
        if (i == num_layers - 1) {
          std::cout << "Output layer: \n";
          std::cout << "Weights: \n";
          layer_weights[i]->print();
          std::cout << "bias \n";
          layer_biases[i]->print();
          continue;
        }

      std::cout << "hidden Layer " << i + 1 << std::endl;
        std::cout << "Weights: \n";
        layer_weights[i]->print(); // Print weights for layer i
        std::cout << "bias \n";
        layer_biases[i]->print();
        std::cout << std::endl;
    }
    std::cout << std::endl;
} 
Matrix* mlp::forward(Matrix* input) {
  assert(input->n_cols == layer_weights[0]->n_rows);

  for(size_t i = 0; i < num_layers; i++) {
    input = *input * layer_weights[i];
    input = input->add_bias(layer_biases[i]);

    if (i < num_layers - 1) {
      input = input->relu();
    } 
    else {
      if (use_one_hot) {
        input = input->softmax();
      }
      else {
        input = input->relu();
      }
    }
  }
  return input;
}

void mlp::update(double& lr) {
  for (int i = num_layers - 1; i >= 0; i--) {

    layer_weights[i]->gradDescent(lr);
    layer_biases[i]->gradDescent(lr);
    // Reset gradients
    layer_biases[i]->zeroGrad();
    layer_weights[i]->zeroGrad(); 
  }
}

Matrix* mlp::mse_loss(Matrix *y_pred, Matrix *y_true) {
  if (use_one_hot) {
    assert(y_true->n_cols == output_size);
    assert(y_true->n_rows == batch_size);
  }

  size_t total_elements = batch_size * output_size;
  
  Matrix* diff = *y_pred - y_true;
  Matrix* diff_squared = diff->square();

  double sum_squared_errors = cblas_dasum(total_elements, diff_squared->_data, 1);
  double mse = sum_squared_errors / total_elements;
  
  Matrix* loss = new Matrix(1, 1, false);
  loss->fill(mse);
  loss->childs.insert(diff_squared);
  
  loss->_backward = [loss, diff_squared, total_elements] () {
    const double scale = 1.0 / (total_elements);
    for (size_t i = 0; i < total_elements; i++) {
      diff_squared->_gradient[i] +=  scale * loss->_gradient[0];
    }
  };
  
  return loss;
}

void mlp::train(Matrix *x, Matrix *y, double lr = 0.001, size_t epochs = 5, bool verbose = false, Matrix* test_x = nullptr, Matrix* test_y = nullptr) {
  assert(x->n_rows == y->n_rows);
    
  Matrix* output;
  Matrix* loss;

  size_t x_n_rows = x->n_rows;
  size_t y_n_rows = y->n_rows;
  
  
  // Save the x and y datas start ptr value in order to reset them later 
  // I modify the ptr during batching to avoid memory copying. 
  double* x_data_start_ptr = x->_data;
  double* y_data_start_ptr = y->_data;

  // verbose helper
  float epoch_loss;
  size_t total_processed_files;
  size_t total_samples = y->n_rows;
  float test_accuracy;
  
  // Check if total_samples is divisible by batch_size
  bool has_partial_batch =  total_samples % batch_size == 0;

  // Do not delete the input data in DeleteGraph
  x->isPersistent = true;
  y->isPersistent = true;

  for(size_t e = 0; e < epochs; e++) {
    total_processed_files = 0;
    epoch_loss = 0.0;
    for (size_t batch = 0; batch < x_n_rows; batch += batch_size) {

      y->batch_subset(batch, batch_size);
      x->batch_subset(batch, batch_size);
      Matrix* y_one_hot = one_hot(y);
    
      // Forward pass
      output = forward(x); 
      loss = mse_loss(output, y_one_hot);

      // Backpropagation
      loss->backward();
      update(lr);
      epoch_loss += loss->at(0, 0);
      

      loss->resetVisited();
      loss->deleteGraph();

      x->n_rows = x_n_rows;
      x->_data = x_data_start_ptr;
      y->n_rows = y_n_rows;
      y->_data = y_data_start_ptr;

      if (verbose) {
        total_processed_files += batch_size;

        std::cout << "Train Epoch: " << e <<" [" << total_processed_files << "/" << x->n_rows << "] " <<
          " Loss: " << epoch_loss / (batch + batch_size)<< std::endl;
      }
    }
    double avg_loss = epoch_loss / total_samples;
    if (verbose) {
      std::cout << "Train loss: " << avg_loss << std::endl;
    } 
  }

  x->n_rows = x_n_rows;
  y->n_rows = y_n_rows;

  x->isPersistent = false;
  y->isPersistent = false;
}

Matrix* mlp::one_hot(Matrix* x) {
  // I assume that data has dimensions B, 1 for categorical data.
  // => for each batch there is one right indice over the output_size
  assert(x->n_cols == 1);
  assert(x->n_rows == batch_size);
  // Note that a Matrix is initialised with zeros
  Matrix* one_hot_matrix = new Matrix(batch_size, output_size, false);
  
  for (size_t i = 0; i < batch_size; i++) {
    size_t non_zero_entry = static_cast<size_t>(x->at(i,0)); 
    
    // the entry must be within output bound
    assert(non_zero_entry < output_size);

    one_hot_matrix->at(i, non_zero_entry) = 1.0;
  }

  return one_hot_matrix;
}


void mlp::predict(Matrix *input) {
  assert(input->n_cols == layer_weights[0]->n_rows);
  size_t* predicted_index = new size_t[batch_size];

  Matrix* softmax_output = forward(input);

  #pragma omp parallel for
  for (size_t row = 0; row < input->n_rows; row++) {
    double highest_prob = 0;
    for (size_t col = 0 ; col < output_size; col++) {
      if (softmax_output->_data[row * output_size + col] > highest_prob) {
        highest_prob = softmax_output->_data[row * output_size + col];
        predicted_index[row] = col;
      }
    }
  }

  // For now just print the predictions. Need to address this later.
  for (size_t i = 0; i < input->n_rows; i++) {
    std::cout << predicted_index[i] << std::endl;
  }

  delete[] predicted_index;
  // TODO: this probably deletes input
  softmax_output->deleteGraph();
}


// Compute the cross entropy loss between the logits and the targets
// y_pred must be the logits (no softmax)
Matrix* mlp::cross_entropy_loss(Matrix* y_pred, Matrix* y_true) {
  double* y_pred_data = y_pred->_data;
  double* y_true_data = y_true->_data;

  Matrix* loss = new Matrix(1,1,false);
  
  // For each row
  // TODO: This can be optimized
  double ce_loss = 0;
  for (size_t i = 0; i < batch_size; i++) {
    for (size_t j = 0; j < output_size; j++) {
      ce_loss += std::log(std::max(y_pred_data[i * output_size + j],1e-6)) * y_true_data[i * output_size + j];
    }
  }
  ce_loss *= -(1.0/batch_size);

  loss->fill(ce_loss);
  loss->childs.insert(y_pred);
  loss->op = "cross_entropy_loss";
  
  loss->_backward = [y_pred, y_true, loss] () {
    for (size_t i = 0; i < y_pred->n_rows; i++) {
      for (size_t j = 0; j < y_pred->n_cols; j++) {
        y_pred->_gradient[i * y_true->n_cols + j] += y_true->_data_at(i * y_true->n_cols + j) / y_pred->_data_at(i * y_true->n_cols + j);
      }
    }
  };

  return loss;
}

