#include "../include/mnist_parser.h"
#include <bits/types/FILE.h>
#include <cstring>


#define BUFFER_SIZE 10000 // How to set this properly
#define IMAGE_SIZE 784

Matrix* read_mnist(const char* s) {
  const char* FILE_PATH = nullptr;
  size_t n_images;

  if (strcmp(s, "train") == 0) {
    FILE_PATH = "/home/jv/GitHub/NeuralNetwork_from_scratch/data/MNIST/mnist_train/mnist_train.csv";
    n_images = 60000;
  }
  else if(strcmp(s, "test") == 0) {
    FILE_PATH = "/home/jv/GitHub/NeuralNetwork_from_scratch/data/MNIST/mnist_test/mnist_test.csv";
    n_images = 10000;
  }
  else {
    std::cerr << "please specify train or test \n";
    return nullptr;
  }

  if(FILE_PATH == nullptr) {
    std::cerr << "FILE_PATH is nullptr \n";
    return nullptr;
  }


  FILE* file = fopen(FILE_PATH, "r");
  if(!file) {
    perror("error opening mnist");
    return 0;
  }

  // Allocate train matrix  
  // IMAGE_SIZE + 1 since I need to extract the label as well
  Matrix* mnist_train_matrix = new Matrix(n_images, IMAGE_SIZE + 1, false);
  size_t n_cols = mnist_train_matrix->n_cols;
  
  char line[BUFFER_SIZE];
  size_t count = 0;

  // populate the matrix
  while(fgets(line, BUFFER_SIZE, file) && count < n_images) {
    char* tok = strtok(line, ",");

    // Skip empty (just to be sure)
    if (!tok) {
      continue;
    }

    // Extract the label which is saved as the first value
    mnist_train_matrix->_data_at(count * n_cols + (IMAGE_SIZE)) = (double) (atof(tok));

    // extract each comma seperated value from a single row
    for (size_t i = 0; i < IMAGE_SIZE; i++){
      tok = strtok(nullptr, ",");
      if(!tok) break;
      // Normalize data
      mnist_train_matrix->_data_at(count * n_cols + i) = (double) (atof(tok) / 255.0);
    }
    count++;
  }

  std::cout << "Read " << count << " images from " << s << " dataset" << std::endl;

  fclose(file);

  return mnist_train_matrix;
}



