#!/bin/bash

build_dir="../build"

cd src
clang++ -Wall -o "$build_dir/main" main.cpp value_matrix.cpp nn.cpp -lopenblas
clang++ -Wall -O3 -o "$build_dir/mnist_training" mnist.cpp value_matrix.cpp nn.cpp ../util/mnist_parser.cpp -lopenblas

exit
