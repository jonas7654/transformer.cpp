
# Compiler and flags
WANDBCPP_DIR = /home/jv/GitHub/NeuralNetwork_from_scratch/wandb-cpp
WANDBCPP_INCLUDE = $(WANDBCPP_DIR)/src 
WANDBCPP_LIB = $(WANDBCPP_DIR)/build

CXX = clang++
PYTHON_INCLUDES := $(shell python3-config --includes)
PYTHON_LIBS := $(shell python3-config --ldflags)


CXXFLAGS = -Wall -O3 -std=c++20 -I/home/jv/miniconda3/include/python3.12 \
           -I/home/jv/miniconda3/lib/python3.12/site-packages/numpy/_core/include \
           -I$(WANDBCPP_INCLUDE) -I$(WANDBCPP_DIR)

LDFLAGS = -lopenblas $(PYTHON_LIBS)

# Directories
SRC_DIR = src
UTIL_DIR = util
BUILD_DIR = build

# Source files
SRC = $(SRC_DIR)/mnist.cpp $(SRC_DIR)/value_matrix.cpp $(SRC_DIR)/nn.cpp $(UTIL_DIR)/mnist_parser.cpp
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Output binary
TARGET = $(BUILD_DIR)/mnist_training
run: $(TARGET)
	@echo "Running neural network..."
	@./$(TARGET)

# Default target: build the program
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

.PHONY: all clean

