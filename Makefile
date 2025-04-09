CXX = g++
CXXFLAGS = -std=c++17 -Wall
SOURCES = main.cpp ./src/Perceptron/mlp.cpp ./src/Perceptron/Output/output_layer.cpp ./src/Perceptron/Hidden/hidden_layer.cpp ./src/Perceptron/BasePerceptron/perceptron.cpp ./src/Datasets/datasets.cpp ./src/Datasets/Parameters/params.cpp ./util/readcsv.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = mlp_program

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

build: $(TARGET)
	rm -f $(OBJECTS)

clean:
	rm -f $(OBJECTS) $(TARGET)