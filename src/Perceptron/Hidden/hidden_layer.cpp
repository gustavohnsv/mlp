#include "hidden_layer.h"

H_layer::H_layer(double learning_rate, int size, int size_conn, int size_weights) 
    : learning_rate(learning_rate), size_conn(size_conn), size_weights(size_weights) {
    this->perceptrons.reserve(size);
    for (int i = 0; i < size; i++) {
        this->perceptrons.emplace_back(size_conn, size_weights);
    }
}

H_layer::~H_layer() {}

Perceptron &H_layer::getPerceptron(size_t index_perceptron) {
    if (index_perceptron >= perceptrons.size()) {
        throw std::out_of_range("Index out of bounds in getPerceptron");
    }
    return this->perceptrons[index_perceptron];
}

std::vector<Perceptron> &H_layer::getPerceptrons() {
    return this->perceptrons;
}

double H_layer::getLearningRate() {
    return this->learning_rate;
}

void H_layer::updateWeight(int index_perceptron, int index_data, int index_weight, O_layer* output_layer, Datasets data) {
    double delta = delta_function(index_perceptron, index_data, output_layer, data);
    double input_value = data.getUniqueParams(index_data).getUniqueX(index_weight);
    double current_weight = this->perceptrons[index_perceptron].getWeight(index_weight);
    double new_weight = current_weight + this->getLearningRate() * delta * input_value;
    this->perceptrons[index_perceptron].updateWeight(new_weight, index_weight);
}

void H_layer::updateBias(int index_perceptron, int index_data, O_layer* output_layer, Datasets data) {
    double current_bias = this->perceptrons[index_perceptron].getBias();
    double delta = delta_function(index_perceptron, index_data, output_layer, data);
    double new_bias = current_bias + this->getLearningRate() * delta;
    this->perceptrons[index_perceptron].updateBias(new_bias);
}

double H_layer::delta_function(int index_perceptron, int index_data, O_layer* output_layer, Datasets data) {
    double out = output(index_perceptron, index_data, data);
    double target = data.getUniqueParams(index_data).getY();
    double derivate = (out > 0) ? 1.0 : 0.0;
    return derivate * (target - out);
}

double H_layer::net(int index_perceptron, int index_data, Datasets data) {
    double sum = 0.0;
    for (size_t i = 0; i < data.getSizeParams(); i++) {
        sum += this->perceptrons[index_perceptron].getWeight(i) * data.getUniqueParams(index_data).getUniqueX(i);
    }
    sum += this->perceptrons[index_perceptron].getBias();
    return sum;
}

double H_layer::output(int index_perceptron, int index_data, Datasets data) {
    double net_out = net(index_perceptron, index_data, data);
    return (net_out > 0) ? net_out : 0; // ReLU activation
}