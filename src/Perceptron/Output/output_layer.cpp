#include "output_layer.h"
#include "../Hidden/hidden_layer.h"
#include <cmath>
#include <stdexcept>

O_layer::O_layer(double learning_rate, int size, int size_conn, int size_weights) 
    : learning_rate(learning_rate), size_conn(size_conn), size_weights(size_weights) {
    this->perceptrons.reserve(size);
    for (int i = 0; i < size; i++) {
        this->perceptrons.emplace_back(size_conn, size_weights);
    }
}

O_layer::~O_layer() {}

Perceptron &O_layer::getPerceptron(size_t index_perceptron) {
    if (index_perceptron >= perceptrons.size()) {
        throw std::out_of_range("Index out of bounds in getPerceptron");
    }
    return this->perceptrons[index_perceptron];
}

std::vector<Perceptron> &O_layer::getPerceptrons() {
    return this->perceptrons;
}

double O_layer::getLearningRate() {
    return this->learning_rate;
}

void O_layer::updateWeight(int index_perceptron, int index_data, int index_weight, int index_hidden_p, H_layer* hidden_layer, Datasets data) {    
    double delta = delta_function(index_perceptron, index_hidden_p, index_data, hidden_layer, data);
    double hidden_output = hidden_layer->output(index_hidden_p, index_data, data);
    double current_weight = this->perceptrons[index_perceptron].getWeight(index_weight);
    double new_weight = current_weight + this->getLearningRate() * delta * hidden_output;
    this->perceptrons[index_perceptron].updateWeight(new_weight, index_weight);
}

void O_layer::updateBias(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data) {
    double current_bias = this->perceptrons[index_perceptron].getBias();
    double delta = delta_function(index_perceptron, index_hidden_p, index_data, hidden_layer, data);
    double new_bias = current_bias + this->getLearningRate() * delta;
    this->perceptrons[index_perceptron].updateBias(new_bias);
}

std::vector<double> O_layer::softmax(const std::vector<double>& net_values) {
    std::vector<double> softmax_values(net_values.size());
    double max_val = net_values[0];
    for (size_t i = 1; i < net_values.size(); i++) {
        if (net_values[i] > max_val)
            max_val = net_values[i];
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < net_values.size(); i++) {
        softmax_values[i] = exp(net_values[i] - max_val);
        sum_exp += softmax_values[i];
    }
    
    for (size_t i = 0; i < net_values.size(); i++) {
        softmax_values[i] /= sum_exp;
    }
    return softmax_values;
}

double O_layer::delta_function(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data) {
    int target_class = data.getUniqueParams(index_data).getY();
    double target = (index_perceptron == target_class) ? 1.0 : 0.0;
    double out = output(index_perceptron, index_hidden_p, index_data, hidden_layer, data);
    return (target - out);
}

double O_layer::net(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data) {
    double sum = 0.0;
    for (size_t i = 0; i < hidden_layer->getPerceptrons().size(); i++) {
        sum += this->perceptrons[index_perceptron].getWeight(i) * hidden_layer->output(i, index_data, data);
    }
    sum += this->perceptrons[index_perceptron].getBias();
    return sum;
}

double O_layer::output(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data) {
    return net(index_perceptron, index_hidden_p, index_data, hidden_layer, data);
}
