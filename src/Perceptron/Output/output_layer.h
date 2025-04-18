#ifndef O_LAYER_H
#define O_LAYER_H

#include <cmath>
#include <vector>
#include <stdexcept>

#include "../Hidden/hidden_layer.h"
#include "../BasePerceptron/perceptron.h"
#include "../../Datasets/datasets.h"

class H_layer;

class O_layer {
private:
    std::vector<Perceptron> perceptrons;
    double learning_rate;
    int size_conn;
    int size_weights;

public:
    O_layer(double learning_rate, int size, int size_conn, int size_weights);
    ~O_layer();
    Perceptron &getPerceptron(size_t index_perceptron);
    std::vector<Perceptron> &getPerceptrons();
    double getLearningRate();
    double net(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data);
    double output(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data);
    void updateWeight(int index_perceptron, int index_data, int index_weight, int index_hidden_p, H_layer* hidden_layer, Datasets data);
    void updateBias(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data);
    double delta_function(int index_perceptron, int index_hidden_p, int index_data, H_layer* hidden_layer, Datasets data);
    std::vector<double> softmax(const std::vector<double>& net_values);
};

#endif // O_LAYER_H