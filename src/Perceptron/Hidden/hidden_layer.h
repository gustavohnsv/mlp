#ifndef H_LAYER_H
#define H_LAYER_H

#include <cmath>
#include <vector>
#include <stdexcept>

#include "../Output/output_layer.h"
#include "../../Datasets/Parameters/params.h"
#include "../../Datasets/datasets.h"
#include "../BasePerceptron/perceptron.h"

class O_layer;

class H_layer {
    private:
        std::vector<Perceptron> perceptrons;
        double learning_rate;
        int size_conn;
        int size_weights;
    public:
        H_layer(double learning_rate, int size, int size_conn, int size_weights);
        ~H_layer();
        Perceptron &getPerceptron(size_t index_perceptron);
        std::vector<Perceptron> &getPerceptrons();
        double getLearningRate();
        void updateWeight(int index_perceptron, int index_data, int index_weight, O_layer* output_layer, Datasets data);
        void updateBias(int index_perceptron, int index_data, O_layer *output_layer, Datasets data);
        double delta_function(int index_perceptron, int index_data, O_layer* output_layer, Datasets data);
        double net(int index_perceptron, int index_data, Datasets data);
        double output(int index_perceptron, int index_data, Datasets data);
};

#endif // H_LAYER_H