#ifndef MLP_H
#define MLP_H

#include <vector>
#include <math.h>
#include <iostream>

#include "./Output/output_layer.h"
#include "./Hidden/hidden_layer.h"

#define YELLOW "\033[33m"
#define RESET "\033[0m"

class MLP {
    private:
        Datasets data;
        int output_size;
        int hidden_size;
        double learning_rate;
        O_layer output_layer;
        H_layer hidden_layer;
        double accuracy;
        double loss;
    public:
        MLP(Datasets data, int output_size, int hidden_size, double learning_rate);
        ~MLP();
        void init();
        void train(int epochs);
        int predict(int index_data);
        int predict(Params &params);
        double getLoss();
        double getAccuracy();
};

#endif // MLP_H