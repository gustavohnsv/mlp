#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <vector>
#include <stdexcept>

class Perceptron {
    private:
        std::vector<double> weights;
        double bias;
        int conn;
    public:
        Perceptron(int conn, int size_w);
        ~Perceptron();
        double &getWeight(size_t index);
        void updateWeight(double value, int index);
        double &getBias();
        void updateBias(double value);
        int getConnections();
        int getSize();
};

#endif // PERCEPTRON_H