#include "perceptron.h"
#include <stdexcept>

Perceptron::Perceptron(int conn, int size_w) : weights(size_w, 0.0), bias(0), conn(conn) {}

Perceptron::~Perceptron() {}

double &Perceptron::getWeight(size_t index) {
    if (index < 0 || index >= this->weights.size()) {
        throw std::out_of_range("Index out of bounds in getWeight");
    }
    return this->weights[index];
}

void Perceptron::updateWeight(double value, int index) {
    this->getWeight(index) = value; 
}

double &Perceptron::getBias() {
    return this->bias;
}

void Perceptron::updateBias(double value) {
    this->getBias() = value;
}

int Perceptron::getConnections() {
    return this->conn;
}

int Perceptron::getSize() {
    return this->weights.size();
}
