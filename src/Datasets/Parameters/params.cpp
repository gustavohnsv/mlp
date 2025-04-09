#include "params.h"
#include <stdexcept>

Params::Params(int size) : y(0), size(size) {
    this->x = std::vector<double>(size);
}

Params::Params() : y(0), size(0) {
    this->x = std::vector<double>(); // Inicializa o vetor vazio
}

void Params::insertX(double value, size_t index) {
    if (index < 0 || index >= this->x.size()) {
        throw std::out_of_range("Index out of bounds in insertX");
    }
    this->x[index] = value;
}

void Params::insertY(int value) {
    this->y = value;
}

double Params::getUniqueX(size_t index) {
    if (index < 0 || index >= this->x.size()) {
        throw std::out_of_range("Index out of bounds in getUniqueX");
    }
    return this->x.at(index);
}

int Params::getY() {
    return this->y;
}

size_t Params::getSize() {
    return this->size;
}
