#include "datasets.h"
#include <vector>
#include <stdexcept>
#include <cmath>

Datasets::Datasets(int size, int size_params) : size(size), size_params(size_params) {
    this->params.reserve(size);
    for (int i = 0; i < size; i++) {
        this->params.push_back(size_params);
    }
    this->isStandardized = false;
    this->means.reserve(size_params);
    this->std_devs.reserve(size_params);
    for (int i = 0; i < size_params; i++) {
        this->means.push_back(0.0);
        this->std_devs.push_back(0.0);
    }
}

Datasets::Datasets() : size(0), size_params(0) {
    this->params.reserve(0);
}

Datasets::~Datasets() {}

void Datasets::insertParams(std::vector<double> x, int y, int size_p, size_t index) {
    if (index >= this->params.size()) {
        throw std::out_of_range("Index out of bounds in insertParams");
    }
    Params new_params(size_p);
    for (size_t i = 0; i < new_params.getSize(); i++) {
        new_params.insertX(x[i], i);
    }
    new_params.insertY(y);
    this->params[index] = new_params;
}

Params Datasets::getUniqueParams(size_t index) {
    if (index < 0 || index >= this->params.size()) {
        throw std::out_of_range("Index out of bounds in getUniqueParams");
    }
    return this->params[index];
}

int Datasets::getSize() {
    return this->params.size();
}

size_t Datasets::getSizeParams() {
    if (this->params.empty()) {
        throw std::runtime_error("No parameters available in getSizeParams");
    }
    return this->params[0].getSize();
}

// standardize via z-score
void Datasets::standardize() {
    std::vector<double> means(this->size_params, 0.0);
    for (int i = 0; i < this->size; i++) {
        for (int j = 0; j < this->size_params; j++) {
            means[j] += this->params[i].getUniqueX(j);
        }
    }
    for (int i = 0; i < this->size_params; i++) {
        means[i] /= this->size;
    }
    std::vector<double> stddevs(this->size_params, 0.0);
    for (int i = 0; i < this->size; i++) {
        for (int j = 0; j < this->size_params; j++) {
            double diff = this->params[i].getUniqueX(j) - means[j];
            stddevs[j] += diff * diff;
        }
    }
    for (int i = 0; i < this->size_params; i++) {
        if (stddevs[i] == 0) {
            throw std::runtime_error("Standard deviation is zero, cannot standardize. Placing 1.0 instead.");
            stddevs[i] = 1.0;
        } else {
            stddevs[i] = sqrt(stddevs[i] / this->size);
        }
    }
    for (int i = 0; i < this->size; i++) {
        for (int j = 0; j < this->size_params; j++) {
            double starndardized_val = (this->params[i].getUniqueX(j) - means[j]) / stddevs[j];
            this->params[i].insertX(starndardized_val, j);
        }
    }
    this->means = means;
    this->std_devs = stddevs;
    this->isStandardized = true;
}

bool Datasets::isStandardizedParams() const {
    return this->isStandardized;
}

void Datasets::standardizeParams(Params &params) {
    if (!this->isStandardized) {
        throw std::runtime_error("Dataset not standardized yet. Call standardize() first.");
    }
    
    if (params.getSize() != (size_t) size_params) {
        throw std::runtime_error("Parameter size mismatch. Cannot standardize.");
    }
    
    for (size_t j = 0; j < params.getSize(); j++) {
        double standardized_val = (params.getUniqueX(j) - means[j]) / std_devs[j];
        params.insertX(standardized_val, j);
    }
}
