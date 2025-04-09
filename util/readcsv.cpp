#include "readcsv.h"

ReadCSV::ReadCSV(const std::string &filename, char delimiter): filename(filename), delimiter(delimiter) {
    this->data = Datasets();
    this->columns = std::vector<std::string>();
}

void ReadCSV::readDataToDatasets(std::vector<std::vector<double>> vars, std::vector<int> labels, int size, int size_params) {
    Datasets new_data = Datasets(size, size_params);
    for (int i = 0; i < size; i++) {
        new_data.insertParams(vars.at(i), labels.at(i), size_params, i);
    }
    this->data = new_data;
}

void ReadCSV::readData() {
    int size;
    int size_params;
    std::filebuf filebuf;
    filebuf.open(filename, std::ios::in);
    if (!filebuf.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
        exit(-1);
    } else {
        std::istream is(&filebuf);
        std::string line;
        int row = 0;
        std::vector<std::string> columns;
        std::vector<std::vector<double>> vars;
        bool isFirstRow = true;
        while (getline(is, line)) { 
            std::stringstream ss (line);
            if (isFirstRow) {
                while (getline(ss, line, this->delimiter)) {
                    columns.push_back(line);
                    size_params = columns.size() - 1;
                    isFirstRow = false;
                }
                this->columns = columns;
            } else {
                std::vector<double> aux_vars;
                    while (getline(ss, line, this->delimiter)) {
                        aux_vars.push_back(std::stod(line));
                    }
                vars.push_back(aux_vars);
                row++;
            }            
        }
        size = row;
        std::vector<std::vector<double>> filtered_vars;
        std::vector<int> filtered_labels;
        for (int i = 0; i < size; i++) {
            std::vector<double> selected_vector = vars.at(i);
            int y = selected_vector.at(size_params);
            filtered_labels.emplace_back(y);
            selected_vector.pop_back();
            filtered_vars.emplace_back(selected_vector);   
        }
        this->readDataToDatasets(filtered_vars, filtered_labels, size, size_params);
    }
}

Datasets ReadCSV::getData() {
    return this->data;
}

void ReadCSV::getLabels() {
    std::cout << ORANGE << "[ReadCSV::getLabels] " << RESET;
    std::cout << "[";
    for (int i = 0; i < this->data.getSize(); i++) {
       std::cout << this->data.getUniqueParams(i).getY();
       if (i < this->data.getSize() - 1) {
           std::cout << ", ";
       }
    }
    std::cout << "]" << std::endl;
}

void ReadCSV::getColumns() {
    std::cout << ORANGE << "[ReadCSV::getColumns] " << RESET;
    std::cout << "[";
    for (std::string column: this->columns) {
        std::cout << "\""<< column << "\"";
        if (column != this->columns.back()) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}
