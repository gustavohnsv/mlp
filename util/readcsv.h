#ifndef READCSV_H
#define READCSV_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "../src/Datasets/datasets.h"

#define ORANGE "\033[38;5;214m"
#define RESET "\033[0m"

class ReadCSV {
    private:
        std::string filename;
        char delimiter;
        Datasets data;
        std::vector<std::string> columns;
    public:
        ReadCSV(const std::string& filename, char delimiter = ',');
        void readDataToDatasets(std::vector<std::vector<double>> vars, std::vector<int> labels, int size, int size_params);
        void readData();
        Datasets getData();
        void getLabels();
        void getColumns();
        int getNumberOfClasses();
};

#endif // READCSV_H