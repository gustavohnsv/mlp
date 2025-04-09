#ifndef READCSV_H
#define READCSV_H

#include "../src/Datasets/datasets.h"
#include <vector>
#include <string>

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