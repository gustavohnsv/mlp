#include <iostream>
#include <vector>
#include <ctime>
#include <fstream>
#include <sstream>
#include <iostream>
#include "./util/readcsv.h"
#include "./src/Perceptron/mlp.h"
#include "./src/Perceptron/Output/output_layer.h"
#include "./src/Perceptron/Hidden/hidden_layer.h"
#include "./src/Perceptron/BasePerceptron/perceptron.h"
#include "./src/Datasets/Parameters/params.h"
#include "./src/Datasets/datasets.h"

#define RED "\033[31m"
#define BLUE "\033[34m"
#define GREEN "\033[32m"
#define RESET "\033[0m"

using namespace std;

void showResults(std::string filename, int output_layer_size, int hidden_layer_size, double learning_rate, int epochs, MLP mlp, Datasets data) {
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Filename: " << filename << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Number of samples: " << data.getSize() << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Number of features: " << data.getSizeParams() << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Hidden layer size: " << hidden_layer_size << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Output layer size: " << output_layer_size << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Learning rate: " << learning_rate << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Loss: " << mlp.getLoss() << std::endl;
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Accuracy: " << mlp.getAccuracy() << "%" << std::endl;
}

double randomFloatNumber() {
    return rand() % 100 / 100.0;
}

int randomIntNumber(int init, int end) {
    return (rand() % (end - init + 1)) + init;
}

int main(int argc, char* argv[]) {

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <csv_file> <output_layer_size> <hidden_layer_size> <learning_rate> <epochs>" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    int output_layer_size = std::stoi(argv[2]);
    int hidden_layer_size = std::stoi(argv[3]);
    double learning_rate = std::stod(argv[4]);
    int epochs = std::stoi(argv[5]);

    ReadCSV readcsv = ReadCSV(filename);
    readcsv.readData();

    Datasets data = readcsv.getData();
    readcsv.getColumns();

    data.standardize();
    
    MLP mlp(data, output_layer_size, hidden_layer_size, learning_rate);
    mlp.init();
    mlp.train(epochs);
    
    // Class distribution verification
    // std::vector<int> class_count(4, 0);
    // for (int i = 0; i < data.getSize(); i++) {
    //     int class_idx = data.getUniqueParams(i).getY();
    //     if (class_idx >= 0 && class_idx < 5) {
    //         class_count[class_idx]++;
    //     }
    // }

    // single prediction test
    srand(time(0));
    int test_index = randomIntNumber(0, data.getSize() - 1);
    mlp.predict(test_index);
    std::cout << GREEN << "[main.cpp] " << RESET;
    std::cout << "Expected class: " << data.getUniqueParams(test_index).getY() << RESET << std::endl;

    showResults(filename, output_layer_size, hidden_layer_size, learning_rate, epochs, mlp, data);    

    while (true) {
        std::cout << GREEN << "[main.cpp] " << RESET;
        std::cout << "Enter the parameters for prediction (comma-separated, or 'exit' to quit): ";
        std::string input;
        std::getline(std::cin, input);
        
        if (input == "exit") {
            break;
        }
        
        std::vector<double> param_values;
        std::stringstream ss(input);
        std::string param;
        
        while (std::getline(ss, param, ',')) {
            param_values.push_back(std::stod(param));
        }
        
        if (param_values.size() != data.getSizeParams()) {
            std::cout << RED << "[main.cpp] " << RESET;
            std::cout << "Invalid number of parameters. Expected: " << data.getSizeParams() << std::endl;
            continue;
        }
        
        Params input_params(param_values.size());
        for (size_t i = 0; i < param_values.size(); i++) {
            input_params.insertX(param_values[i], i);
        }

        data.standardizeParams(input_params);
        
        mlp.predict(input_params);
    }
    return 0;
}
