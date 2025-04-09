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