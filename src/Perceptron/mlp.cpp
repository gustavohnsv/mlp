#include "mlp.h"

MLP::MLP(Datasets data, int output_size, int hidden_size, double learning_rate) : 
    data(data), 
    output_size(output_size), 
    hidden_size(hidden_size),
    learning_rate(learning_rate),
    output_layer(learning_rate, output_size, hidden_size, hidden_size),
    hidden_layer(learning_rate, hidden_size, data.getSizeParams(), data.getSizeParams()),
    accuracy(0.0),
    loss(0.0) {}

MLP::~MLP() {}

void MLP::init() {
    srand(time(nullptr));
    for (int j = 0; j < this->hidden_size; j++) {
        for (size_t p = 0; p < data.getSizeParams(); p++) {
            double range = sqrt(2.0 / (data.getSizeParams() + hidden_size));
            double random_val = ((double)rand() / RAND_MAX) * 2 * range - range;
            hidden_layer.getPerceptron(j).updateWeight(random_val, p);
        }
    }
    for (int k = 0; k < this->output_size; k++) {
        for (int j = 0; j < hidden_size; j++) {
            double range = sqrt(2.0 / (data.getSizeParams() + hidden_size));
            double random_val = ((double)rand() / RAND_MAX) * 2 * range - range;
            output_layer.getPerceptron(k).updateWeight(random_val, j);        
        }
    }
    for (int j = 0; j < this->hidden_size; j++) {
        double range = sqrt(2.0 / (data.getSizeParams() + hidden_size));
        double random_val = ((double)rand() / RAND_MAX) * 2 * range - range;
        hidden_layer.getPerceptron(j).updateBias(random_val);    
    }
    for (int k = 0; k < this->output_size; k++) {
        double range = sqrt(2.0 / (data.getSizeParams() + hidden_size));
        double random_val = ((double)rand() / RAND_MAX) * 2 * range - range;
        output_layer.getPerceptron(k).updateBias(random_val); 
    }
}

void MLP::train(int epochs) {
    int epoch = 1;
    while (epoch <= epochs) {
        double total_error = 0.0;
        
        for (int i = 0; i < data.getSize(); i++) {
            Params params = data.getUniqueParams(i);
            int target_class = params.getY();
            
            std::vector<double> hidden_outputs(hidden_size);
            for (int h = 0; h < hidden_size; h++) {
                hidden_outputs[h] = hidden_layer.output(h, i, data);
            }
            
            std::vector<double> net_values(output_size, 0.0);
            for (int j = 0; j < output_size; j++) {
                for (int h = 0; h < hidden_size; h++) {
                    net_values[j] += output_layer.getPerceptron(j).getWeight(h) * hidden_outputs[h];
                }
                net_values[j] += output_layer.getPerceptron(j).getBias();
            }
            
            // softmax
            std::vector<double> outputs = output_layer.softmax(net_values);
            
            // errors and cross-entropy loss
            std::vector<double> errors(output_size);
            for (int j = 0; j < output_size; j++) {
                double target = (j == target_class) ? 1.0 : 0.0;
                errors[j] = target - outputs[j];
                
                // cross-entropy error
                if (j == target_class) {
                    total_error -= log(outputs[j] > 1e-10 ? outputs[j] : 1e-10);
                }

            }
            
            // backpropagation
            for (int j = 0; j < output_size; j++) {
                for (int h = 0; h < hidden_size; h++) {
                    double delta = errors[j] * hidden_outputs[h];
                    double current_weight = output_layer.getPerceptron(j).getWeight(h);
                    double new_weight = current_weight + learning_rate * delta;
                    output_layer.getPerceptron(j).updateWeight(new_weight, h);
                }
                // bias update
                double current_bias = output_layer.getPerceptron(j).getBias();
                double new_bias = current_bias + learning_rate * errors[j];
                output_layer.getPerceptron(j).updateBias(new_bias);
            }
            
            // backpropagation
            std::vector<double> hidden_errors(hidden_size, 0.0);
            for (int h = 0; h < hidden_size; h++) {
                double error_sum = 0.0;
                for (int j = 0; j < output_size; j++) {
                    error_sum += errors[j] * output_layer.getPerceptron(j).getWeight(h);
                }
                double derivate = (hidden_outputs[h] > 0) ? 1.0 : 0.0;
                hidden_errors[h] = error_sum * derivate;
            }
            
            // weight update for hidden layer
            for (int h = 0; h < hidden_size; h++) {
                for (size_t p = 0; p < params.getSize(); p++) {
                    double input_val = params.getUniqueX(p);
                    double delta = hidden_errors[h] * input_val;
                    double current_weight = hidden_layer.getPerceptron(h).getWeight(p);
                    double new_weight = current_weight + learning_rate * delta;
                    hidden_layer.getPerceptron(h).updateWeight(new_weight, p);
                }
                double current_bias = hidden_layer.getPerceptron(h).getBias();
                double new_bias = current_bias + learning_rate * hidden_errors[h];
                hidden_layer.getPerceptron(h).updateBias(new_bias);
            }
        }

        // set loss
        this->loss = total_error / data.getSize();

        // show epochs
        std::cout << YELLOW << "[MLP::train] " << RESET;
        std::cout << "Epoch [" << epoch << "/" << epochs << "]: Average loss = " << (total_error / data.getSize()) << std::endl;
        epoch++;
    }
}

int MLP::predict(int index_data) {
    Params params = data.getUniqueParams(index_data);
    
    std::vector<double> hidden_outputs(hidden_size);
    for (int h = 0; h < hidden_size; h++) {
        hidden_outputs[h] = hidden_layer.output(h, index_data, data);
        // std::cout << "Hidden output " << h << ": " << hidden_outputs[h] << std::endl;
    }
    
    std::vector<double> net_values(output_size, 0.0);
    for (int j = 0; j < output_size; j++) {
        for (int h = 0; h < hidden_size; h++) {
            double weight = output_layer.getPerceptron(j).getWeight(h);
            net_values[j] += weight * hidden_outputs[h];
            // std::cout << "Output " << j << ", weight " << h << ": " << weight << " * " 
            //           << hidden_outputs[h] << " = " << (weight * hidden_outputs[h]) << std::endl;
        }
        net_values[j] += output_layer.getPerceptron(j).getBias();
        // std::cout << "Output " << j << " pre-activation: " << net_values[j] << std::endl;
    }
    
    std::vector<double> outputs = output_layer.softmax(net_values);
    
    int predicted_class = 0;
    double max_output = outputs[0];
    for (int j = 1; j < output_size; j++) {
        if (outputs[j] > max_output) {
            max_output = outputs[j];
            predicted_class = j;
        }
    }
    std::cout << YELLOW << "[MLP::predict] " << RESET;
    std::cout << "Probs per class: ";
    for (int j = 0; j < output_size; j++) {
        std::cout << "Class " << j << ": " << outputs[j] << ", ";
    }
    std::cout << std::endl; 
    std::cout << YELLOW << "[MLP::predict] " << RESET;
    std::cout << "Predicted class: " << predicted_class << " (Value: " << max_output << ")" << std::endl;
    return predicted_class;
}

int MLP::predict(Params& params) {
    std::vector<double> hidden_outputs(hidden_size);
    for (int h = 0; h < hidden_size; h++) {
        double net_sum = 0.0;
        for (size_t p = 0; p < params.getSize(); p++) {
            net_sum += hidden_layer.getPerceptron(h).getWeight(p) * params.getUniqueX(p);
        }
        net_sum += hidden_layer.getPerceptron(h).getBias();
        hidden_outputs[h] = (net_sum > 0) ? net_sum : 0;
    }
    std::vector<double> net_values(output_size, 0.0);
    for (int j = 0; j < output_size; j++) {
        for (int h = 0; h < hidden_size; h++) {
            double weight = output_layer.getPerceptron(j).getWeight(h);
            net_values[j] += weight * hidden_outputs[h];
        }
        net_values[j] += output_layer.getPerceptron(j).getBias();
    }
    std::vector<double> outputs = output_layer.softmax(net_values); 
    int predicted_class = 0;
    double max_output = outputs[0];
    for (int j = 1; j < output_size; j++) {
        if (outputs[j] > max_output) {
            max_output = outputs[j];
            predicted_class = j;
        }
    }
    
    std::cout << YELLOW << "[MLP::predict] " << RESET;
    std::cout << "Probs per class: ";
    for (int j = 0; j < output_size; j++) {
        std::cout << "Class " << j << ": " << outputs[j] << ", ";
    }
    std::cout << std::endl; 
    std::cout << YELLOW << "[MLP::predict] " << RESET;
    std::cout << "Predicted class: " << predicted_class << " (Value: " << max_output << ")" << std::endl;
    
    return predicted_class;
}

double MLP::getLoss() {
    return this->loss;
}

double MLP::getAccuracy() {
    // calculate accuracy
    for (int i = 0; i < data.getSize(); i++) {
        std::streambuf* original_cout_state = std::cout.rdbuf(nullptr); // disable std::cout
        int predicted_class = this->predict(i);
        std::cout.rdbuf(original_cout_state); // re-enable std::cout
        if (predicted_class == data.getUniqueParams(i).getY()) {
            accuracy++;
        }
    }
    return (accuracy / data.getSize()) * 100;
}