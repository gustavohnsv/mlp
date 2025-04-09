# Multilevel Perceptron Neural Network in C++

- This project implements a Multilevel Perceptron (MLP) Neural Network in C++ from scratch. It includes functionalities for forward propagation, backpropagation, and training using gradient descent. The code is designed to be modular and efficient, making it suitable for educational purposes or as a foundation for more complex neural network implementations.

- The project is based on an MLP with only one hidden layer (to simplify studies). The dataset columns must also be aligned, with the last column dedicated to labels.

## Commands
```bash
make        # Build do projeto mantendo todos os objetos
make build  # Build do projeto apenas com o objeto final
make clean  # Limpa todos os arquivos 
```

## Execution

- Remember that the number of neurons in the output layer must be equal to the number of distinct classes in the dataset!

```bash
./mlp_program <filename> <output_layer_size> <hidden_layer_size> <learning_rate> <epochs>
```

- During the program execution, you can input data for the model to make predictions. Give it a try!

## Repo tree
```bash
.
├── data.csv
├── LICENSE
├── main.cpp
├── main.h
├── Makefile
├── README.md
├── src
│   ├── Datasets
│   │   ├── datasets.cpp
│   │   ├── datasets.h
│   │   └── Parameters
│   │       ├── params.cpp
│   │       └── params.h
│   └── Perceptron
│       ├── BasePerceptron
│       │   ├── perceptron.cpp
│       │   └── perceptron.h
│       ├── Hidden
│       │   ├── hidden_layer.cpp
│       │   └── hidden_layer.h
│       ├── mlp.cpp
│       ├── mlp.h
│       └── Output
│           ├── output_layer.cpp
│           └── output_layer.h
└── util
    ├── readcsv.cpp
    └── readcsv.h
```