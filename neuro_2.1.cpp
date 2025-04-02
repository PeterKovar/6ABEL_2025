#include <iostream>
#include <cmath>
#include <cstdlib>

// Sigmoid-Aktivierungsfunktion
double sigmoid(double s) {
    return 1.0 / (1.0 + exp(-s));
}

// Ableitung der Sigmoid-Funktion (für Backpropagation)
double sigmoid_derivative(double s) {
    return s * (1.0 - s); // s = sigmoid(x)
}

// Neuronenklasse
class Neuron {
public:
    double* weights;
    double bias;
    double output;
    double delta;
    int numInputs;

    Neuron() {}

    // Konstruktor mit Anzahl der Inputs
    /*Neuron(int numInputs) : numInputs(numInputs) {
        weights = new double[numInputs];
        for (int i = 0; i < numInputs; ++i) {
            weights[i] = ((double) rand() / RAND_MAX) * 2 - 1; // Initialisierung [-1, 1]
        }
        bias = ((double) rand() / RAND_MAX) * 2 - 1;
        output = 0.0;
        delta = 0.0;
    }*/

    ~Neuron() {
        delete[] weights;
    }
};

// Layerklasse
class Layer {
public:
    Neuron* neurons;
    int numNeurons;
    int numInputs;

    Layer() {}

    void makeNeurons(int inputsPerNeuron) {
        neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; ++i) {
            //neurons[i] = Neuron(); // Standard-Konstruktor
    
            // Manuell initialisieren:
            neurons[i].numInputs = inputsPerNeuron;
            neurons[i].weights = new double[inputsPerNeuron];
            for (int j = 0; j < inputsPerNeuron; ++j) {
                neurons[i].weights[j] = ((double) rand() / RAND_MAX) * 2 - 1;
            }
            neurons[i].bias = ((double) rand() / RAND_MAX) * 2 - 1;
            neurons[i].output = 0.0;
            neurons[i].delta = 0.0;
        }
    }

    ~Layer() {
        delete[] neurons;
    }
};

// Netzwerkklasse
class NeuronNet {
public:
    Layer* layers;
    int numLayers;

    NeuronNet(int* layer_struc, int maxLayer) {
        numLayers = maxLayer;
        layers = new Layer[numLayers];

        for (int i = 0; i < numLayers; ++i) {
            layers[i].numNeurons = layer_struc[i];
            if (i == 0) {
                layers[i].numInputs = 0; // Input-Layer
                layers[i].makeNeurons(0);
            } else {
                layers[i].numInputs = layers[i - 1].numNeurons;
                layers[i].makeNeurons(layers[i].numInputs);
            }
        }
    }

    ~NeuronNet() {
        delete[] layers;
    }

    // Vorwärtsausbreitung
    void forward(double* input) {
        // Input-Layer direkt setzen
        for (int i = 0; i < layers[0].numNeurons; ++i) {
            layers[0].neurons[i].output = input[i];
        }

        // Hidden & Output Layer
        for (int i = 1; i < numLayers; ++i) {
            for (int j = 0; j < layers[i].numNeurons; ++j) {
                double sum = 0.0;
                for (int k = 0; k < layers[i].numInputs; ++k) {
                    sum += layers[i - 1].neurons[k].output * layers[i].neurons[j].weights[k];
                }
                sum += layers[i].neurons[j].bias;
                layers[i].neurons[j].output = sigmoid(sum);
            }
        }
    }

    // Training mit Backpropagation
    void train(double* input, double* target, double learning_rate) {
        forward(input);

        // Fehler im Output-Layer
        for (int i = 0; i < layers[numLayers - 1].numNeurons; ++i) {
            double error = target[i] - layers[numLayers - 1].neurons[i].output;
            layers[numLayers - 1].neurons[i].delta = error * sigmoid_derivative(layers[numLayers - 1].neurons[i].output);
        }

        // Fehler Backpropagation für Hidden-Layer
        for (int i = numLayers - 2; i > 0; --i) {
            for (int j = 0; j < layers[i].numNeurons; ++j) {
                double error = 0.0;
                for (int k = 0; k < layers[i + 1].numNeurons; ++k) {
                    error += layers[i + 1].neurons[k].delta * layers[i + 1].neurons[k].weights[j];
                }
                layers[i].neurons[j].delta = error * sigmoid_derivative(layers[i].neurons[j].output);
            }
        }

        // Gewichte & Bias aktualisieren
        for (int i = 1; i < numLayers; ++i) {
            for (int j = 0; j < layers[i].numNeurons; ++j) {
                for (int k = 0; k < layers[i].numInputs; ++k) {
                    layers[i].neurons[j].weights[k] += learning_rate * layers[i].neurons[j].delta * layers[i - 1].neurons[k].output;
                }
                layers[i].neurons[j].bias += learning_rate * layers[i].neurons[j].delta;
            }
        }
    }
};

// Hauptprogramm
int main() {
    // Netzwerkstruktur
    int layer_struc[] = {2, 3, 5, 1};
    NeuronNet net(layer_struc, sizeof(layer_struc) / sizeof(int));

    // UND-Trainingsdaten
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4][1] = {
        {0},
        {0},
        {1},
        {0}
    };

    double learning_rate = 0.5;
    int epochs = 5000;

    // Training
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0.0;
        for (int i = 0; i < 4; ++i) {
            net.train(inputs[i], targets[i], learning_rate);
            net.forward(inputs[i]);
            double error = pow(targets[i][0] - net.layers[net.numLayers - 1].neurons[0].output, 2);
            total_error += error;
        }
        if (epoch % 500 == 0) {
            std::cout << "Epoche " << epoch << " - Fehler: " << total_error / 4.0 << std::endl;
        }
    }

    // Testen
    std::cout << "\nTrainiertes Netz:\n";
    for (int i = 0; i < 4; ++i) {
        net.forward(inputs[i]);
        std::cout << "Eingang: " << inputs[i][0] << ", " << inputs[i][1]
                  << " -> Ausgabe: " << net.layers[net.numLayers - 1].neurons[0].output << std::endl;
    }

    return 0;
}
