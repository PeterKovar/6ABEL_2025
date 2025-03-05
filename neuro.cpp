// Proj. Neuronales Netz v1.0
//(c) kova 26.02.2025

# include <iostream>
# include <cmath>

double sigmoid(double s){
    return 1.0 / (1.0 + exp(-s));   
}
double sigmoid_derivative(double s){  // Sigmoid Wert muss uebergeben werden
    return s * (1.0 -s);
}
//Neuronenklasse
class Neuron{
    public:
    double weight;
    double output;
    //Konstruktor
    Neuron() : weight(0.5), output(0.0) {}    
};
//Layerklasse
class Layer{
    public:
    Neuron* neurons;
    int numNeurons;
    int numInputs;
    //Konstruktor
    Layer() {}
    void makeNeurons(){
        neurons = new Neuron[numNeurons];
    }
    ~Layer(){
        delete[] neurons;  //Speicher freigeben
    }
};
//Neuronales Netz
class NeuronNet {
    public:
    Layer* layers;
    int numLayers;
    //Konstruktor
    NeuronNet(int *layer_struct, int numLayers) {
        this->numLayers = numLayers;
        //Layer erzeugen
        layers = new Layer[this->numLayers];
        
    }
};

int main(int argc, char **argv){
    int layer_struct[] = {2, 3, 1};
    NeuronNet net(layer_struct, sizeof(layer_struct)/sizeof(int));

 return 0;   
}

