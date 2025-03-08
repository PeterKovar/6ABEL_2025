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
        //Neuronen in den einzelnen Layer erzeugen
       for (int i=0; i<numLayers; ++i) {
        layers[i].numNeurons = layer_struct[i]; //Anzahl Neuronen über übernehmen
        layers[i].makeNeurons(); //Neuronen erzeugen
        if (i==0) layers[i].numInputs = 0; //Layer 0 kein Input
        else layers[i].numInputs = layers[i-1].numNeurons;  // Anz. Inputs =
       }                                                    // Anz. Neur. im vorigen Layer
        
    }
    // Destruktor
    ~NeuronNet() {
        delete [] layers; //Speicher wieder freigeben
    }

    //Vorwaertsfunktion
    void forward (double *input) {
        // Setzen der Input-Werte in den Eingangslayer
        for (int i=0;i<layers[0].numNeurons; ++i){
          layers[0].neurons[i].output = input[i];
        }
        //Berechnen der Ausgaben für jede Schicht
        for(int i=1; i < numLayers; ++i) {     // Alle Layer durchgehen, Eingangslayer bereits erledigt
          for(int j=0; j<layers[i].numNeurons; ++j){  // Alle Neuronen durchgehen
            double sum = 0.0;                         // Ausgaenge summieren
            for (int k=0; k < layers[i].numInputs; ++k) {   //Alle Eingaenge durchgehen
              //Alle Eingaenge mit dem Gewicht mult. und summieren
              sum += layers[i-1].neurons[k].output *
                     layers[i].neurons[j].weight;
            }
            layers[i].neurons[j].output = sigmoid(sum); // Mit Aktivierungsfunktion norm.
          }
        }
    }
};

int main(int argc, char **argv){
    int layer_struct[] = {2, 3, 1};  // Struktur des Netzwerkes festlegen
    NeuronNet net(layer_struct, sizeof(layer_struct)/sizeof(int)); // net instanzieren
    double input[] = {0.5, 0.8};
    // Ausgang berechnen
    net.forward(input);
    std::cout << net.layers[sizeof(layer_struct)/sizeof(int)-1].neurons[0].output;

 return 0;   
}

