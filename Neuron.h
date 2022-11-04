#include <vector>
#include <cstdlib>
#include <iostream>
#include <ctime>
#include "TransferFns.h"
using namespace std;

class Neuron;

typedef vector<Neuron> Layer;

// Use a simple struct to store more information in one element of a vector
// The connection struct stores a weight corresponding to a neuron in the next
// layer. It also keeps a memory of how much the weight changed when it was
// updated last for calculations needed down the line

// TODO: Make this connection struct into a class
// Neds to construct itself with some value for each weight
// May want to change the ruction of these initial weights for some reason
struct Connection
{
    double weight;
    double weight_change;
};

class Neuron
{
private:
    // Transfer functions
    static double _transfer_fn(double);
    static double _deriv_transfer_fn(double);

    // neuron output value
    double _output;

    // output weights
    vector<Connection> _output_weights;

    // Handle creating the weights for each connection
    // Possibly change this in the future
    static double gen_weight()
    {
        srand(time(NULL));
        return rand() / (double)RAND_MAX;
    }

    // Each neuron will know its index in the layer
    unsigned _my_index;

    // Neuron will keep track of it's gradient
    double _gradient;

    // learning rate
    static double eta; // [0.0, ..., 1.0]

    // Multiplier of last weight change (momentum)
    static double alpha; // [0.0, ..., 1.0]

public:
    Neuron(unsigned num_outputs, unsigned my_index);

    // Setters and getters
    void set_output_value(double val) { _output = val; };
    double get_output_val(void) { return _output; };

    // Feedforward the values from the previous layers
    void feedforward(Layer &prev_layer);

    // Calculate the gradients
    void calc_gradients(double target_val);
    void calc_hidden_gradients(Layer &next_layer);
    double sum_dow(Layer &next_layer);
    void update_weights(Layer &prev_layer);
};
