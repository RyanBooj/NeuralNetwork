/*
    Inspired from http://www.millermattson.com/dave/?p=54
*/

#include <vector>
#include "Neuron.h"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace std;

// The types we are using can be explicitly defined for clairity
// A layer will be a vector of neuron objects
typedef vector<Neuron> Layer;

class Network
{
private:
    // The network will contain a vector of layer objects
    // Can be accessed with two [] operators
    vector<Layer> _layers; // _layers[layer_num][neuron_num]

    // Need to know the network error
    double _error;

public:
    // TODO: Define topology
    // The input to the network ructor will contain details
    // about the network to be create
    // i.e. number of neurons in each layer, number of layers, ect.
    Network(vector<unsigned> &topology);

    // Neural network needs to be able to feed some value into the network
    // once it is defined
    // Call this operation,  "Feedforward"
    // Pass a reference to a vector of doubles to this function
    // This input values vector will not change, make
    void feedforward(vector<double> &input_vals);

    // The network will be able to train itself given a correct answer
    // Same vector of doubles as used above
    // target_values will not be read-only in this function
    void back_prop(vector<double> &target_vals);

    // The network will want some way to describe its results
    // to the user
    // Input a container for the network to fill with results
    // This function does not modify the network at all, hence it is
    // This function will change result_vals (it will fill with network outputs)
    void get_results(vector<double> &result_vals);

    // Save some values for getting a recent average of the network's training process
    double _recent_average_error;
    double _recent_average_error_smoothing_factor;
};
