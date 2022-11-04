#include "Network.h"

using namespace std;

// How will the network be specified
// some array like (3, 2, 1)
// "Make 3 neurons, make 2 neurons, make 1 neuron"
// We know the number of layers (size of input array)
// We know the number of elements in each layer (values in the array)
// A network can be built from this information
Network::Network(vector<unsigned> &topology)
{
    // Construct a network
    // Know the number of layers
    unsigned num_layers = topology.size();

    for (unsigned layer = 0; layer < num_layers; layer++)
    {
        // Create a new layer in the network
        // This network will have a bias neuron in each layer
        // Note the "<=" used here to create 1+ the number of elements
        // requested in topology
        _layers.push_back(Layer());

        // Store the number of outputs of each neuron
        // (Number of neurons in the next layer)
        // Not fancy enough to make this one line if statement
        unsigned num_outputs;
        if (layer == num_layers - 1)
            num_outputs = 0;
        else
            num_outputs = topology[layer + 1];

        for (unsigned neuron = 0; neuron <= topology[layer]; neuron++)
        {
            // Create a new neuron in this layer
            _layers.back().push_back(Neuron(num_outputs, neuron));
            cout << "Created a new neuron in layer " << layer << endl;
        }
        // Set the bias neuron to 1.0
        _layers.back().back().set_output_value(1.0);
    }
}

// Define all network functions here
void Network::feedforward(vector<double> &input_vals)
{
    // TODO: Better error handling
    assert(input_vals.size() == _layers[0].size() - 1);

    // The input layer should "latch" the input values
    for (size_t i = 0; i < input_vals.size(); ++i)
    {
        // Assign value to input neuron
        _layers[0][i].set_output_value(input_vals[i]);
    }

    // Input elements are set
    // Do forward propogation
    // Tell every neuron in each layer to do their "feedforward" function
    // Skip the input layer (Start at 1)
    for (size_t layer = 1; layer < _layers.size(); layer++)
    {
        Layer &prev_layer = _layers[layer - 1];
        // Loop to each neuron in layer
        // Skip the bias neuron
        for (size_t neuron = 0; neuron < _layers[layer].size() - 1; neuron++)
        {
            // Feedforward will require a handle to the previous layer of neurons
            _layers[layer][neuron].feedforward(prev_layer);
        }
    }
}

void Network::back_prop(vector<double> &target_vals)
{
    // Needs to do:
    // Calculate overall net error
    //  Use RMS
    // Calculate gradients of all laers
    // update connection weights

    // Overall error
    // loop through all output neurons
    Layer &output_layer = _layers.back();
    _error = 0.0;

    // Note -1 to not include the bias in the back propogation
    for (unsigned n = 0; n < output_layer.size() - 1; ++n)
    {
        double delta = target_vals[n] - output_layer[n].get_output_val();
        _error += delta * delta;
    }

    _error /= output_layer.size() - 1; // get average error squared
    _error = sqrt(_error);             // Get RMS

    // Creating data for live updates on training status
    _recent_average_error = (_recent_average_error * _recent_average_error_smoothing_factor + _error) / (_recent_average_error_smoothing_factor + 1);

    // Calculate output layer gradients
    for (unsigned n = 0; n < output_layer.size() - 1; ++n)
    {
        // Call each neuron to calculate it's output gradients
        output_layer[n].calc_gradients(target_vals[n]);
    }

    // Calculate hidden layer gradients
    for (unsigned layer = _layers.size() - 2; layer > 0; --layer)
    {
        Layer &hidden_layer = _layers[layer];
        Layer &next_layer = _layers[layer + 1];

        for (unsigned n = 0; n < hidden_layer.size() - 1; ++n)
        {
            // Calculate gradients
            hidden_layer[n].calc_hidden_gradients(next_layer);
        }
    }

    // Update connection weights
    for (unsigned layer = _layers.size() - 1; layer > 0; --layer)
    {
        Layer &curr_layer = _layers[layer];
        Layer &prev_layer = _layers[layer - 1];

        for (unsigned neuron = 0; neuron < curr_layer.size() - 1; ++neuron)
        {
            curr_layer[neuron].update_weights(prev_layer);
        }
    }
}

// Will store the output values of the network in the result_vals vector
void Network::get_results(vector<double> &result_vals)
{
    result_vals.clear();

    for (unsigned n = 0; n < _layers.back().size() - 1; ++n)
    {
        result_vals.push_back(_layers.back()[n].get_output_val());
    }
}