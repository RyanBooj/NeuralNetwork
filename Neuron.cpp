#include "Neuron.h"

// Set learning rate and momentum multiplier
double Neuron::eta = 0.15;
double Neuron::alpha = 0.1;

Neuron::Neuron(unsigned num_outputs, unsigned my_index)
{
    _my_index = my_index;
    // Neurons know the number of neurons in the next layer
    for (unsigned i = 0; i < num_outputs; i++)
    {
        // Append a new element to the output weights
        _output_weights.push_back(Connection());
        // TODO: Possibly change how the weights are initially defined
        _output_weights.back().weight = gen_weight();
    }
}

void Neuron::feedforward(const Layer &prev_layer)
{
    // Sum the values of all the previous layers
    double sum = 0.0;

    // Loop through all the neurons in the previous layer
    for (unsigned n = 0; n < prev_layer.size(); ++n)
    {
        // Weighted sum of each neuron in the previous layer
        sum += prev_layer[n].get_output_val() * prev_layer[n]._output_weights[_my_index].weight;
    }
    _output = Neuron::_transfer_fn(sum);
}

double Neuron::_transfer_fn(double x)
{
    // Choose a transfer fn defined in the TransferFns.h file
    // For now, use the hyperbolic tan as hardcoded.
    return hyperbolic_tan(x, false);
}

double Neuron::_deriv_transfer_fn(double x)
{
    // Same as above
    return hyperbolic_tan(x, true);
}

// This is only one method to calculate the gradience
// TODO: Make this smarter
void Neuron::calc_gradients(double target_val)
{
    double delta = target_val - _output;
    _gradient = delta * Neuron::_deriv_transfer_fn(_output);
}

void Neuron::calc_hidden_gradients(const Layer &next_layer)
{
    // There is no target value to compare to here
    // Look at the sum of the derivative of the weights of the next layer
    double dow = sum_dow(next_layer);
    _gradient = dow * Neuron::_deriv_transfer_fn(_output);
}

// This function will calculate the sum of the derivatives of the weights of the given layer
// For use in calculating the hidden gradients
double Neuron::sum_dow(const Layer &next_layer)
{
    double sum = 0.0;
    for (unsigned n = 0; n < next_layer.size() - 1; ++n)
    {
        sum += _output_weights[n].weight * next_layer[n]._gradient;
    }
    return sum;
}

void Neuron::update_weights(Layer &prev_layer)
{
    // Remember the weights are stored in the Connection container.
    for (unsigned n = 0; n < prev_layer.size(); ++n)
    {
        Neuron &neuron = prev_layer[n];
        double old_weight_change = neuron._output_weights[_my_index].weight_change;

        // Weight change is composed of different things:
        // Learinging rate - eta
        // previous neuron's output value
        // Gradient
        // <Momentum> A fraction of the previous weight change - alpha
        double new_weight_change = (eta * neuron.get_output_val() * _gradient) + (alpha * old_weight_change);
        neuron._output_weights[_my_index].weight_change = new_weight_change;
        neuron._output_weights[_my_index].weight += new_weight_change;
    }
}
