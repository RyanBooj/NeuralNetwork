#include "Network.h"
#include <vector>
#include <stdio.h>
#include <iostream>
#include <ctime>
using namespace std;

int main()
{

    // Create the network's topology
    vector<unsigned> topology;

    /*
        Create a network with the following structure:
        Layer1:
            2 neurons
            3 neurons
            3 neurons
            1 neuron
    */
    topology.push_back(2);
    topology.push_back(3);
    topology.push_back(3);
    topology.push_back(1);

    // Lets make a new neural network
    Network my_net(topology);

    // Network created
    cout << "Network created." << endl;

    // How to handle giving inputs and training the network?
    // Feedforward inputs values into the neural network
    // Backprop trains the network with correct answers
    // Pairing the input and output data in some container
    //      [result, expected]
    // Protptype this process here in main test

    // Lets learn the NAND function
    int training_data_size = 2000;
    // Input vector size == number of nodes in input layer
    // target vector size == number of nodes in the output layer
    vector<double> inputs;
    vector<double> target;
    vector<double> result;
    // Generate random data
    cout << "Generate random data" << endl;
    srand(time(NULL));
    for (int i = 0; i < training_data_size; ++i)
    {
        inputs.clear();
        target.clear();
        // Choose inputs
        double A = rand() % 2;
        double B = rand() % 2;
        inputs.push_back(A);
        inputs.push_back(B);
        target.push_back(!(A && B));

        // Apply inputs
        my_net.feedforward(inputs);

        // Get result
        cout << "[" << A << ", " << B << "]" << endl;
        cout << "Target: " << !(A && B) << endl;

        my_net.get_results(result);
        cout << "Result: " << result.back() << endl;

        // Perform backpropogation
        my_net.back_prop(target);

        // Print recent average error for insight
        cout << "Recent Error: " << my_net._recent_average_error << endl;
    }
    return 0;
}