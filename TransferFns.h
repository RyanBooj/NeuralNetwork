#ifndef TRANSFER_FNS_H
#define TRANSFER_FNS_H

#include <cmath>
using namespace std;

/*
Hard coded function options for using as a transfer function of a neural network.

Inspired from:
    https://mlnotebook.github.io/post/transfer-functions/
*/

// 1 to 1 transfer function. Derivative constant
double linear(double x, bool derivative = false);

// Sigmoid Function
// Good for backpropogation for math reasons
// Input mapped to a value between 0-1
// Never equal to 0 or 1
double sigmoid(double x, bool derivaive = false);

// Hyperbolic tan function
// Maps all inputs to (-1, 1)
// Useful as above sigmoid function
// Allows applying penalties to nodes (negative values)
// Natural threshold is 0
double hyperbolic_tan(double x, bool derivative = false);
#endif
