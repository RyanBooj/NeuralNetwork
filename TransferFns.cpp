#include "TransferFns.h"

double linear(double x, bool derivative)
{
    if (!derivative)
    {
        return x;
    }
    else
    {
        return 1.0;
    }
}

double sigmoid(double x, bool derivaive)
{
    // sigmoid(0) returns 0.5
    // Values above or equal to 0.5 == 1
    // Values below 0.5 == 0
    if (!derivaive)
    {
        return 1 / (1 + exp(-x));
    }
    else
    {
        //  Recursively call ourselves
        // Since default bool is false (recursive case)
        double sig = sigmoid(x);
        return sig;
    }
}

double hyperbolic_tan(double x, bool derivative)
{
    if (!derivative)
    {
        return tanh(x);
    }
    else
    {
        return 1 - pow(tanh(x), 2);
    }
}