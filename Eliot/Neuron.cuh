//./Eliot/Neuron.cuh
#pragma once
#include "Neuron.h"

// Only include CUDA stuff if compiling with nvcc
#ifdef __CUDACC__
extern "C" {
    __global__ void scream(Neuron* neurons, float* inputs, int total_neurons, int input_count);
    __global__ void adapt(Neuron* neurons, float* inputs, int total_neurons, float learn_rate);
}
#endif