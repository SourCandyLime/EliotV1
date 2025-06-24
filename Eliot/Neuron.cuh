// Kernels.cuh
#pragma once

#include "Neuron.h"

__global__ void scream(Neuron* neurons, float* inputs, int total_neurons, int input_count);
__global__ void adapt(Neuron* neurons, float* inputs, int total_neurons, float learn_rate);
__global__ array allocateNeurons(char cortex_name, int neuron_count, int input_count)