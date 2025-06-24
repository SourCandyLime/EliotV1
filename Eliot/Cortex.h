//Cortex.h
#pragma once

#include "Neuron.h"
#include "Neuron.cuh"
#include <cuda_runtime.h>

struct Cortex {
	Neuron* neurons;  // Pointer to an array of Neurons
	int neuron_count; // Total number of neurons in the cortex
	int input_neurons; // Number of input neurons
	char name; // Identifier for the cortex
	Cortex(char cortex_name, int neuron_count, int input_neurons)
		: name(cortex_name), neuron_count(neuron_count), input_neurons(input_neurons) {
		neurons = allocateNeurons(cortex_name, neuron_count, input_neurons);
	}
	~Cortex() {
		delete[] neurons; // Free allocated memory for neurons
	}
};