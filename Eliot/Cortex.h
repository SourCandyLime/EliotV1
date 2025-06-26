//./Eliot/Cortex.h
#pragma once

#include "Neuron.h"
#include "Neuron.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <string>

Neuron* allocateNeurons(const int cortex_number, int neuron_count, int input_neurons);

struct Cortex {
	Neuron* neurons;  // Pointer to an array of Neurons
	int neuron_count; // Total number of neurons in the cortex
	int input_neurons; // Number of input neurons
	int num; // Identifier for the cortex

	Cortex(const int cortex_number, int neuron_count, int input_neurons)
		: neuron_count(neuron_count), input_neurons(input_neurons), num(cortex_number) {
		neurons = allocateNeurons(cortex_number, neuron_count, input_neurons);
	}
	~Cortex() {
		delete[] neurons; // Free allocated memory for neurons
	}
};  