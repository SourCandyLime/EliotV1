//./Eliot/Cortex.h
#pragma once

#include "Neuron.h"
#include "Neuron.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <string>

Neuron* allocateNeurons(const char* cortex_name, int neuron_count, int input_neurons);

struct Cortex {
	Neuron* neurons;  // Pointer to an array of Neurons
	int neuron_count; // Total number of neurons in the cortex
	int input_neurons; // Number of input neurons
	char name[64]; // Identifier for the cortex

	Cortex(const char* cortex_name, int neuron_count, int input_neurons)
		: neuron_count(neuron_count), input_neurons(input_neurons) {
		snprintf(name, sizeof(name), "%s", cortex_name);
		neurons = allocateNeurons(cortex_name, neuron_count, input_neurons);
	}
	~Cortex() {
		delete[] neurons; // Free allocated memory for neurons
	}
};  

void generateLinkMap(Cortex& cortex, float link_chance = 0.05f);