//CortexManager.cpp
#include "Neuron.h"
#include "Neuron.cuh"
#include <cuda_runtime.h>

Neuron* allocateNeurons(char cortex_name, int neuron_count, int input_neurons) {
    Neuron* h_neurons = new Neuron[neuron_count];

    for (int i = 0; i < neuron_count; ++i) {
        float* weights;
        cudaMallocManaged(&weights, sizeof(float) * input_neurons);  // Unified Memory

        for (int j = 0; j < input_neurons; ++j) {
            weights[j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // -1 to 1
        }

        h_neurons[i].weights = weights;
        h_neurons[i].num_weights = input_neurons;
        h_neurons[i].threshold = ((rand() % 100) / 100.0f);
        h_neurons[i].last_output = 0.0f;
		h_neurons[i].is_recursive = (rand() / RAND_MAX) < 0.1f;  // Randomly set recursive
        h_neurons[i].is_input = (i < input_neurons);  // First X are inputs
		h_neurons[i].is_negative = (rand() / RAND_MAX) < 0.25f; // Randomly set negative
        h_neurons[i].id = cortex_name + ".neuron." + std::to_string(i);
    }

    return h_neurons;
}

array audioCortex = allocateNeurons('audio', 256, 16);
