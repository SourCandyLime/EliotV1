//./Eliot/Cortex.cpp
#include "Cortex.h"
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <vector>

Neuron* allocateNeurons(const char* cortex_name, int neuron_count, int input_neurons) {
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
        snprintf(h_neurons[i].id, sizeof(h_neurons[i].id), "%s.neuron.%d", cortex_name, i);
    }

    return h_neurons;
}

void generateLinkMap(Cortex& cortex, float link_chance) {
    for (int i = 0; i < cortex.neuron_count; ++i) {
        std::vector<int> links;

        for (int j = 0; j < cortex.neuron_count; ++j) {
            if ((float)rand() / RAND_MAX < link_chance || (i == j && cortex.neurons[i].is_recursive)) {
                links.push_back(j);
            }
        }

        int link_count = links.size();
        cortex.neurons[i].num_weights = link_count;

        cudaMallocManaged(&cortex.neurons[i].weights, sizeof(float) * link_count);
        cudaMallocManaged(&cortex.neurons[i].linked_indices, sizeof(int) * link_count);

        for (int k = 0; k < link_count; ++k) {
            cortex.neurons[i].weights[k] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            cortex.neurons[i].linked_indices[k] = links[k];
        }
    }
}
