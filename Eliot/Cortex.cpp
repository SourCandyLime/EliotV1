//./Eliot/Cortex.cpp
#include "Cortex.h"
#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>

Neuron* allocateNeurons(const int cortex_number, int neuron_count, int input_neurons) {
    Neuron* h_neurons = new Neuron[neuron_count];

	//initialize each neuron

    for (int i = 0; i < neuron_count; ++i) {
        h_neurons[i].num_links = 0;
        h_neurons[i].link_table = nullptr;
        h_neurons[i].threshold = (float) rand() / RAND_MAX;
        h_neurons[i].last_output = 0.0f;
        h_neurons[i].is_recursive = (rand() / (float)RAND_MAX) < 0.1f;
        h_neurons[i].is_input = (i < input_neurons);
        // future: give an is_output flag to neurons who link to more than their cortex
        h_neurons[i].is_negative = (rand() / (float)RAND_MAX) < 0.25f;
        h_neurons[i].id.cortex_id = cortex_number;
        h_neurons[i].id.neuron_index = i;
    }

	// Create link Tables for each neuron
    for (int i = 0; i < neuron_count; ++i) {
        std::vector<NeuronID> targets;

        // Input link
        if (h_neurons[i].is_input) {
            targets.emplace_back(-cortex_number, i, true);
        }

        // Recursive link
        if (h_neurons[i].is_recursive) {
            targets.emplace_back(cortex_number, i, false);
        }

        // Random links
        for (int j = 0; j < neuron_count; ++j) {
            if (j == i) continue;
            if ((rand() / (float)RAND_MAX) < 0.1f) {
                targets.emplace_back(cortex_number, j, false);
            }
        }

        int link_count = static_cast<int>(targets.size());
        h_neurons[i].num_links = link_count;

        // Allocate GPU-mappable memory for the links
        cudaHostAlloc((void**)&h_neurons[i].link_table, sizeof(LinkWeight) * link_count, cudaHostAllocMapped);

        // Fill in the link table
        for (int j = 0; j < link_count; ++j) {
            h_neurons[i].link_table[j].target_id = targets[j];
            h_neurons[i].link_table[j].weight = (rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        }
    }


    return h_neurons;
}