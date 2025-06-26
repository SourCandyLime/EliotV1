//./Eliot/Neuron.cu
#include "Neuron.cuh"
#include <iostream>

__global__ void scream(Neuron* neurons, float* inputs, int total_neurons, int input_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_neurons) return;

    Neuron& n = neurons[idx];
    float sum = 0.0f;

	// Calculate the weighted sum of all links
    for (int i = 0; i < n.num_links; ++i) {
        LinkWeight& link = n.link_table[i];
        if (link.target_id.is_input) {
            sum += link.weight * inputs[n.id.neuron_index];
        }
        else {
            // grab neuron output via link table
            for (int i = 0; i < total_neurons; ++i) {
                if (link.target_id.cortex_id == neurons[i].id.cortex_id && link.target_id.neuron_index == neurons[i].id.neuron_index) {
					sum += link.weight * neurons[i].last_output;
                }
            }
        }
    }

    if (n.is_negative) {
        n.last_output = (sum <= n.threshold) ? -1.0f : 0.0f;
    } 
    else {
        n.last_output = (sum >= n.threshold) ? 1.0f : 0.0f;
    }
}

__global__ void adapt(Neuron* neurons, float* inputs, int neuron_count, float learn_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neuron_count) return;
    Neuron& n = neurons[idx];

	float history = n.last_output;

    for (int i = 0; i < n.num_links; ++i) {
        LinkWeight& link = n.link_table[i];
        for (int i = 0; i < neuron_count; ++i) {
            if (link.target_id.cortex_id == neurons[i].id.cortex_id && link.target_id.neuron_index == neurons[i].id.neuron_index) {
                if (link.target_id.is_input) {
                    // Adjust input link weights based on input
                    link.weight += (n.is_negative) ? -learn_rate * inputs[link.target_id.neuron_index] * history : learn_rate * inputs[link.target_id.neuron_index] * history; // Adjust weight based on input
                } else {
                    // Adjust output link weights based on last output
                    link.weight += (n.is_negative) ? -learn_rate * neurons[i].last_output * history : learn_rate * neurons[i].last_output * history; // Adjust weight based on last output
                }
            }
        }
	}

    //adjust threshold based on last output
    if (history != 0.0f) {
        n.threshold += (n.is_negative) ? -learn_rate : learn_rate; // Increase threshold if neuron fired to prevent it from firing too easily
    }
    else {
        n.threshold += (n.is_negative) ? learn_rate : -learn_rate; // Decrease threshold if neuron did not fire to make it more sensitive
    }

}