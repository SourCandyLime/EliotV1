//./Eliot/Neuron.cu
#include "Neuron.cuh"
#include <iostream>

__global__ void scream(Neuron* neurons, int neuron_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neuron_count) return;

    Neuron& n = neurons[idx];
    float sum = 0.0f;

    for (int i = 0; i < n.num_weights; ++i) {
        int source_idx = n.linked_indices[i];
        float input = neurons[source_idx].last_output;
        sum += (n.is_negative ? -1.0f : 1.0f) * input * n.weights[i];
    }

    if (n.is_negative)
        n.last_output = (sum <= n.threshold) ? -1.0f : 0.0f;
    else
        n.last_output = (sum >= n.threshold) ? 1.0f : 0.0f;
}

__global__ void adapt(Neuron* neurons, float* inputs, int neuron_count, float learn_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neuron_count) return;
    Neuron& n = neurons[idx];
    float history = n.last_output;
    for (int i = 0; i < n.num_weights; ++i) {
        n.weights[i] += learn_rate * history * inputs[i]; // Adjust weights based on last output and if input contributed to fire
    }
    //adjust threshold based on last output
    if (history != 0.0f) {
        n.threshold += (n.is_negative) ? -learn_rate : learn_rate; // Increase threshold if neuron fired to prevent it from firing too easily
    }
    else {
        n.threshold += (n.is_negative) ? learn_rate : -learn_rate; // Decrease threshold if neuron did not fire to make it more sensitive
    }

}