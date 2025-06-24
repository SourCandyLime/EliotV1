// Kernels.cu
#include "Neuron.cuh"

__global__ void scream(Neuron* neurons, float* inputs, int neuron_count, int input_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= neuron_count) return;

    Neuron& n = neurons[idx];
    float sum = 0.0f;

    // Adjust for Negative Neurons
    if (n.negative) {
        for (int i = 0; i < n.num_weights && i < input_count; ++i) {
            sum -= inputs[i] * n.weights[i]; // Negate the contribution of inputs for negative neurons
        }
        n.last_output = (sum <= n.threshold) ? -1.0f : 0.0f; // Fire if sum is less than or equal to threshold
        
    }
    else {
        for (int i = 0; i < n.num_weights && i < input_count; ++i) {
            sum += inputs[i] * n.weights[i];
        }
        n.last_output = (sum >= n.threshold) ? 1.0f : 0.0f;
        
    }
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
        n.threshold += (n.negative) ? -learn_rate : learn_rate; // Increase threshold if neuron fired to prevent it from firing too easily
    }
    else {
        n.threshold += (n.negative) ? learn_rate : -learn_rate; // Decrease threshold if neuron did not fire to make it more sensitive
    }

}
