//./Eliot/Simulation.cu
#include "Simulation.cuh"

void runSimulationStep(Cortex& cortex, float* inputs) {
    int threads = 256;
    int blocks = (cortex.neuron_count + threads - 1) / threads;

    scream<<<blocks, threads>>>(cortex.neurons, inputs, cortex.neuron_count, cortex.input_neurons);
    cudaDeviceSynchronize();

    adapt<<<blocks, threads>>>(cortex.neurons, inputs, cortex.neuron_count, 0.01f);
    cudaDeviceSynchronize();
}