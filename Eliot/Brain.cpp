#include <iostream>
#include <cuda_runtime.h>
#include "Cortex.h"
#include "Neuron.cuh"
#include "Simulation.cuh"
#include "Visualization.h"

static void printNeuronStates(Neuron* neurons, int count) {
    for (int i = 0; i < count; ++i)
        std::cout << "[" << i << "] output: " << neurons[i].last_output << "\n";
}

int main() {
    std::cout << "Booting up Eliot...\n";

    Cortex audioCortex("Audio", 4096, 1024);
    generateLinkMap(audioCortex);

    float* inputs = nullptr;
    cudaHostAlloc((void**)&inputs, sizeof(float) * audioCortex.input_neurons, cudaHostAllocMapped);

    cudaDeviceSynchronize();

    cudaFree(0); // this forces CUDA context init

    for (int i = 0; i < audioCortex.input_neurons; ++i)
        inputs[i] = 0.0f; // Start clean

    std::cout << "Allocating input buffer: " << audioCortex.input_neurons << " floats\n";

    GLFWwindow* window = initWindow("Neuron Visualizer", audioCortex.neuron_count, 8.0f);
    if (!window) return -1;

    while (!glfwWindowShouldClose(window)) {
        // generate new random input
        for (int i = 0; i < audioCortex.input_neurons; ++i)
			inputs[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random input between -1 and 1
        
        runSimulationStep(audioCortex, inputs);
		printNeuronStates(audioCortex.neurons, audioCortex.neuron_count);
        drawNeuronGrid(audioCortex);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    cudaFree(inputs);
    return 0;
}
