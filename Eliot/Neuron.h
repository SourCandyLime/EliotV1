//./Eliot/Neuron.h
#pragma once

struct Neuron {
    float threshold;
    float last_output;
    float* weights;
    int* linked_indices; // indices of other neurons to pull inputs from
    bool is_input;
    bool is_recursive;
    bool is_negative;
    int num_weights;
    char id[64];  // use snprintf to fill this
#if defined(__CUDACC__)
    __host__ __device__
#endif
        Neuron()
        : threshold(0.5f), last_output(0.0f), weights(nullptr),
        is_input(false), is_recursive(false), is_negative(false), num_weights(0) {
    }
};
