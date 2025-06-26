#pragma once

struct NeuronID {
    int cortex_id;  // Unique identifier for the cortex
    int neuron_index;  // Index of the neuron within the cortex
    bool is_input; // Flag to indicate if this is an inputID, for input channels

#if defined(__CUDACC__)
    __host__ __device__
#endif
        NeuronID(int cortex = 0, int index = 0, bool input = false)
        :cortex_id(cortex), neuron_index(index), is_input(input) {
    }
};

struct LinkWeight {
	NeuronID target_id;  // NeuronID of the target neuron
	float weight; // Weight of the link

#if defined(__CUDACC__)
    __host__ __device__
#endif
    LinkWeight(const NeuronID target = NeuronID(), float w = 0.0f)
		: weight(w), target_id(target) {
	}
};

struct Neuron {
    float threshold;
    float last_output;

    LinkWeight* link_table;  // Array of LinkWeight
    int num_links;

    bool is_input;
    bool is_recursive;
    bool is_negative;

    NeuronID id;  // Unique neuron identifier

#if defined(__CUDACC__)
    __host__ __device__
#endif
    Neuron()
        : threshold(0.5f), last_output(0.0f),
        link_table(nullptr), num_links(0),
        is_input(false), is_recursive(false), is_negative(false) {
    }
};
