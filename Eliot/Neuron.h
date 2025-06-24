// Neuron.h
#pragma once

struct Neuron {
	float threshold;
	float last_output;
	float* weights;
	bool is_input;
	bool is_recursive;
	bool is_negative;
	int num_weights;
	char id;

	__host__ __device__ Neuron()
		: threshold(0.5f), last_output(0.0f), weights(nullptr), is_input(false), is_recursive(false), is_negative(false), num_weights(0) {}
};
