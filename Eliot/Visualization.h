//./Eliot/Visualization.cuh
#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "Cortex.h"

GLuint LoadShaders(const char* vertex_path, const char* geometry_path, const char* fragment_path);
GLFWwindow* initWindow(const char* title, int neuron_count, float point_size = 4.0f);
void drawNeuronGrid(const Cortex& cortex);
