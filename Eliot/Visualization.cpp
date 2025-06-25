//./Eliot/Visualization.cu
#include "Visualization.h"

GLuint VAO, VBO;
GLuint shaderProgram;
int windowPixelWidth = 800;
float pointSize; // Default point size

GLuint LoadShaders(const char* vertex_path, const char* geometry_path, const char* fragment_path) {
    auto load = [](const char* path) -> std::string {
        std::ifstream file(path);
        std::stringstream ss;
        ss << file.rdbuf();
        return ss.str();
        };

    auto compileShader = [](GLuint type, const char* source) -> GLuint {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char log[512];
            glGetShaderInfoLog(shader, 512, nullptr, log);
            std::cerr << "Shader Compile Error:\n" << log << "\n";
        }
        return shader;
        };

    std::string vSource = load(vertex_path);
    std::string gSource = load(geometry_path);
    std::string fSource = load(fragment_path);

    GLuint vs = compileShader(GL_VERTEX_SHADER, vSource.c_str());
    GLuint gs = compileShader(GL_GEOMETRY_SHADER, gSource.c_str());
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fSource.c_str());

    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, gs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Shader Link Error:\n" << log << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(gs);
    glDeleteShader(fs);
    return program;
}

GLFWwindow* initWindow(const char* title, int neuron_count, float point_size) {
	int side = (int)std::sqrt(neuron_count) * point_size;
	pointSize = point_size; // Set the point size for rendering

    windowPixelWidth = side;

    if (!glfwInit()) {
        std::cerr << "GLFW failed to init\n";
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(side, side, title, nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        return nullptr;
    }

    shaderProgram = LoadShaders("Neuron.vert", "Neuron.geom", "Neuron.frag");

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    // Allocate space but don’t fill yet
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * neuron_count * 4, NULL, GL_DYNAMIC_DRAW);

    // Attributes: vec2 aPos, float aOutput, float aIsInput
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);

    glViewport(0, 0, side, side);
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Set background color
    return window;
}

void drawNeuronGrid(const Cortex& cortex) {
    glClear(GL_COLOR_BUFFER_BIT);

    int size = cortex.neuron_count;
    int width = (int)std::sqrt(size);
	int height = width; // Assume square grid for simplicity

    std::vector<float> vertexData(size * 4); // 4 floats per neuron

    for (int i = 0; i < size; ++i) {
        float x = ((i % width) + 0.5f) / (float)width * 2.0f - 1.0f;
        float y = ((i / width) + 0.5f) / (float)height * 2.0f - 1.0f;
        
        vertexData[i * 4 + 0] = x;
        vertexData[i * 4 + 1] = y;
        vertexData[i * 4 + 2] = cortex.neurons[i].last_output;
        vertexData[i * 4 + 3] = cortex.neurons[i].is_input ? 1.0f : 0.0f;
    }

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertexData.size() * sizeof(float), vertexData.data());

    GLint sizeLoc = glGetUniformLocation(shaderProgram, "size");
    glUniform1f(sizeLoc, 2.0f / std::sqrt(cortex.neuron_count));  // Normalize to NDC

    glDrawArrays(GL_POINTS, 0, size);
}
