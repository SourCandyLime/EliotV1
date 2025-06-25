//./Eliot/Shader.cpp
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <glad/glad.h>
#include "Shader.h"

GLuint LoadShaders(const char* vertex_path, const char* fragment_path) {
    std::ifstream vFile(vertex_path);
    std::ifstream fFile(fragment_path);

    std::stringstream vStream, fStream;
    vStream << vFile.rdbuf();
    fStream << fFile.rdbuf();
    std::string vSource = vStream.str();
    std::string fSource = fStream.str();

    const char* vCode = vSource.c_str();
    const char* fCode = fSource.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vCode, nullptr);
    glCompileShader(vertexShader);

    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(vertexShader, 512, nullptr, log);
        std::cerr << "Vertex Shader Error: " << log << "\n";
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fCode, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(fragmentShader, 512, nullptr, log);
        std::cerr << "Fragment Shader Error: " << log << "\n";
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);

    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, nullptr, log);
        std::cerr << "Shader Link Error: " << log << "\n";
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return program;
}
