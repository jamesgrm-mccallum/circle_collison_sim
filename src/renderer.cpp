#include "renderer.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

Renderer::Renderer(int w, int h) : width(w), height(h), 
    shaderProgram(0), VAO(0), VBO(0), instanceVBO(0), vbo(0) {}

Renderer::~Renderer() {
    cleanup();
}

static std::string readFile(const char *path) {
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

unsigned int Renderer::compileShader(const char *vertexPath, const char *fragmentPath) {
    std::string vertexCode = readFile(vertexPath);
    std::string fragmentCode = readFile(fragmentPath);
    
    const char *vShaderCode = vertexCode.c_str();
    const char *fShaderCode = fragmentCode.c_str();
    
    unsigned int vertex, fragment;
    int success;
    char infoLog[512];
    
    // Vertex shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    glGetShaderiv(vertex, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertex, 512, NULL, infoLog);
        std::cerr << "Vertex shader compilation failed:\n" << infoLog << std::endl;
    }
    
    // Fragment shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    glGetShaderiv(fragment, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragment, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation failed:\n" << infoLog << std::endl;
    }
    
    // Shader program
    unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed:\n" << infoLog << std::endl;
    }
    
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    
    return program;
}

void Renderer::setupCircleMesh() {
    const int segments = 32;
    float vertices[segments * 2];
    
    for (int i = 0; i < segments; i++) {
        float angle = 2.0f * 3.14159f * i / segments;
        vertices[i * 2 + 0] = std::cos(angle);
        vertices[i * 2 + 1] = std::sin(angle);
    }
    
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void Renderer::setupInstanceBuffer() {
    glGenBuffers(1, &instanceVBO);
}

void Renderer::init() {
    shaderProgram = compileShader("shaders/circle.vert", "shaders/circle.frag");
    setupCircleMesh();
    setupInstanceBuffer();
    
    // Create VBO for CUDA interop (needs to be large enough)
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Allocate 10000 circles * 6 floats per circle
    glBufferData(GL_ARRAY_BUFFER, 10000 * 6 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    std::cout << "Renderer initialized. VBO ID: " << vbo << std::endl;
}

void Renderer::render(Circle *circles, int count) {
    glUseProgram(shaderProgram);
    
    // Set projection matrix
    float left = 0.0f, right = (float)width;
    float bottom = (float)height, top = 0.0f;
    float ortho[16] = {
        2.0f/(right-left), 0, 0, 0,
        0, 2.0f/(top-bottom), 0, 0,
        0, 0, -1, 0,
        -(right+left)/(right-left), -(top+bottom)/(top-bottom), 0, 1
    };
    
    int projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, ortho);
    
    // Prepare instance data
    float *instanceData = new float[count * 6]; // x, y, r, g, b, radius
    for (int i = 0; i < count; i++) {
        instanceData[i * 6 + 0] = circles[i].x;
        instanceData[i * 6 + 1] = circles[i].y;
        instanceData[i * 6 + 2] = circles[i].r;
        instanceData[i * 6 + 3] = circles[i].g;
        instanceData[i * 6 + 4] = circles[i].b;
        instanceData[i * 6 + 5] = circles[i].radius;
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, count * 6 * sizeof(float), instanceData, GL_DYNAMIC_DRAW);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    
    // Instance attributes
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);
    
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);
    
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 32, count);
    
    glBindVertexArray(0);
    delete[] instanceData;
}

void Renderer::render_interop(int count) {
    glUseProgram(shaderProgram);
    
    float left = 0.0f, right = (float)width;
    float bottom = (float)height, top = 0.0f;
    float ortho[16] = {
        2.0f/(right-left), 0, 0, 0,
        0, 2.0f/(top-bottom), 0, 0,
        0, 0, -1, 0,
        -(right+left)/(right-left), -(top+bottom)/(top-bottom), 0, 1
    };
    
    int projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, ortho);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);
    
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);
    
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(3);
    glVertexAttribDivisor(3, 1);
    
    glDrawArraysInstanced(GL_TRIANGLE_FAN, 0, 32, count);
    glBindVertexArray(0);
}

void Renderer::cleanup() {
    if (VAO) glDeleteVertexArrays(1, &VAO);
    if (VBO) glDeleteBuffers(1, &VBO);
    if (instanceVBO) glDeleteBuffers(1, &instanceVBO);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (shaderProgram) glDeleteProgram(shaderProgram);
}