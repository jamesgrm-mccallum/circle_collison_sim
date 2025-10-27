#ifndef RENDERER_H
#define RENDERER_H

#include "circle.h"

class Renderer {
public:
    Renderer(int width, int height);
    ~Renderer();
    
    void init();
    void render(Circle *circles, int count);
    void render_interop(int count); // For CUDA-OpenGL interop
    void cleanup();
    
    unsigned int getVBO() const { return vbo; }
    
private:
    int width, height;
    unsigned int shaderProgram;
    unsigned int VAO, VBO, instanceVBO;
    unsigned int vbo; // For interop
    
    void setupCircleMesh();
    void setupInstanceBuffer();
    unsigned int compileShader(const char *vertexPath, const char *fragmentPath);
};

#endif // RENDERER_H