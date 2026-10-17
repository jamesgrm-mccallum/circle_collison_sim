#ifndef RENDERER_H
#define RENDERER_H

#include "circle.h"

// Draws circles as instanced quads; the fragment shader turns each quad into an
// anti-aliased disc. Instance data is 16 bytes per circle (CircleInstance).
//
// Two instance buffers are used: a streamed one that the CPU and GPU (Copy)
// modes upload into, and one registered with CUDA that GPU (Interop) mode writes
// directly. The interop buffer is never re-specified with glBufferData, since
// that would invalidate the CUDA registration.
class Renderer {
public:
    Renderer(float worldWidth, float worldHeight);
    ~Renderer();

    bool init(int capacity);
    void cleanup();

    // Upload instance data from host memory (CPU and GPU Copy modes).
    void upload(const CircleInstance *instances, int count);
    // Call after CUDA has written the interop buffer.
    void markInteropUpdated() { drawFromInterop = true; }
    void draw(int count);

    unsigned int interopBuffer() const { return interopVBO; }

private:
    float worldWidth, worldHeight;
    int capacity = 0;
    unsigned int program = 0;
    unsigned int quadVBO = 0;
    unsigned int streamVBO = 0, streamVAO = 0;
    unsigned int interopVBO = 0, interopVAO = 0;
    bool drawFromInterop = false;

    void setupVAO(unsigned int vao, unsigned int instanceVBO);
};

#endif // RENDERER_H
