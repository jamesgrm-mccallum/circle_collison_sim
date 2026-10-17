#ifndef UI_H
#define UI_H

#include <GLFW/glfw3.h>

enum SimMode {
    MODE_CPU = 0,
    MODE_GPU_COPY = 1,
    MODE_GPU_INTEROP = 2
};

struct UIState {
    // Mode
    int  mode;
    bool gpuAvailable;
    bool interopAvailable;
    const char *deviceName;

    // Simulation
    int   circleCount;
    int   maxCircles;
    float gravity;
    float damping;
    float elasticity;
    int   substeps;
    int   iterations;
    bool  paused;
    bool  resetSimulation;

    // CPU options
    bool cpuUseGrid;
    int  cpuThreads;
    int  maxThreads;

    // Display
    bool vsync;

    // Metrics
    float fps;
    float stepWallMs;     // host time spent in physics this frame
    float gpuStepMs;      // GPU time from CUDA events
    float maxPenetration;
    float kineticEnergy;
    unsigned contacts;
    int   gridWidth;
    int   gridHeight;
};

class UI {
public:
    UI(GLFWwindow *window);
    ~UI();

    void init();
    void newFrame();
    void render(UIState &state);
    void shutdown();

private:
    GLFWwindow *window;
};

#endif // UI_H
