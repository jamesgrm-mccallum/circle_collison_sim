#ifndef UI_H
#define UI_H

#include <GLFW/glfw3.h>

struct UIState {
    int circleCount;
    float gravity;
    float damping;
    float elasticity;
    bool useCPU;
    bool useInterop;
    bool paused;
    float fps;
    float cpuTime;
    float gpuTime;
    bool resetSimulation;
    bool interopAvailable;
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