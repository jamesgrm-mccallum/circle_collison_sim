#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <chrono>

#include "circle.h"
#include "renderer.h"
#include "ui.h"

const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const int MAX_CIRCLES = 5000;

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, 
                                          "GPU Circle Collision Simulation", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    
    // Load OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }
    
    // CRITICAL: Set CUDA device for OpenGL interop AFTER OpenGL context is created
    std::cout << "Setting up CUDA-OpenGL interop..." << std::endl;
    
    // Find CUDA device that can work with OpenGL
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;
    
    int cudaDevice = -1;
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device " << i << ": " << prop.name << std::endl;
        
        // Just select the first available device
        cudaDevice = i;
        std::cout << "  - Selected for OpenGL interop" << std::endl;
        break;
    }
    
    if (cudaDevice == -1) {
        std::cerr << "No CUDA device available for OpenGL interop" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Set the CUDA device for this thread
    cudaError_t err = cudaSetDevice(cudaDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        glfwTerminate();
        return -1;
    }
    
    std::cout << "CUDA device set successfully for OpenGL interop" << std::endl;
    
    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Initialize components
    Renderer renderer(WINDOW_WIDTH, WINDOW_HEIGHT);
    renderer.init();
    
    std::cout << "Renderer initialized, VBO ID: " << renderer.getVBO() << std::endl;
    
    UI ui(window);
    ui.init();
    
    // Initialize simulation state
    UIState uiState = {
        .circleCount = 500,
        .gravity = 100.0f,
        .damping = 0.995f,
        .elasticity = 0.8f,
        .useCPU = true,
        .useInterop = false,
        .paused = false,
        .fps = 0.0f,
        .cpuTime = 0.0f,
        .gpuTime = 0.0f,
        .resetSimulation = false,
        .interopAvailable = false  // Will be set after registration attempt
    };
    
    SimParams simParams = {
        .gravity = uiState.gravity,
        .damping = uiState.damping,
        .elasticity = uiState.elasticity,
        .width = (float)WINDOW_WIDTH,
        .height = (float)WINDOW_HEIGHT
    };
    
    // Allocate circles
    Circle *circles = new Circle[MAX_CIRCLES];
    Circle *d_circles = nullptr;
    
    physics_cpu_init(circles, uiState.circleCount, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
    physics_gpu_init(&d_circles, MAX_CIRCLES);
    
    // Setup CUDA-OpenGL interop - MUST happen after OpenGL context AND cudaGLSetGLDevice
    std::cout << "Registering VBO for CUDA-OpenGL interop..." << std::endl;
    physics_gpu_register_vbo(renderer.getVBO(), MAX_CIRCLES);
    
    // Copy initial data to both GPU buffers
    cudaMemcpy(d_circles, circles, uiState.circleCount * sizeof(Circle), cudaMemcpyHostToDevice);
    physics_gpu_sync_persistent(circles, uiState.circleCount);
    
    // Timing variables
    auto lastTime = std::chrono::high_resolution_clock::now();
    float deltaTime = 0.0f;
    int frameCount = 0;
    float fpsTimer = 0.0f;
    
    int previousCircleCount = uiState.circleCount;
    bool previousUseCPU = uiState.useCPU;
    bool previousUseInterop = uiState.useInterop;
    
    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;
        
        // FPS calculation
        frameCount++;
        fpsTimer += deltaTime;
        if (fpsTimer >= 1.0f) {
            uiState.fps = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0.0f;
        }
        
        processInput(window);
        
        // Update simulation parameters
        simParams.gravity = uiState.gravity;
        simParams.damping = uiState.damping;
        simParams.elasticity = uiState.elasticity;
        
        // Handle reset or circle count change
        if (uiState.resetSimulation || previousCircleCount != uiState.circleCount) {
            physics_cpu_init(circles, uiState.circleCount, (float)WINDOW_WIDTH, (float)WINDOW_HEIGHT);
            // Copy to both GPU buffers
            cudaMemcpy(d_circles, circles, uiState.circleCount * sizeof(Circle), 
                      cudaMemcpyHostToDevice);
            physics_gpu_sync_persistent(circles, uiState.circleCount);
            
            uiState.resetSimulation = false;
            previousCircleCount = uiState.circleCount;
        }
        
        // Handle mode switching - sync data between CPU and GPU
        if (previousUseCPU != uiState.useCPU || previousUseInterop != uiState.useInterop) {
            if (uiState.useCPU) {
                // Switching to CPU mode - copy GPU data back to CPU
                if (previousUseInterop) {
                    // Data is in persistent buffer, we can't directly copy it
                    // Just keep current CPU state
                } else {
                    // Data is in d_circles
                    cudaMemcpy(circles, d_circles, uiState.circleCount * sizeof(Circle), 
                              cudaMemcpyDeviceToHost);
                }
            } else if (uiState.useInterop) {
                // Switching to interop mode - ensure persistent buffer is synced
                if (previousUseCPU) {
                    // Coming from CPU mode - sync CPU data to persistent buffer
                    physics_gpu_sync_persistent(circles, uiState.circleCount);
                } else {
                    // Coming from GPU copy mode - copy d_circles to persistent buffer
                    cudaMemcpy(circles, d_circles, uiState.circleCount * sizeof(Circle), 
                              cudaMemcpyDeviceToHost);
                    physics_gpu_sync_persistent(circles, uiState.circleCount);
                }
            } else {
                // Switching to GPU copy mode
                if (previousUseCPU) {
                    // Copy CPU data to d_circles
                    cudaMemcpy(d_circles, circles, uiState.circleCount * sizeof(Circle), 
                              cudaMemcpyHostToDevice);
                } else if (previousUseInterop) {
                    // Can't easily copy from persistent buffer, use current CPU state
                    cudaMemcpy(d_circles, circles, uiState.circleCount * sizeof(Circle), 
                              cudaMemcpyHostToDevice);
                }
            }
            
            previousUseCPU = uiState.useCPU;
            previousUseInterop = uiState.useInterop;
        }
        
        // Physics update
        if (!uiState.paused) {
            float dt = deltaTime;
            if (dt > 0.033f) dt = 0.033f; // Cap at 30 FPS
            
            auto physicsStart = std::chrono::high_resolution_clock::now();
            
            if (uiState.useCPU) {
                // CPU mode
                physics_cpu_update(circles, uiState.circleCount, &simParams, dt);
                auto physicsEnd = std::chrono::high_resolution_clock::now();
                uiState.cpuTime = std::chrono::duration<float, std::milli>(physicsEnd - physicsStart).count();
                uiState.gpuTime = 0.0f;
            } else if (uiState.useInterop) {
                // GPU mode with CUDA-OpenGL interop
                physics_gpu_update_vbo(renderer.getVBO(), uiState.circleCount, &simParams, dt);
                cudaDeviceSynchronize();
                auto physicsEnd = std::chrono::high_resolution_clock::now();
                uiState.gpuTime = std::chrono::duration<float, std::milli>(physicsEnd - physicsStart).count();
                uiState.cpuTime = 0.0f;
            } else {
                // GPU mode with copy
                physics_gpu_update(circles, d_circles, uiState.circleCount, &simParams, dt);
                auto physicsEnd = std::chrono::high_resolution_clock::now();
                uiState.gpuTime = std::chrono::duration<float, std::milli>(physicsEnd - physicsStart).count();
                uiState.cpuTime = 0.0f;
            }
        }
        
        // Render
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        if (uiState.useInterop) {
            renderer.render_interop(uiState.circleCount);
        } else {
            renderer.render(circles, uiState.circleCount);
        }
        
        // UI
        ui.newFrame();
        ui.render(uiState);
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    // Cleanup
    physics_gpu_unregister_vbo();
    physics_gpu_cleanup(d_circles);
    delete[] circles;
    
    renderer.cleanup();
    ui.shutdown();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}