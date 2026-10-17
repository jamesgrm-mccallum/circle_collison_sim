#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "circle.h"
#include "renderer.h"
#include "ui.h"

namespace {

// The simulation runs in a fixed 1600x900 world; the window matches it 1:1.
const int WINDOW_WIDTH = 1600;
const int WINDOW_HEIGHT = 900;
const float WORLD_WIDTH = static_cast<float>(WINDOW_WIDTH);
const float WORLD_HEIGHT = static_cast<float>(WINDOW_HEIGHT);
const int MAX_CIRCLES = 20000;

// Fixed physics timestep: frame-rate independent and reproducible. At most
// MAX_STEPS_PER_FRAME steps run per frame so a slow frame can't snowball.
const float FIXED_DT = 1.0f / 120.0f;
const int MAX_STEPS_PER_FRAME = 4;

void framebuffer_size_callback(GLFWwindow *, int width, int height) {
    // The framebuffer can be larger than the window on high-DPI displays.
    glViewport(0, 0, width, height);
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

bool is_gpu_mode(int mode) {
    return mode == MODE_GPU_COPY || mode == MODE_GPU_INTEROP;
}

} // namespace

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // The world is fixed-size; resizing would only stretch it.
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT,
                                          "GPU Circle Collision Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    int fbWidth = 0, fbHeight = 0;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    glViewport(0, 0, fbWidth, fbHeight);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Renderer renderer(WORLD_WIDTH, WORLD_HEIGHT);
    if (!renderer.init(MAX_CIRCLES)) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // CUDA must be initialised after the GL context exists, on the GPU driving it.
    const bool gpuAvailable = physics_gpu_init(MAX_CIRCLES, true);
    const bool interopAvailable = gpuAvailable && physics_gpu_register_vbo(renderer.interopBuffer());
    std::cout << "CUDA device: " << physics_gpu_device_name()
              << (interopAvailable ? " (interop available)" : "") << std::endl;

    UI ui(window);
    ui.init();

    UIState state{};
    state.mode = interopAvailable ? MODE_GPU_INTEROP : (gpuAvailable ? MODE_GPU_COPY : MODE_CPU);
    state.gpuAvailable = gpuAvailable;
    state.interopAvailable = interopAvailable;
    state.deviceName = physics_gpu_device_name();
    state.circleCount = 5000;
    state.maxCircles = MAX_CIRCLES;
    state.gravity = 100.0f;
    state.damping = 0.995f;
    state.elasticity = 0.8f;
    state.substeps = 8;
    state.iterations = 2;
    state.cpuUseGrid = true;
    state.maxThreads = cpu_max_threads();
    state.cpuThreads = state.maxThreads;
    state.vsync = true;
    glfwSwapInterval(1);

    CircleArrays circles;
    CpuSolver *cpuSolver = cpu_solver_create(MAX_CIRCLES);
    if (!circles_alloc(&circles, MAX_CIRCLES) || !cpuSolver) {
        std::cerr << "Failed to allocate simulation state" << std::endl;
        return -1;
    }
    std::vector<CircleInstance> instances(MAX_CIRCLES);

    uint32_t seed = 1;
    float maxRadius = 0.0f;
    // Initialises on the host, then pushes the same state to every consumer:
    // CPU grid, GPU buffers and the render buffer (so a paused reset still shows).
    auto resetSimulation = [&]() {
        maxRadius = circles_init_random(&circles, state.circleCount, WORLD_WIDTH, WORLD_HEIGHT, seed++);
        cpu_solver_configure_grid(cpuSolver, WORLD_WIDTH, WORLD_HEIGHT, maxRadius);
        if (gpuAvailable) {
            physics_gpu_upload(&circles, state.circleCount, maxRadius, WORLD_WIDTH, WORLD_HEIGHT);
        }
        circles_pack_instances(&circles, state.circleCount, instances.data());
        renderer.upload(instances.data(), state.circleCount);

        const float cellSize = std::max(2.0f * maxRadius, 1.0f);
        state.gridWidth = static_cast<int>(std::ceil(WORLD_WIDTH / cellSize));
        state.gridHeight = static_cast<int>(std::ceil(WORLD_HEIGHT / cellSize));
    };
    resetSimulation();

    auto lastTime = std::chrono::steady_clock::now();
    float accumulator = 0.0f;
    float fpsTimer = 0.0f;
    int frameCount = 0;

    int previousCount = state.circleCount;
    int previousMode = state.mode;
    bool previousVsync = state.vsync;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        processInput(window);

        const auto now = std::chrono::steady_clock::now();
        const float frameDt = std::chrono::duration<float>(now - lastTime).count();
        lastTime = now;

        frameCount++;
        fpsTimer += frameDt;
        if (fpsTimer >= 1.0f) {
            state.fps = frameCount / fpsTimer;
            frameCount = 0;
            fpsTimer = 0.0f;
        }

        if (state.vsync != previousVsync) {
            glfwSwapInterval(state.vsync ? 1 : 0);
            previousVsync = state.vsync;
        }

        if (state.resetSimulation || state.circleCount != previousCount) {
            resetSimulation();
            state.resetSimulation = false;
            previousCount = state.circleCount;
        }

        // The authoritative state lives on the host in CPU mode and on the
        // device in both GPU modes, so only CPU <-> GPU switches transfer it.
        if (state.mode != previousMode) {
            if (!is_gpu_mode(previousMode) && is_gpu_mode(state.mode)) {
                physics_gpu_upload(&circles, state.circleCount, maxRadius, WORLD_WIDTH, WORLD_HEIGHT);
            } else if (is_gpu_mode(previousMode) && !is_gpu_mode(state.mode)) {
                physics_gpu_download(&circles, state.circleCount);
                circles_pack_instances(&circles, state.circleCount, instances.data());
                renderer.upload(instances.data(), state.circleCount);
            }
            previousMode = state.mode;
        }

        SimParams params;
        params.gravity = state.gravity;
        params.damping = state.damping;
        params.elasticity = state.elasticity;
        params.width = WORLD_WIDTH;
        params.height = WORLD_HEIGHT;
        params.substeps = state.substeps;
        params.iterations = state.iterations;

        if (!state.paused) {
            accumulator += std::min(frameDt, 0.25f);
            int steps = static_cast<int>(accumulator / FIXED_DT);
            if (steps > MAX_STEPS_PER_FRAME) {
                steps = MAX_STEPS_PER_FRAME;
                accumulator = 0.0f;   // drop the backlog rather than spiral
            } else {
                accumulator -= steps * FIXED_DT;
            }

            if (steps > 0) {
                const auto physicsStart = std::chrono::steady_clock::now();
                const int n = state.circleCount;

                if (state.mode == MODE_CPU) {
                    SimStats stats;
                    for (int s = 0; s < steps; s++) {
                        cpu_solver_step(cpuSolver, &circles, n, &params, FIXED_DT,
                                        state.cpuUseGrid ? 1 : 0, state.cpuThreads, &stats);
                    }
                    circles_pack_instances(&circles, n, instances.data());
                    renderer.upload(instances.data(), n);
                    state.maxPenetration = stats.maxPenetration;
                    state.kineticEnergy = stats.kineticEnergy;
                    state.contacts = stats.contacts;
                } else {
                    const GpuOutput finalOutput = state.mode == MODE_GPU_INTEROP ? GPU_OUTPUT_VBO : GPU_OUTPUT_HOST;
                    for (int s = 0; s < steps; s++) {
                        // Only the last step of the frame needs to produce render data.
                        physics_gpu_step(&params, FIXED_DT, s == steps - 1 ? finalOutput : GPU_OUTPUT_NONE);
                    }
                    if (state.mode == MODE_GPU_INTEROP) {
                        renderer.markInteropUpdated();
                    } else {
                        renderer.upload(physics_gpu_host_instances(), n);
                    }
                }

                const auto physicsEnd = std::chrono::steady_clock::now();
                state.stepWallMs = std::chrono::duration<float, std::milli>(physicsEnd - physicsStart).count();
            }
        }

        if (is_gpu_mode(state.mode)) {
            GpuFrameStats gpuStats;
            if (physics_gpu_poll_stats(&gpuStats)) {
                state.gpuStepMs = gpuStats.stepMs;
                state.maxPenetration = gpuStats.sim.maxPenetration;
                state.kineticEnergy = gpuStats.sim.kineticEnergy;
                state.contacts = gpuStats.sim.contacts;
            }
        }

        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        renderer.draw(state.circleCount);

        ui.newFrame();
        ui.render(state);

        glfwSwapBuffers(window);
    }

    // Unregister the interop buffer before GL deletes it.
    physics_gpu_shutdown();
    cpu_solver_destroy(cpuSolver);
    circles_free(&circles);

    ui.shutdown();
    renderer.cleanup();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
