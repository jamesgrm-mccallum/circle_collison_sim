#include "ui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

UI::UI(GLFWwindow *win) : window(win) {}

UI::~UI() {}

void UI::init() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    // Must match the 3.3 core context created in main.cpp
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

void UI::newFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

static void radio_mode(const char *label, UIState &state, int mode, bool enabled, const char *disabledReason) {
    if (!enabled) ImGui::BeginDisabled();
    if (ImGui::RadioButton(label, state.mode == mode)) state.mode = mode;
    if (!enabled) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("%s", disabledReason);
        }
    }
}

void UI::render(UIState &state) {
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Simulation Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Text("Performance");
    ImGui::Text("FPS: %.1f", state.fps);
    ImGui::Text("Physics (host wall): %.2f ms", state.stepWallMs);
    if (state.mode != MODE_CPU) {
        ImGui::Text("Physics (GPU events): %.2f ms", state.gpuStepMs);
    }
    ImGui::Text("Contacts: %u   Max overlap: %.2f px", state.contacts, state.maxPenetration);
    ImGui::Text("Kinetic energy: %.3g", state.kineticEnergy);
    ImGui::Text("Grid: %d x %d cells", state.gridWidth, state.gridHeight);
    ImGui::Separator();

    ImGui::Text("Mode");
    radio_mode("CPU", state, MODE_CPU, true, "");
    radio_mode("GPU (Copy)", state, MODE_GPU_COPY, state.gpuAvailable,
               "No usable CUDA device.");
    radio_mode("GPU (Interop)", state, MODE_GPU_INTEROP, state.interopAvailable,
               "CUDA-OpenGL interop is not available on this system.\nUse GPU (Copy) instead.");
    ImGui::TextDisabled("Device: %s", state.deviceName);
    if (state.mode == MODE_CPU) {
        ImGui::Checkbox("Spatial grid (off = all pairs, O(n^2))", &state.cpuUseGrid);
        ImGui::SliderInt("CPU threads", &state.cpuThreads, 1, state.maxThreads);
    }
    ImGui::Separator();

    ImGui::Text("Simulation Parameters");
    ImGui::SliderInt("Circle Count", &state.circleCount, 10, state.maxCircles);
    ImGui::SliderFloat("Gravity", &state.gravity, 0.0f, 500.0f);
    ImGui::SliderFloat("Damping", &state.damping, 0.9f, 1.0f);
    ImGui::SliderFloat("Elasticity", &state.elasticity, 0.1f, 1.0f);
    ImGui::SliderInt("Sub-steps", &state.substeps, 1, 16);
    ImGui::SliderInt("Iterations / sub-step", &state.iterations, 1, 8);
    ImGui::Separator();

    ImGui::Checkbox("Paused", &state.paused);
    ImGui::SameLine();
    ImGui::Checkbox("VSync", &state.vsync);

    if (ImGui::Button("Reset Simulation")) {
        state.resetSimulation = true;
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UI::shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}
