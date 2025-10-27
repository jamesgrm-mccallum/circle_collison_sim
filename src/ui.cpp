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
    ImGui_ImplOpenGL3_Init("#version 130");
}

void UI::newFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UI::render(UIState &state) {
    ImGui::Begin("Simulation Controls");
    
    ImGui::Text("Performance");
    ImGui::Text("FPS: %.1f", state.fps);
    ImGui::Text("CPU Time: %.2f ms", state.cpuTime);
    ImGui::Text("GPU Time: %.2f ms", state.gpuTime);
    ImGui::Separator();
    
    ImGui::Text("Mode Selection");
    if (ImGui::RadioButton("CPU Mode", state.useCPU)) {
        state.useCPU = true;
        state.useInterop = false;
    }
    if (ImGui::RadioButton("GPU Mode (Copy)", !state.useCPU && !state.useInterop)) {
        state.useCPU = false;
        state.useInterop = false;
    }
    
    // Disable interop option if not available
    if (!state.interopAvailable) {
        ImGui::BeginDisabled();
    }
    if (ImGui::RadioButton("GPU Mode (Interop)", state.useInterop)) {
        state.useCPU = false;
        state.useInterop = true;
    }
    if (!state.interopAvailable) {
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("CUDA-OpenGL interop not available on this system.\nUse GPU Mode (Copy) instead.");
        }
    }
    
    ImGui::Separator();
    
    ImGui::Text("Simulation Parameters");
    ImGui::SliderInt("Circle Count", &state.circleCount, 10, 5000);
    ImGui::SliderFloat("Gravity", &state.gravity, 0.0f, 500.0f);
    ImGui::SliderFloat("Damping", &state.damping, 0.9f, 1.0f);
    ImGui::SliderFloat("Elasticity", &state.elasticity, 0.1f, 1.0f);
    ImGui::Separator();
    
    ImGui::Checkbox("Paused", &state.paused);
    
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