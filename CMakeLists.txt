cmake_minimum_required(VERSION 3.18)
project(GPUCircleSimulation LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(CIRCLE_SIM_ENABLE_CUDA "Build the CUDA solver when a CUDA compiler is available" ON)
option(CIRCLE_SIM_BUILD_APP "Build the windowed app (needs external/glfw, external/glad, external/imgui)" ON)

# ── CUDA (optional: without it the app runs in CPU mode only) ────────

set(CIRCLE_SIM_HAVE_CUDA OFF)
if(CIRCLE_SIM_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        set(CIRCLE_SIM_HAVE_CUDA ON)
    else()
        message(STATUS "No CUDA compiler found: building CPU-only")
    endif()
endif()

if(MSVC)
    set(CIRCLE_SIM_WARNINGS /W4)
else()
    set(CIRCLE_SIM_WARNINGS -Wall -Wextra)
endif()

find_package(OpenMP COMPONENTS C)
if(CIRCLE_SIM_HAVE_CUDA OR CIRCLE_SIM_BUILD_APP)
    # cuda_gl_interop.h includes the GL headers, so the GPU solver needs them too
    find_package(OpenGL REQUIRED COMPONENTS OpenGL)
endif()

# ── Physics (no graphics dependencies) ───────────────────────────────

add_library(physics_cpu STATIC src/physics_cpu.c)
target_include_directories(physics_cpu PUBLIC src)
target_compile_options(physics_cpu PRIVATE ${CIRCLE_SIM_WARNINGS})
if(OpenMP_C_FOUND)
    target_link_libraries(physics_cpu PUBLIC OpenMP::OpenMP_C)
else()
    message(STATUS "OpenMP not found: CPU solver will be single-threaded")
endif()
if(UNIX)
    target_link_libraries(physics_cpu PUBLIC m)
endif()

if(CIRCLE_SIM_HAVE_CUDA)
    add_library(physics_gpu STATIC src/physics_gpu.cu)
    target_link_libraries(physics_gpu PUBLIC physics_cpu CUDA::cudart PRIVATE OpenGL::OpenGL)
    target_compile_options(physics_gpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo -Xcompiler=-Wall>)
else()
    add_library(physics_gpu STATIC src/physics_gpu_stub.cpp)
    target_link_libraries(physics_gpu PUBLIC physics_cpu)
endif()

# Headless benchmark: prints CPU (1 thread / OpenMP) and GPU timings
add_executable(circle_bench src/benchmark.cpp)
target_link_libraries(circle_bench PRIVATE physics_gpu)
target_compile_options(circle_bench PRIVATE ${CIRCLE_SIM_WARNINGS})

# ── Tests ────────────────────────────────────────────────────────────

enable_testing()
add_executable(test_physics_cpu tests/test_physics_cpu.c)
target_link_libraries(test_physics_cpu PRIVATE physics_cpu)
target_compile_options(test_physics_cpu PRIVATE ${CIRCLE_SIM_WARNINGS})
add_test(NAME physics_cpu COMMAND test_physics_cpu)

# ── Windowed app ─────────────────────────────────────────────────────

if(CIRCLE_SIM_BUILD_APP)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(external/glfw)

    add_library(glad STATIC external/glad/src/glad.c)
    target_include_directories(glad PUBLIC external/glad/include)

    set(IMGUI_DIR external/imgui)
    add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw OpenGL::OpenGL)

    add_executable(circle_sim
        src/main.cpp
        src/renderer.cpp
        src/ui.cpp
    )
    target_link_libraries(circle_sim PRIVATE physics_gpu glad glfw imgui OpenGL::OpenGL)
    target_compile_options(circle_sim PRIVATE ${CIRCLE_SIM_WARNINGS})
endif()
