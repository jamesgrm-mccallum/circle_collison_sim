cmake_minimum_required(VERSION 3.18)
project(GPUCircleSimulation LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(CUDA REQUIRED)

# Additional OpenGL libraries for Linux
if(UNIX AND NOT APPLE)
    find_package(X11 REQUIRED)
endif()

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/glfw)

# GLAD
add_library(glad external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)

# ImGui
set(IMGUI_DIR external/imgui)
add_library(imgui
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC 
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw ${OPENGL_LIBRARIES})

# Main executable
add_executable(circle_sim
    src/main.cpp
    src/renderer.cpp
    src/ui.cpp
    src/physics_cpu.c
)

# Compile CUDA file separately as a library
add_library(physics_gpu src/physics_gpu.cu)
target_include_directories(physics_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

target_include_directories(circle_sim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(circle_sim
    glad
    glfw
    imgui
    physics_gpu
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBRARIES}
)

# Link X11 on Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(circle_sim ${X11_LIBRARIES})
endif()

# Set CUDA architecture
set_target_properties(physics_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;86;89"
)

set_target_properties(circle_sim PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Copy shaders to build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/shaders DESTINATION ${CMAKE_BINARY_DIR})