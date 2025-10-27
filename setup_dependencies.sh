#!/bin/bash

# GPU Circle Simulation - Dependency Setup Script
# This script downloads and sets up all required dependencies

set -e

echo "Setting up GPU Circle Simulation dependencies..."

# Create external directory
mkdir -p external
cd external

# Setup GLFW
if [ ! -d "glfw" ]; then
    echo "Cloning GLFW..."
    git clone https://github.com/glfw/glfw.git
    cd glfw
    git checkout 3.3.8
    cd ..
else
    echo "GLFW already exists, skipping..."
fi

# Setup ImGui
if [ ! -d "imgui" ]; then
    echo "Cloning ImGui..."
    git clone https://github.com/ocornut/imgui.git
    cd imgui
    git checkout v1.89.9
    cd ..
else
    echo "ImGui already exists, skipping..."
fi

# Setup GLAD
if [ ! -d "glad" ]; then
    echo "Setting up GLAD..."
    echo "Please download GLAD manually from: https://glad.dav1d.de/"
    echo "Settings:"
    echo "  - Language: C/C++"
    echo "  - Specification: OpenGL"
    echo "  - gl: Version 3.3+"
    echo "  - Profile: Core"
    echo "  - Generate a loader: YES"
    echo ""
    echo "Extract the downloaded glad.zip to external/glad/"
    echo "The structure should be:"
    echo "  external/glad/include/glad/glad.h"
    echo "  external/glad/include/KHR/khrplatform.h"
    echo "  external/glad/src/glad.c"
    echo ""
    read -p "Press enter once you've downloaded and extracted GLAD..."
    
    if [ ! -f "glad/src/glad.c" ]; then
        echo "ERROR: GLAD not found at external/glad/"
        echo "Please download and extract it correctly before continuing."
        exit 1
    fi
else
    echo "GLAD already exists, skipping..."
fi

cd ..

echo ""
echo "Dependency setup complete!"
echo ""
echo "Next steps:"
echo "  1. Make sure CUDA Toolkit is installed"
echo "  2. Run: mkdir build && cd build"
echo "  3. Run: cmake .."
echo "  4. Run: make -j\$(nproc)"
echo "  5. Run: ./circle_sim"
echo ""