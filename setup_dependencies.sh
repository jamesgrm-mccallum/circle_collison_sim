#!/bin/bash

# GPU Circle Simulation - Dependency Setup Script
# Downloads GLFW and ImGui, and generates the GLAD OpenGL 3.3 core loader.

set -e

echo "Setting up GPU Circle Simulation dependencies..."

mkdir -p external
cd external

if [ ! -d "glfw" ]; then
    echo "Cloning GLFW 3.3.8..."
    git clone --depth 1 --branch 3.3.8 https://github.com/glfw/glfw.git glfw
else
    echo "GLFW already exists, skipping..."
fi

if [ ! -d "imgui" ]; then
    echo "Cloning ImGui v1.89.9..."
    git clone --depth 1 --branch v1.89.9 https://github.com/ocornut/imgui.git imgui
else
    echo "ImGui already exists, skipping..."
fi

# GLAD 0.1.x is a pure-Python generator, so the loader is generated here
# instead of being downloaded by hand from glad.dav1d.de.
if [ ! -f "glad/src/glad.c" ]; then
    echo "Generating GLAD (OpenGL 3.3 core)..."
    PYTHON=$(command -v python3 || command -v python)
    if [ -z "$PYTHON" ]; then
        echo "ERROR: python3 is required to generate GLAD."
        exit 1
    fi
    GENERATOR_DIR=$(mktemp -d)
    git clone --depth 1 --branch v0.1.36 https://github.com/Dav1dde/glad.git "$GENERATOR_DIR/glad"
    (cd "$GENERATOR_DIR/glad" && "$PYTHON" -m glad --profile core --api gl=3.3 \
        --generator c --spec gl --out-path "$OLDPWD/glad")
    rm -rf "$GENERATOR_DIR"
else
    echo "GLAD already exists, skipping..."
fi

cd ..

echo ""
echo "Dependency setup complete!"
echo ""
echo "Next steps:"
echo "  1. Install the CUDA Toolkit (12.0+) for the GPU modes (optional; CPU mode builds without it)"
echo "  2. cmake -S . -B build -DCMAKE_BUILD_TYPE=Release"
echo "  3. cmake --build build -j"
echo "  4. ./build/circle_sim          (or ./build/circle_bench for headless timings)"
