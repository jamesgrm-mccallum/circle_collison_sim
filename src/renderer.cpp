#include "renderer.h"

#include <glad/glad.h>

#include <cstddef>
#include <iostream>

namespace {

// Shaders are embedded so the app runs from any working directory.
const char *kVertexShader = R"(#version 330 core
layout (location = 0) in vec2 aCorner;   // unit quad corner in [-1, 1]
layout (location = 1) in vec3 iCircle;   // instance: x, y, radius
layout (location = 2) in vec4 iColor;    // instance: rgba8, normalised

uniform mat4 uProjection;

out vec2 vLocal;      // offset from the circle centre, in world units
out float vRadius;
out vec4 vColor;

void main() {
    // One unit of padding leaves room for the anti-aliased edge.
    vLocal = aCorner * (iCircle.z + 1.0);
    vRadius = iCircle.z;
    vColor = iColor;
    gl_Position = uProjection * vec4(iCircle.xy + vLocal, 0.0, 1.0);
}
)";

const char *kFragmentShader = R"(#version 330 core
in vec2 vLocal;
in float vRadius;
in vec4 vColor;

out vec4 FragColor;

void main() {
    float dist = length(vLocal);
    float coverage = clamp(vRadius - dist + 0.5, 0.0, 1.0);
    if (coverage <= 0.0) discard;
    float shade = 0.75 + 0.25 * (1.0 - dist / max(vRadius, 1.0));
    FragColor = vec4(vColor.rgb * shade, vColor.a * coverage);
}
)";

GLuint compile(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::cerr << "Shader compilation failed:\n" << log << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint link_program(const char *vs, const char *fs)
{
    GLuint vert = compile(GL_VERTEX_SHADER, vs);
    GLuint frag = compile(GL_FRAGMENT_SHADER, fs);
    if (!vert || !frag) {
        glDeleteShader(vert);
        glDeleteShader(frag);
        return 0;
    }
    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glDeleteShader(vert);
    glDeleteShader(frag);

    GLint ok = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::cerr << "Shader program linking failed:\n" << log << std::endl;
        glDeleteProgram(program);
        return 0;
    }
    return program;
}

} // namespace

Renderer::Renderer(float worldWidth, float worldHeight)
    : worldWidth(worldWidth), worldHeight(worldHeight) {}

Renderer::~Renderer()
{
    cleanup();
}

void Renderer::setupVAO(unsigned int vao, unsigned int instanceVBO)
{
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    const GLsizei stride = sizeof(CircleInstance);
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void *>(offsetof(CircleInstance, x)));
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);
    glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride,
                          reinterpret_cast<void *>(offsetof(CircleInstance, color)));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glBindVertexArray(0);
}

bool Renderer::init(int instanceCapacity)
{
    static_assert(sizeof(CircleInstance) == 16, "CircleInstance must match the 16-byte VBO layout");
    static_assert(offsetof(CircleInstance, radius) == offsetof(CircleInstance, x) + 2 * sizeof(float),
                  "x, y, radius must be contiguous for the vec3 attribute");

    capacity = instanceCapacity;
    program = link_program(kVertexShader, kFragmentShader);
    if (!program) return false;

    // Y grows downwards so gravity (positive y) pulls circles to the bottom of the window.
    const float l = 0.0f, r = worldWidth, t = 0.0f, b = worldHeight;
    const float projection[16] = {
        2.0f / (r - l), 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f / (t - b), 0.0f, 0.0f,
        0.0f, 0.0f, -1.0f, 0.0f,
        -(r + l) / (r - l), -(t + b) / (t - b), 0.0f, 1.0f,
    };
    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "uProjection"), 1, GL_FALSE, projection);
    glUseProgram(0);

    const float quad[8] = { -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f };
    glGenBuffers(1, &quadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);

    const GLsizeiptr bytes = static_cast<GLsizeiptr>(sizeof(CircleInstance)) * capacity;
    glGenBuffers(1, &streamVBO);
    glBindBuffer(GL_ARRAY_BUFFER, streamVBO);
    glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_STREAM_DRAW);

    glGenBuffers(1, &interopVBO);
    glBindBuffer(GL_ARRAY_BUFFER, interopVBO);
    glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenVertexArrays(1, &streamVAO);
    glGenVertexArrays(1, &interopVAO);
    setupVAO(streamVAO, streamVBO);
    setupVAO(interopVAO, interopVBO);
    return true;
}

void Renderer::upload(const CircleInstance *instances, int count)
{
    if (!instances || count <= 0) return;
    if (count > capacity) count = capacity;
    glBindBuffer(GL_ARRAY_BUFFER, streamVBO);
    // Orphan the old storage so the driver doesn't stall waiting for the
    // previous frame's draw to finish reading it.
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeof(CircleInstance)) * capacity, nullptr, GL_STREAM_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, static_cast<GLsizeiptr>(sizeof(CircleInstance)) * count, instances);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    drawFromInterop = false;
}

void Renderer::draw(int count)
{
    if (!program || count <= 0) return;
    if (count > capacity) count = capacity;
    glUseProgram(program);
    glBindVertexArray(drawFromInterop ? interopVAO : streamVAO);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, count);
    glBindVertexArray(0);
}

void Renderer::cleanup()
{
    if (streamVAO) glDeleteVertexArrays(1, &streamVAO);
    if (interopVAO) glDeleteVertexArrays(1, &interopVAO);
    if (quadVBO) glDeleteBuffers(1, &quadVBO);
    if (streamVBO) glDeleteBuffers(1, &streamVBO);
    if (interopVBO) glDeleteBuffers(1, &interopVBO);
    if (program) glDeleteProgram(program);
    streamVAO = interopVAO = quadVBO = streamVBO = interopVBO = program = 0;
}
