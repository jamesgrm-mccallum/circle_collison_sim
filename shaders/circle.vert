#version 330 core

layout (location = 0) in vec2 aPos;        // Circle mesh position
layout (location = 1) in vec2 aOffset;     // Instance position
layout (location = 2) in vec3 aColor;      // Instance color
layout (location = 3) in float aRadius;    // Instance radius

out vec3 fragColor;

uniform mat4 projection;

void main() {
    vec2 pos = aPos * aRadius + aOffset;
    gl_Position = projection * vec4(pos, 0.0, 1.0);
    fragColor = aColor;
}