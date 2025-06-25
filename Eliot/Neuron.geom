#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

in float outputVal[];
in float isInput[];

out vec3 fragColor;

uniform float size;

void emitQuad(vec2 center, float halfSize, vec3 color) {
    fragColor = color;

    vec2 offsets[4] = vec2[](
        vec2(-halfSize, -halfSize),
        vec2( halfSize, -halfSize),
        vec2(-halfSize,  halfSize),
        vec2( halfSize,  halfSize)
    );

    for (int i = 0; i < 4; ++i) {
        gl_Position = vec4(center + offsets[i], 0.0, 1.0);
        EmitVertex();
    }
    EndPrimitive();
}

void main() {
    float spacing = size;
    float inner = spacing * 0.7;  // Fill inside
    float outer = spacing * 0.95; // Slightly smaller than full spacing to avoid overlap

    vec2 center = gl_in[0].gl_Position.xy;
    float output = outputVal[0];
    float input = isInput[0];

    // Always draw outline (dark blue for all, green if input)
    vec3 outlineColor = (input > 0.5) ? vec3(0.0, 1.0, 0.0) : vec3(0.0, 0.0, 0.4);
    emitQuad(center, outer * 0.5, outlineColor);

    // Draw neuron fill
    vec3 fillColor = vec3(0.0, 0.0, 1.0);
    if (output == -1.0) fillColor = vec3(0.0, 0.0, 0.0);
    else if (output == 1.0) fillColor = vec3(0.5, 0.5, 1.0);
    emitQuad(center, inner * 0.5, fillColor);
}
