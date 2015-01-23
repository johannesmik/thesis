#version 120

attribute vec3 position;
attribute vec3 color;
attribute vec2 texcoords;
attribute vec3 normal;

uniform mat4 MMatrix;
uniform mat4 VPMatrix;

varying vec4 normal0;
varying vec2 texcoords0;
varying vec4 basecolor;

void main() {
    // Passing variables to fragment shader
    basecolor = clamp(vec4(color, 1), 0, 1);
    normal0 = vec4(normalize(normal), 0);
    texcoords0 = texcoords;

    // Calculate position in Clip space
    gl_Position = VPMatrix * MMatrix * vec4(position, 1);
    gl_PointSize = 5.0;
}