#version 120

attribute vec3 position;
attribute vec3 color;
attribute vec3 normal;

uniform mat4 transform;

varying vec4 normal0;

varying vec4 basecolor;

void main() {
    basecolor = clamp(vec4(color, 1), 0, 1);
    normal0 = vec4(normal, 0);
    gl_Position = transform * vec4(position, 1);
}