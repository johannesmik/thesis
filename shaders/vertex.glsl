#version 120

attribute vec3 position;
attribute vec2 texcoords;
attribute vec3 normal;

uniform mat4 MMatrix;
uniform mat4 PMatrix;
uniform mat4 VMatrix;

varying vec3 normal0;
varying vec2 texcoords0;
varying vec3 position0;

void main() {
    // Passing variables to fragment shader
    normal0 = normalize(normal);
    texcoords0 = texcoords;

    // Position in Camera Space
    position0 = (VMatrix * MMatrix * vec4(position, 1)).xyz;

    // Calculate position in Clip space
    gl_Position = PMatrix * VMatrix * MMatrix * vec4(position, 1);

    // Define point size
    gl_PointSize = 5.0;
}