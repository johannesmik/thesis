#version 150

attribute vec3 position;
attribute vec2 texcoords;
attribute vec3 normal;

uniform mat4 MMatrix;
uniform mat4 VMatrix;
uniform mat4 PMatrix;

out vec3 normal0;
out vec2 texcoords0;
out vec3 position_w;
out vec3 position_c;

void main() {
    // Passing variables to fragment shader
    normal0 = normalize(normal);
    texcoords0 = texcoords;

    // Position in World Space
    position_w = (MMatrix * vec4(position, 1)).xyz;

    // Position in Camera Space
    position_c = (VMatrix * MMatrix * vec4(position, 1)).xyz;

    // Calculate position in Clip space
    gl_Position = PMatrix * VMatrix * MMatrix * vec4(position, 1);

    // Define point size
    gl_PointSize = 5.0;
}