#version 120

varying vec4 normal0;
varying vec2 texcoords0;
varying vec4 basecolor;

void main(){

    // Simple tests
    // They stop the GLSL compiler from too much optimization
    if (normal0.x == -1 || texcoords0.x == -1 || basecolor.x == -1)
    {
        gl_FragColor = normal0;
    }

    gl_FragColor = normal0;
}