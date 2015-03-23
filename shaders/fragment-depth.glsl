#version 120

uniform sampler2D colormap;
uniform vec3 basecolor;

varying vec3 normal0;
varying vec2 texcoords0;
varying vec3 position_w;
varying vec3 position_c;

void main(){

    // Simple tests
    // They stop the GLSL compiler from too much optimization
    if (normal0.x == -1 || texcoords0.x == -1)
    {
        gl_FragColor = vec4(normal0, 1);
    }

    float color = gl_FragCoord.z;
    gl_FragColor = vec4(color, color, color, 1);
}