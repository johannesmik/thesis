#version 120

varying vec3 normal0;
varying vec2 texcoords0;
varying vec3 position1;

void main(){

    // Simple tests
    // They stop the GLSL compiler from too much optimization
    if (normal0.x == -1 || texcoords0.x == -1)
    {
        gl_FragColor = vec4(normal0, 1);
    }

    // Normalize and map resulting range from (-1,1) to (0,1)
    gl_FragColor = vec4((normalize(normal0) + 1 ) / 2, 1);
}