#version 150

in vec3 normal0;
in vec2 texcoords0;
in vec3 position1;

out vec4 out_color;

void main(){
    // Normalize and map resulting range from (-1,1) to (0,1)
    out_color = vec4((normalize(normal0) + 1 ) / 2, 1);
}