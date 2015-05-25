#version 150

uniform sampler2D colormap;
uniform vec3 basecolor;

in vec3 position_w;
in vec3 position_c;

out vec4 out_color;

void main(){
    float color = gl_FragCoord.z;
    out_color = vec4(color, color, color, 1);
}