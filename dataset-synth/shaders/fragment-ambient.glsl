#version 130

uniform sampler2D colormap;
uniform vec4 basecolor;

in vec3 normal0;
in vec2 texcoords0;
in vec3 position_c;

out vec4 out_color;

#define MAX_AMBIENT_LIGHTS 5

struct AmbientLight
{
    vec4 color;
};
uniform AmbientLight ambientlights[MAX_AMBIENT_LIGHTS];

vec4 ambientcolor(vec4 basecolor, AmbientLight light){
    return basecolor *  light.color;
}


void main(){

    vec4 ambient = vec4(0, 0, 0, 0);
    for(int i = 0; i < MAX_AMBIENT_LIGHTS; i++){
        ambient +=  ambientcolor(basecolor + texture2D(colormap, texcoords0), ambientlights[i]);
    }
    gl_FragColor = clamp(ambient, 0., 1.0);
}
