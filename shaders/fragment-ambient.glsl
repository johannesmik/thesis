#version 130

uniform sampler2D colormap;
uniform vec3 basecolor;

in vec3 normal0;
in vec2 texcoords0;
in vec3 position_c;

out vec4 out_color;

#define MAX_AMBIENT_LIGHTS 3

struct AmbientLight
{
    vec4 color;
};
uniform AmbientLight ambientlights[MAX_AMBIENT_LIGHTS];

vec3 ambientcolor(vec3 basecolor, AmbientLight light){
    return basecolor *  light.color.rgb;
}


void main(){

    vec3 ambient = vec3(0, 0, 0);
    for(int i = 0; i < MAX_AMBIENT_LIGHTS; i++){
        ambient +=  ambientcolor(basecolor + texture2D(colormap, texcoords0).rgb, ambientlights[i]);
    }
    gl_FragColor = clamp(vec4(ambient, 1), 0., 1.0);
}
