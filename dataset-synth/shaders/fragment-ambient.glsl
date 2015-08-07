#version 130

uniform sampler2D colormap;
uniform vec4 basecolor;
uniform bool ir_active;

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

    if (ir_active) {
        gl_FragColor = clamp(vec4(ambient.a, ambient.a, ambient.a, 1), 0, 1);
    } else {
        gl_FragColor = clamp(vec4(ambient.rgb, 1), 0., 1.0);
    }
}
