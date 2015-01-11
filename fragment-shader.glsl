#version 120

uniform vec3 light_position;
uniform vec3 light_direction;
uniform float light_intensity;
uniform bool light_isDirectional;
uniform bool light_isPoint;

uniform mat4 transform;

varying vec4 normal0;
varying vec4 basecolor;

void main(){
    vec4 viewing_direction = transform * vec4(0, 0, -1, 0);
    vec4 color = (dot(normalize(vec4(-light_direction, 1)), normalize(normal0))) *  basecolor;
    gl_FragColor = clamp(color, 0.0, 1.0);
}