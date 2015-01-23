#version 120

uniform vec3 light_position;
uniform vec4 light_direction;
uniform float light_intensity;
uniform bool light_isDirectional;
uniform bool light_isPoint;


uniform sampler2D colormap;
uniform mat4 VPMatrix;

varying vec4 normal0;
varying vec2 texcoords0;
varying vec4 basecolor;

float rand(vec2 co){
  return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

void main(){
    vec4 viewing_direction = normalize(VPMatrix * vec4(0, 0, -1, 1));
    vec4 half_angle = normalize( viewing_direction - light_direction);

    vec4 ambient = basecolor;
    ambient = ambient +  1.00 * texture2D(colormap, texcoords0);

    vec4 diffuse = (dot(normalize(-light_direction), normalize(normal0))) *  vec4(1,1,1,1);

    float spec = max(0, dot(half_angle, normalize(normal0)));
    spec = pow(spec, 100);
    vec4 specular = spec * vec4(1,1,1,1);

    vec4 color = 0.5 * ambient + 0.5 * diffuse + 2 * specular;
    gl_FragColor = clamp(color, 0.0, 1.0);

    // Apply some cheap noise
    gl_FragColor = gl_FragColor + .1 *(vec4(rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), 1) - 0.5);
}
