#version 120

uniform vec3 light_position;
uniform float light_intensity;
uniform bool light_isDirectional;
uniform bool light_isPoint;

uniform bool use_normalmap;

uniform sampler2D colormap;
uniform sampler2D normalmap;

uniform vec3 basecolor;

varying vec3 normal0;
varying vec2 texcoords0;
varying vec3 position0;

#define MAX_AMBIENT_LIGHTS 3
#define MAX_DIRECTION_LIGHTS 3

struct AmbientLight
{
    vec4 color;
};
uniform AmbientLight ambientlights[MAX_AMBIENT_LIGHTS];

struct DirectionLight
{
    vec4 color;
    vec3 direction;
};
uniform DirectionLight directionlights[MAX_DIRECTION_LIGHTS];

float rand(vec2 co){
    /* This can be used for simple, and fast noise */
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

vec3 ambient_intensity(vec3 basecolor, AmbientLight light){
    return basecolor *  light.color.rgb;
}

vec3 diffuse_intensity(DirectionLight light, vec3 normal0){
    /* The diffuse component (Lambertian) */

    // Normalize the light direction if not a null-vector
    vec3 direction;
    if (length(light.direction) != 0)
        direction = normalize(-light.direction);
    else
        direction = -light.direction;

    return dot(direction, normal0) * light.color.rgb;
}

void main(){

    vec3 normal;
    if (use_normalmap) {
     // Also map (0, 1) range to (-1, 1)
     normal = (texture2D(normalmap, texcoords0).rgb * 2 - 1) * texture2D(normalmap, texcoords0).a;
    }
    else {
     normal = normalize(normal0);
    }

    vec3 color = basecolor * texture2D(normalmap, texcoords0).rgb;

    /* Ambient */

    vec3 ambient = vec3(0, 0, 0);
    for(int i = 0; i < MAX_AMBIENT_LIGHTS; i++){
        ambient +=  ambient_intensity(color, ambientlights[i]);
    }

    /* Diffuse */

    vec3 diffuse = vec3(0, 0, 0);
    for(int i = 0; i < MAX_DIRECTION_LIGHTS; i++){
        // TODO account for light distance
        diffuse += diffuse_intensity(directionlights[i], normal);
    }

    // TODO think about another name for 'color'
    color = ambient + diffuse;

    gl_FragColor = clamp(vec4(color, 1), 0.0, 1.0);

    // Apply some cheap noise
    //gl_FragColor = gl_FragColor + .1 *(vec4(rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), 1) - 0.5);
}
