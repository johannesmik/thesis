#version 130
#extension GL_ARB_gpu_shader5 : enable

uniform bool use_colormap;
uniform bool use_normalmap;
uniform bool use_depthmap;

uniform sampler2D colormap;
uniform sampler2D normalmap;
uniform sampler2D depthmap;

uniform vec3 basecolor;
uniform float specularity;
uniform vec3 specular_color;

uniform mat4 MMatrix;
uniform mat4 VMatrix;
uniform mat4 PMatrix;

in vec3 normal0;
in vec2 texcoords0;
in vec3 position_w;
in vec3 position_c;

out vec4 out_color;
out vec4 out_normal;

#define MAX_AMBIENT_LIGHTS 2
#define MAX_DIRECTION_LIGHTS 2
#define MAX_POINT_LIGHTS 2
#define MAX_SPOT_LIGHTS 2

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

struct PointLight
{
    vec4 color;
    vec3 position;
    float falloff;
};
uniform PointLight pointlights[MAX_POINT_LIGHTS];

struct SpotLight
{
    vec4 color;
    vec3 position;
    vec3 direction;
    float falloff;
    float cone_angle; // in Radians
};
uniform SpotLight spotlights[MAX_SPOT_LIGHTS];

float rand(in vec2 co){
    /* This can be used for simple, and fast noise */
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float attenuation(in float falloff, in float distance) {
    return 1. / ( 1 + falloff * distance * distance);
}

/**************** Ambient Intensities ********************/

vec3 ambient_intensity(in AmbientLight light){
    return clamp(light.color.rgb, 0, 1);
}

/**************** Diffuse Intensities ********************/

vec3 diffuse_intensity(in DirectionLight light, in vec3 normal0){
    // Normalize the light direction if not a null-vector
    vec3 direction;
    if (length(light.direction) != 0)
        direction = normalize(-light.direction);
    else
        direction = -light.direction;

    return dot(direction, normal0) * light.color.rgb;
}

vec3 diffuse_intensity(in PointLight light, in vec3 normal0, in vec3 position){
    float distance = length(position - light.position);
    vec3 direction = - normalize(position - light.position);

    vec3 intensity = dot(direction, normal0) * light.color.rgb;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

vec3 diffuse_intensity(in SpotLight light, in vec3 normal0, in vec3 position){
    float distance = length(position - light.position);
    vec3 direction = - normalize(position - light.position);

    // Check if angle (between ray-direction and cone-direction) is within the limits
    float angle = acos(dot(direction, normalize(light.direction)));
    if (angle > light.cone_angle)
        return vec3(0, 0, 0);

    vec3 intensity = dot(direction, normal0) * light.color.rgb;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

/**************** Specular Intensities ********************/

vec3 specular_intensity(in DirectionLight light, in vec3 normal0, in vec3 position, in float specularity){
    // Todo
    return vec3(0, 0, 0);
}

vec3 specular_intensity(in PointLight light, in vec3 normal0, in vec3 position_c, in vec3 position_w, in float specularity){

    float distance = length(position_w - light.position);

    // Todo Simplify
    vec3 V = normalize((inverse(VMatrix) * vec4(0, 0, 0, 1)).xyz - (inverse(VMatrix) * vec4(position_c, 1)).xyz);
    //V = vec3(0, 0, 1);
    vec3 L = normalize(light.position - position_w);
    vec3 H = normalize(V+L);

    float theta = dot(normal0, H);

    // Todo use specular color
    vec3 intensity = pow(theta, specularity) * light.color.rgb;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

vec3 diffuse_intensity(in SpotLight light, in vec3 normal0, in vec3 position, in float specularity){
    // Todo
    return vec3(0, 0, 0);
}

void main(){

    vec3 normal;
    vec3 color;

    if (use_normalmap) {
     // Also map (0, 1) range to (-1, 1)
     //normal = normalize((texture2D(normalmap, texcoords0).rgb * 2 - 1) * texture2D(normalmap, texcoords0).a);
    }
    else {
     normal = normalize(normal0);
    }

    if (use_colormap) {
        color = texture2D(normalmap, texcoords0).rgb;
    }
    else {
        color = basecolor;
    }

    /* Depthmap */
    if (use_depthmap) {
        float depthmap_factor = 5; // TODO define this in a uniform
        float focal_length = 1; // TODO define this in a uniform

        //position_c = position_c + texture2D(depthmap, texcoords0).r;
    }

    /* Ambient */

    vec3 ambient = vec3(0, 0, 0);
    for(int i = 0; i < MAX_AMBIENT_LIGHTS; i++){
        ambient +=  ambient_intensity(ambientlights[i]);
    }

    /* Diffuse and Specular */

    vec3 diffuse = vec3(0, 0, 0);
    vec3 specular = vec3(0, 0, 0);
    for(int i = 0; i < MAX_DIRECTION_LIGHTS; i++){
        diffuse += diffuse_intensity(directionlights[i], normal);
        //specular += specular_intensity(directionlights[i], normal);
    }

    /* Point Lights */
    for(int i = 0; i < MAX_POINT_LIGHTS; i++){
        diffuse += diffuse_intensity(pointlights[i], normal, position_w);
        specular += specular_intensity(pointlights[i], normal, position_c, position_w, specularity);
    }

    /* Spot Lights */
    for(int i = 0; i < MAX_SPOT_LIGHTS; i++){
        diffuse += diffuse_intensity(spotlights[i], normal, position_w);
    }

    out_color = clamp(vec4(color * (ambient + diffuse) + specular_color * (specular), 1), 0.0, 1.0);

    // Apply some cheap noise
    //out_color = out_color + .1 *(vec4(rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), 1) - 0.5);
}
