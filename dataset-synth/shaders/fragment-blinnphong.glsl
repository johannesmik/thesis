#version 130
#extension GL_ARB_gpu_shader5 : enable

uniform bool use_colormap;
uniform bool use_normalmap;
uniform bool use_depthmap;
uniform bool ir_active;

uniform sampler2D colormap;
uniform sampler2D normalmap;
uniform sampler2D depthmap;

uniform vec4 basecolor;
uniform float specularity;
uniform vec4 specular_color;

uniform mat4 MMatrix;
uniform mat4 VMatrix;
uniform mat4 PMatrix;

in vec3 normal0;
in vec2 texcoords0;
in vec3 position_w;
in vec3 position_c;

out vec4 out_color;
out vec4 out_normal;

#define MAX_AMBIENT_LIGHTS 5
#define MAX_DIRECTION_LIGHTS 5
#define MAX_POINT_LIGHTS 10
#define MAX_SPOT_LIGHTS 5

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

vec4 ambient_intensity(in AmbientLight light){
    return clamp(light.color, 0, 1);
}

/**************** Diffuse Intensities ********************/

vec4 diffuse_intensity(in DirectionLight light, in vec3 normal0){
    // Normalize the light direction if not a null-vector
    vec3 direction;
    if (length(light.direction) != 0)
        direction = normalize(-light.direction);
    else
        direction = -light.direction;

    return dot(direction, normal0) * light.color;
}

vec4 diffuse_intensity(in PointLight light, in vec3 normal0, in vec3 position){
    float distance = length(position - light.position);
    vec3 direction = - normalize(position - light.position);

    vec4 intensity = dot(direction, normal0) * light.color;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

vec4 diffuse_intensity(in SpotLight light, in vec3 normal0, in vec3 position){
    float distance = length(position - light.position);
    vec3 direction = - normalize(position - light.position);

    // Check if angle (between ray-direction and cone-direction) is within the limits
    float angle = acos(dot(direction, normalize(light.direction)));
    if (angle > light.cone_angle)
        return vec4(0, 0, 0, 0);

    vec4 intensity = dot(direction, normal0) * light.color;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

/**************** Specular Intensities ********************/

vec4 specular_intensity(in DirectionLight light, in vec3 normal0, in vec3 position_w, in float specularity){

    vec3 camera_pos_w = (inverse(VMatrix) * vec4(0, 0, 0, 1)).xyz;
    vec3 V = normalize(camera_pos_w - position_w);
    vec3 L = normalize(-light.direction);
    vec3 H = normalize(V+L);

    float theta = dot(normal0, H);

    if (light.color != vec4(0, 0, 0, 0)) {
        vec4 intensity = pow(theta, specularity) * light.color;
        return clamp(intensity, 0, 1);
    }

    return vec4(0, 0, 0, 0);
}

vec4 specular_intensity(in PointLight light, in vec3 normal0, in vec3 position_w, in float specularity){

    float distance = length(position_w - light.position);

    vec3 camera_pos_w = (inverse(VMatrix) * vec4(0, 0, 0, 1)).xyz;
    vec3 V = normalize(camera_pos_w - position_w);
    vec3 L = normalize(light.position - position_w);
    vec3 H = normalize(V+L);

    float theta = dot(normal0, H);

    if (light.color != vec4(0, 0, 0, 0)) {
        vec4 intensity = pow(theta, specularity) * light.color;
        return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
    }
    return vec4(0, 0, 0, 0);
}

vec4 specular_intensity(in SpotLight light, in vec3 normal0, in vec3 position_w, in float specularity){
    // Todo check if this is correct
    float distance = length(position_w - light.position);

    vec3 camera_pos_w = (inverse(VMatrix) * vec4(0, 0, 0, 1)).xyz;
    vec3 V = normalize(camera_pos_w - position_w);
    vec3 L = normalize(light.position - position_w);
    vec3 H = normalize(V+L);

    // Check if angle (between ray-direction and cone-direction) is within the limits
    float angle = acos(dot(L, normalize(light.direction)));
    if (angle > light.cone_angle)
        return vec4(0, 0, 0, 0);

    float theta = dot(normal0, H);

    if (light.color != vec4(0, 0, 0, 0)) {
        vec4 intensity = pow(theta, specularity) * light.color;
        return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
    }
    return vec4(0, 0, 0, 0);
}

void main(){

    vec3 normal;
    vec4 color;

    if (use_normalmap) {
     // Also map (0, 1) range to (-1, 1)
     //normal = normalize((texture2D(normalmap, texcoords0).rgb * 2 - 1) * texture2D(normalmap, texcoords0).a);
    }
    else {
     normal = normalize(normal0);
    }

    if (use_colormap) {
        color = texture2D(normalmap, texcoords0).rgba;
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

    vec4 ambient = vec4(0, 0, 0, 0);
    for(int i = 0; i < MAX_AMBIENT_LIGHTS; i++){
        ambient +=  ambient_intensity(ambientlights[i]);
    }

    /* Diffuse and Specular */

    vec4 diffuse = vec4(0, 0, 0, 0);
    vec4 specular = vec4(0, 0, 0, 0);
    for(int i = 0; i < MAX_DIRECTION_LIGHTS; i++){
        diffuse += diffuse_intensity(directionlights[i], normal);
        specular += specular_intensity(directionlights[i], normal, position_w, specularity);
    }

    /* Point Lights */
    for(int i = 0; i < MAX_POINT_LIGHTS; i++){
        diffuse += diffuse_intensity(pointlights[i], normal, position_w);
        specular += specular_intensity(pointlights[i], normal, position_w, specularity);
    }

    /* Spot Lights */
    for(int i = 0; i < MAX_SPOT_LIGHTS; i++){
        diffuse += diffuse_intensity(spotlights[i], normal, position_w);
    }


    if (ir_active) {
        float t = (color * (ambient + diffuse) + specular_color * (specular)).a;
        out_color = clamp(vec4(t, t, t, 1), 0, 1);
    } else {
        out_color = clamp(vec4((color * (ambient + diffuse) + specular_color * (specular)).rgb, 1), 0.0, 1.0);
    }
    // Apply some cheap noise
    //out_color = out_color + .1 *(vec4(rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), 1) - 0.5);
}
