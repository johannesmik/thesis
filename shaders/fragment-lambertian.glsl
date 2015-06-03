#version 150

uniform bool use_normalmap;
uniform bool use_depthmap;

uniform sampler2D colormap;
uniform sampler2D normalmap;
uniform sampler2D depthmap;

uniform vec3 basecolor;

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

float rand(vec2 co){
    /* This can be used for simple, and fast noise */
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float attenuation(float falloff, float distance) {
    return 1. / ( 1 + falloff * distance * distance);
}

vec3 ambient_intensity(vec3 basecolor, AmbientLight light){
    return basecolor *  light.color.rgb;
}

vec3 diffuse_intensity(DirectionLight light, vec3 normal0){
    /* The diffuse component (Lambertian) for DirectionLight */

    // Normalize the light direction if not a null-vector
    vec3 direction;
    if (length(light.direction) != 0)
        direction = normalize(-light.direction);
    else
        direction = -light.direction;


    vec3 intensity = dot(direction, normal0) * light.color.rgb;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

vec3 diffuse_intensity(PointLight light, vec3 normal0, vec3 position){
    /* The diffuse component (Lambertian) for PointLight */
    // TODO: What happens if the PointLight is exactly on the point of the surface, ie distance=0

    float distance = length(position - light.position);
    vec3 direction = - normalize(position - light.position);

    vec3 intensity = dot(direction, normal0) * light.color.rgb;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

vec3 diffuse_intensity(SpotLight light, vec3 normal0, vec3 position){
    /* The diffuse component (Lambertian) for SpotLight */

    // Check if angle is in
    float distance = length(position - light.position);
    vec3 direction = - normalize(position - light.position);

    // Check if angle (between ray-direction and cone-direction) is within the limits
    float angle = acos(dot(direction, normalize(light.direction)));
    if (angle > light.cone_angle)
        return vec3(0, 0, 0);

    vec3 intensity = dot(direction, normal0) * light.color.rgb;
    return clamp(attenuation(light.falloff, distance) * intensity, 0, 1);
}

void main(){

    vec3 normal;
    if (use_normalmap) {
     // Also map (0, 1) range to (-1, 1)
     normal = normalize((texture2D(normalmap, texcoords0).rgb * 2 - 1) * texture2D(normalmap, texcoords0).a);
    }
    else {
     normal = normalize(normal0);
    }

    vec3 color = basecolor;

    /* Depthmap */
    if (use_depthmap) {
        float depthmap_factor = 5; // TODO define this in a uniform
        float focal_length = 1; // TODO define this in a uniform

        //position_c = position_c + texture2D(depthmap, texcoords0).r;
    }

    /* Ambient */

    vec3 ambient = vec3(0, 0, 0);
    for(int i = 0; i < MAX_AMBIENT_LIGHTS; i++){
        ambient +=  ambient_intensity(color, ambientlights[i]);
    }

    /* Diffuse */

    vec3 diffuse = vec3(0, 0, 0);
    for(int i = 0; i < MAX_DIRECTION_LIGHTS; i++){
        diffuse += diffuse_intensity(directionlights[i], normal);
    }

    /* Point Lights */
    for(int i = 0; i < MAX_POINT_LIGHTS; i++){
        diffuse += diffuse_intensity(pointlights[i], normal, position_w);
    }

    /* Spot Lights */
    for(int i = 0; i < MAX_SPOT_LIGHTS; i++){
        diffuse += diffuse_intensity(spotlights[i], normal, position_w);
    }

    out_color = clamp(vec4(color * (ambient + diffuse), 1), 0.0, 1.0);

    // Apply some cheap noise
    //out_color = out_color + .1 *(vec4(rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), 1) - 0.5);
}
