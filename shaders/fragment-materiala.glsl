#version 120

uniform vec3 light_position;
uniform float light_intensity;
uniform bool light_isDirectional;
uniform bool light_isPoint;

uniform bool use_depthmap;

uniform sampler2D colormap;
uniform sampler2D normalmap;
uniform sampler2D depthmap;

uniform vec3 basecolor;

varying vec3 normal0;
varying vec2 texcoords0;
varying vec3 position_w;
varying vec3 position_c;

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

    return dot(direction, normal0) * light.color.rgb;
}

vec3 diffuse_intensity(PointLight light, vec3 normal0, vec3 position){
    /* The diffuse component (Lambertian) for PointLight */
    // TODO: What happens if the PointLight is exactly on the point of the surface, ie distance=0

    float distance = length(position - light.position);
    vec3 direction = - normalize(position - light.position);

    vec3 intensity = dot(direction, normal0) * light.color.rgb;
    float attenuation = 1 / ( 1 + light.falloff * distance * distance);
    return clamp(attenuation * intensity, 0, 1);
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
    float attenuation = 1 / ( 1 + light.falloff * distance * distance);
    return clamp(attenuation * intensity, 0, 1);
}

void main(){

    vec3 normal;

    vec3 color = basecolor * texture2D(normalmap, texcoords0).rgb;

    /* Depthmap */
    if (use_depthmap) {
        // Calculate the normals from the depthmap using cross products

        // depthmap_factor: how far is the depthmap away from the camera?
        float depthmap_factor = 5; // TODO define this in a uniform
        float focal_length = 1; // TODO define this in a uniform

        //position_c = position_c + texture2D(depthmap, texcoords0).r;
        vec3 vectorA, vectorB;
        float diff = 0.003;
        vectorA = vec3(0, 2*diff, texture2D(depthmap, texcoords0 + vec2(0, diff)).r - texture2D(depthmap, texcoords0 + vec2(0, -diff)).r);
        vectorB = vec3(2*diff, 0, texture2D(depthmap, texcoords0 + vec2(diff, 0)).r - texture2D(depthmap, texcoords0 + vec2(-diff, 0)).r);
        normal = - normalize(cross(vectorA, vectorB));
        normal = (normal + 1) / 2;
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

    // TODO think about another name for 'color'
    color = ambient + diffuse;

    gl_FragColor = clamp(vec4(normal, 1), 0.0, 1.0);

    // Apply some cheap noise
    //gl_FragColor = gl_FragColor + .1 *(vec4(rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), 1) - 0.5);
}
