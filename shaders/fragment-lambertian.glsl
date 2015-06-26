#version 130

uniform bool use_colormap;
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

float rand(in vec2 co){
    /* This can be used for simple, and fast noise */
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float attenuation(in float falloff, in float distance) {
    return 1. / ( 1 + falloff * distance * distance);
}

vec3 ambient_intensity(in vec3 basecolor, in AmbientLight light){
    return basecolor *  light.color.rgb;
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


vec3 from_texture_to_camera(in vec2 texture_coords, in sampler2D depthmap){

  // Hard coded camera params
  const float fx = 368.096588;
  const float fy = 368.096588;
  const float ox = 261.696594;
  const float oy = 202.522202;
  const float correction = 10;
  const int W = 512;
  const int H = 424;

  float z = - textureLod(depthmap, texture_coords, 0).r * correction;

  // From Texture to Pixel Coordinates
  float xs = texture_coords.x * W;
  float ys = - texture_coords.y * H + H;

  // From Pixel to Camera Coordinates
  float x = -z * (xs - ox) / fx;
  float y = - (-z * (ys - oy) / fy);

  return vec3(x, y, z);
}

void main(){

    vec3 color;
    vec3 normal;

    if (use_normalmap) {
     // Also map (0, 1) range to (-1, 1)
     normal = normalize((texture2D(normalmap, texcoords0).rgb * 2 - 1) * texture2D(normalmap, texcoords0).a);
    }
    else if (use_depthmap) {
        vec3 pointD, pointF, pointH, pointB;
        vec2  texturesize = textureSize(depthmap, 0);
        float diff_x = 1.0 / texturesize.x;
        float diff_y = 1.0 / texturesize.y;

        // Get the points in camera coordinates
        pointD = from_texture_to_camera(texcoords0 + vec2(-diff_x, 0), depthmap);
        pointF = from_texture_to_camera(texcoords0 + vec2(diff_x, 0), depthmap);
        pointH = from_texture_to_camera(texcoords0 + vec2(0, -diff_y), depthmap) ;
        pointB = from_texture_to_camera(texcoords0 + vec2(0, diff_y), depthmap);

        vec3 vectorDF = pointF - pointD;
        vec3 vectorHB = pointB - pointH;

        normal = normalize(cross(vectorDF, vectorHB));
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
