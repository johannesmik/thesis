#version 150

uniform vec3 light_position;
uniform float light_intensity;
uniform bool light_isDirectional;
uniform bool light_isPoint;

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

#define MAX_AMBIENT_LIGHTS 0
#define MAX_DIRECTION_LIGHTS 0
#define MAX_POINT_LIGHTS 0
#define MAX_SPOT_LIGHTS 0

struct AmbientLight
{
    vec4 color;
};
uniform AmbientLight ambientlights[max(1, MAX_AMBIENT_LIGHTS)];

struct DirectionLight
{
    vec4 color;
    vec3 direction;
};
uniform DirectionLight directionlights[max(1, MAX_DIRECTION_LIGHTS)];

struct PointLight
{
    vec4 color;
    vec3 position;
    float falloff;
};
uniform PointLight pointlights[max(1, MAX_POINT_LIGHTS)];

struct SpotLight
{
    vec4 color;
    vec3 position;
    vec3 direction;
    float falloff;
    float cone_angle; // in Radians
};
uniform SpotLight spotlights[max(1, MAX_SPOT_LIGHTS)];

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

    //const float diff = .01;
    const float f = 1;
    const float correction = 3; //TODO where does this come from?

    /* Depthmap */
    /* Overwrites normal */
    if (use_depthmap) {
        // Calculate the normals from the depthmap using cross products


        vec2  texturesize = textureSize(depthmap, 0);
        float diff_x = 1.0 / texturesize.x;
        float diff_y = 1.0 / texturesize.y;

        // depthmap_factor: how far is the depthmap away from the camera?
        // TODO not used
        float depthmap_factor = 5; // TODO define this in a uniform

        //position_c = position_c + textureLod(depthmap, texcoords0, 0).r;
        vec3 vectorAC, vectorDB;

        float pointA_z = textureLod(depthmap, texcoords0 + vec2(-diff_x, 0), 0).r;
        float pointC_z = textureLod(depthmap, texcoords0 + vec2(diff_x, 0), 0).r;
        vectorAC = vec3(correction*(2*f + pointA_z + pointC_z)*diff_x/f, 0, (pointC_z - pointA_z));

        float pointB_z = textureLod(depthmap, texcoords0 + vec2(0, diff_y), 0).r;
        float pointD_z = textureLod(depthmap, texcoords0 + vec2(0, -diff_y), 0).r;
        vectorDB = vec3(0, correction*(2*f + pointB_z + pointD_z)*diff_y/f, (pointB_z - pointD_z));

        normal = normalize(cross(vectorAC, vectorDB));

        // Map range (-1, 1) to range (0, 1) when visualized in color
        normal = (normal + 1) / 2.;
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

    out_color = clamp(vec4(ambient + diffuse, 1), 0.0, 1.0);
    out_normal = clamp(vec4(normal, 1), 0.0, 1.0);

    // Apply some cheap noise
    //gl_FragColor = gl_FragColor + .1 *(vec4(rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), rand(gl_FragCoord.xy), 1) - 0.5);
}
