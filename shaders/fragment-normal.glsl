#version 150
#extension GL_ARB_explicit_attrib_location : enable

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

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal;

void main(){

    vec3 normal;

    /* Use Depthmap */
    /* Calculate the normals from the depthmap using cross products */
    /* Overwrites normal */
    if (use_depthmap) {

        //const float diff = .01;
        const float f = 1;
        const float correction = 3; //TODO where does this come from?

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
    else {
        // Normalize and map resulting range from (-1,1) to (0,1)
        normal = (normalize(normal0) + 1 ) / 2;
    }

    //out_color = vec4(normal, 1);
    out_color = vec4(0,1,1,1);
    out_normal = vec4(1,0,0,1);
}