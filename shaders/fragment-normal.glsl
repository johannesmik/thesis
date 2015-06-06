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

        vec2  texturesize = textureSize(depthmap, 0);
        float diff_x = 1.0 / texturesize.x;
        float diff_y = 1.0 / texturesize.y;
        const float f = 10;

        float correction = 1;
        float correction_z = 1;


        // depthmap_factor: how far is the depthmap away from the camera?
        // TODO not used
        //float depthmap_factor = 5; // TODO define this in a uniform

        //position_c = position_c + textureLod(depthmap, texcoords0, 0).r;
        vec3 vectorDF, vectorHB;
        float x, y, z;
        vec2 coords = texcoords0;
        coords = (coords - vec2(0.5, 0.5)) * 2;

        float pointD_z = - correction - textureLod(depthmap, texcoords0 + vec2(-diff_x, 0), 0).r;
        float pointF_z = - correction - textureLod(depthmap, texcoords0 + vec2(diff_x, 0), 0).r;
        x = (((coords.x + diff_x) * pointF_z) - ((coords.x - diff_x) * pointD_z) )/ f;
        z = pointF_z - pointD_z;
        vectorDF = vec3(x, 0, z);

        float pointB_z = - correction - textureLod(depthmap, texcoords0 + vec2(0, diff_y), 0).r;
        float pointH_z = - correction - textureLod(depthmap, texcoords0 + vec2(0, -diff_y), 0).r;
        y = (((coords.y + diff_y) * pointB_z) - ((coords.y - diff_y) * pointH_z) )/ f;
        z = pointB_z - pointD_z;
        vectorHB = vec3(0, y, z);

        normal = normalize(cross(vectorDF, vectorHB));

        // Map range (-1, 1) to range (0, 1) when visualized in color
        normal = (normal + 1) / 2.;
    }
    else {
        // Normalize and map resulting range from (-1,1) to (0,1)
        normal = (normalize(normal0) + 1 ) / 2;
    }
    out_color = vec4(normal, 1);
    //out_color = vec4(diff_x, diff_y, 0, 1);
}