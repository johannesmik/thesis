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

  // TODO why -z?
  return vec3(x, y, z);
}

void main(){

    vec3 normal;

    /* Use Depthmap */
    /* Calculate the normals from the depthmap using cross products */
    /* Overwrites normal */
    vec3 pointD, pointF, pointH, pointB;

    if (use_depthmap) {

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
        normal = (normal + 1) / 2.; // Map range (-1, 1) to range (0, 1) when visualized in color
    }
    else {
        // Normalize and map resulting range from (-1,1) to (0,1)
        normal = (normalize(normal0) + 1 ) / 2;
    }

    out_color = vec4(normal, 1);
    //out_color = vec4(textureLod(depthmap, texcoords0, 0).rgb, 1);
}