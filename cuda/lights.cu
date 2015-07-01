// Author: Johannes Mikulasch
// July 2015

#ifndef LIGHTS_CU
#define LIGHTS_CU

#include "utils.cu"

inline __device__ float3 light_directional() {
  return make_float3(0, 0, 1);
}

inline __device__ float3 light_point(float3 world) {
  return normalize(world);
}

inline __device__ float attenuation(float falloff, float distance) {
    return 1.0f / (1 + falloff * distance * distance);
}

#endif
