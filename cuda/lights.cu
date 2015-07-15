// Author: Johannes Mikulasch
// July 2015

#ifndef LIGHTS_CU
#define LIGHTS_CU

#include "utils.cu"

class Light {
public:
  virtual inline __device__ float3 direction(float3 point) = 0;
};

class Pointlight : public Light {
public:
  inline __device__ Pointlight(float3 position, float falloff)
    : position(position), falloff(falloff) { }

  inline __device__ float3 direction(float3 point){
    if (position - point == make_float3(0, 0, 0))
      return make_float3(0, 0, 0);
    else
      return normalize(position - point);
  }

private:
  float3 position;
  float falloff;
};

/* DEFINITIONS */

inline __device__ float3 light_directional() {
  return make_float3(0, 0, 1);
}

inline __device__ float3 light_point(float3 point, float3 camera) {
  if (camera - point == make_float3(0, 0, 0))
    return make_float3(0, 0, 0);
  else
    return normalize(camera - point);
}

inline __device__ float3 light_point(float3 point) {
  // Assume Lightpoint is at (0, 0, 0)
  return light_point(point, make_float3(0, 0, 0));
}

inline __device__ float attenuation(float falloff, float distance) {
    return 1.0f / (1 + falloff * distance * distance);
}

#endif
