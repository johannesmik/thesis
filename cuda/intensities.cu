// Author: Johannes Mikulasch
// July 2015

#ifndef INTENSITIES_CU
#define INTENSITIES_CU

#include "lights.cu"


__device__ float intensity(const float3 &normal, const float3 &w) {

  const float albedo = 0.8;
  const float falloff = 1.0;
  const float3 light = light_directional();

  return attenuation(falloff, len(w)) * albedo * dot(normal, light);
}

#endif
