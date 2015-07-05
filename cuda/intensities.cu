// Author: Johannes Mikulasch
// July 2015

#ifndef INTENSITIES_CU
#define INTENSITIES_CU

#include "lights.cu"


__device__ float intensity(const float3 &normal, const float3 &w) {

  if (w.z == 0)
     return 0;

  const float ambient = 0.2;
  const float albedo = 0.8;
  const float falloff = 0.0;
  const float3 camera = make_float3(0, 0, 0);
  const float3 light = light_point(w);

  return clamp(ambient + attenuation(falloff, dist(w, camera)) * albedo * dot(normal, light), 0, 1);
}

#endif
