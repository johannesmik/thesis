// Author: Johannes Mikulasch
// July 2015

#ifndef INTENSITIES_CU
#define INTENSITIES_CU

#include "lights.cu"
#include "cameras.cu"

class LightingModel {
public:
  virtual __device__ float intensity(const float3 &normal, const float3 &w) = 0;
  virtual __device__ void set_material_properties(const float4 &material_props) = 0;
};

class LambertianLightingModel : public LightingModel {
public:

  __device__ LambertianLightingModel() {};
  __device__ LambertianLightingModel(Light &lighttest, Camera &camera)
   : m_light(&lighttest), m_camera(&camera), m_albedo(0.0) {};

  __device__ float intensity(const float3 &normal, const float3 &point){
    if (point.z == 0)
       return 0;

    const float theta = dot(normal, m_light->direction(point));

    const float intensity = m_light->intensity() * m_light->attenuation(dist(point, m_camera->position()));

    return clamp(intensity * m_albedo * theta, 0, 1);
  }

  __device__ void set_material_properties(const float4 &material_props){
    m_albedo = material_props.x;
  }
private:
  Light *m_light;
  Camera *m_camera;
  float m_albedo;
};

class SpecularLightingModel : public LightingModel {
public:
  __device__ SpecularLightingModel() {};

  __device__ float intensity(const float3 &normal, const float3 &w){
    // TODO implement

    return 0;
  }
};

#endif
