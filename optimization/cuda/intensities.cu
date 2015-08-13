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
  __device__ LambertianLightingModel(Light &light, Camera &camera)
   : m_light(&light), m_camera(&camera), m_albedo(0.0) {};

  __device__ float intensity(const float3 &normal, const float3 &point){
    if (point.z == 0)
       return 0;

    const float cos_phi = dot(normal, m_light->direction(point));

    const float intensity = m_light->intensity() * m_light->attenuation(dist(point, m_camera->position()));

    return clamp(intensity * m_albedo * cos_phi, 0, 1);
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
  __device__ SpecularLightingModel(Light &light, Camera &camera)
   : m_light(&light), m_camera(&camera), m_kd(0.0), m_ks(0.0), m_n(0.0) {};

  __device__ float intensity(const float3 &normal, const float3 &point){
    if (point.z == 0)
       return 0;


    const float intensity_in = m_light->intensity() * m_light->attenuation(dist(point, m_camera->position()));

    const float cos_phi = dot(normal, m_light->direction(point));

    const float intensity_diff = intensity_in * m_kd * cos_phi;

    float3 camera_vector = m_camera->position() - point;
    float3 light_vector =  - m_light->direction(point);
    float3 half_vector = (camera_vector + light_vector) / (len(camera_vector + light_vector));

    const float cos_theta = dot(normal, half_vector);
    const float intensity_spec = intensity_in * m_ks * pow(cos_theta, m_n);

    return clamp(intensity_in * intensity_diff + intensity_in * intensity_spec, 0, 1);
  }

  __device__ void set_material_properties(const float4 &material_props){
    m_kd = material_props.x;
    m_ks = material_props.y;
    m_n = material_props.z;
  }

private:
  Light *m_light;
  Camera *m_camera;
  float m_kd;
  float m_ks;
  float m_n;
};

#endif
