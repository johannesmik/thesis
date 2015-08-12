// Author: Johannes Mikulasch
// July 2015

#ifndef LIGHTS_CU
#define LIGHTS_CU

#include "utils.cu"

class Light {
public:
  virtual inline __device__ float3 direction(float3 point) = 0;
  virtual __device__ float attenuation(float distance) = 0;
  virtual __device__ float intensity() = 0;
};

class PointLight : public Light {
public:
  inline __device__ PointLight(float3 position, float falloff, float intensity = 1.0)
    : m_position(position), m_falloff(falloff), m_intensity(intensity) { }

  inline __device__ float3 direction(float3 point){
    if (m_position - point == make_float3(0, 0, 0))
      return make_float3(0, 0, 0);
    else
      return normalize(m_position - point);
  }

  __device__ float attenuation(float distance) {
    return 1.0f / (1 + m_falloff * distance * distance);
  }

  __device__ float attenuation(float3 point) {
    const float distance = dist(point, m_position);
    return attenuation(distance);
  }

  __device__ float falloff() {
    return m_falloff;
  }

  __device__ void set_falloff(float falloff) {
    m_falloff = falloff;
  }

  __device__ float intensity(){
    return m_intensity;
  }

  __device__ void set_intensity(float new_intensity){
    m_intensity = new_intensity;
  }

  __device__ float3 position(){
    return m_position;
  }

  __device__ void set_position(float3 position) {
    m_position = position;
  }

private:
  float m_falloff;
  float m_intensity;
  float3 m_position;
};

class DirectionalLight : public Light {
public:
  inline __device__ DirectionalLight(float3 direction)
    : m_light_direction(direction) { }


  inline __device__ float intensity(){
    return 1.0;
  }

  __device__ float attenuation(float distance) {
    return 1.0;
  }

  inline __device__ float3 direction(float3 point){
      return normalize(m_light_direction);
  }

  __device__ void set_direction(float3 direction) {
    m_light_direction = normalize(direction);
  }

private:
  float3 m_light_direction;
};

#endif
