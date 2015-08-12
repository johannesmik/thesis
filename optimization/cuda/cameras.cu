#ifndef CAMERAS_CU
#define CAMERAS_CU

class Camera {
public:
  virtual __device__ float3 position() = 0;
  virtual __device__ float fx() = 0;
  virtual __device__ float fy() = 0;
  virtual __device__ float ox() = 0;
  virtual __device__ float oy() = 0;
};

class KinectCamera : public Camera {
public:

  __device__ KinectCamera(float3 position = make_float3(0, 0, 0))
   : m_position(position) {
      m_fx = 368.096588;
      m_fy = 368.096588;
      m_ox = 261.696594;
      m_oy = 202.522202;
   }

  __device__ float3 position(){
    return m_position;
  }

  __device__ void set_position(float3 position){
    m_position = position;
  }

  __device__ float fx() { return m_fx; }
  __device__ float fy() { return m_fy; }
  __device__ float ox() { return m_oy; }
  __device__ float oy() { return m_ox; }

private:
  float3 m_position;
  float m_fx, m_fy, m_ox, m_oy;
};

#endif