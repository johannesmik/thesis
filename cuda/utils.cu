// Author: Johannes Mikulasch
// July 2015

#ifndef UTILS_CU
#define UTILS_CU

/* ADDITION */

inline __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ float2 operator+(float2 a, float b) {
  return make_float2(a.x + b, a.y + b);
}

inline __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

/* SUBTRACTION */

inline __device__ float2 operator-(float2 a, float b) {
  return make_float2(a.x - b, a.y - b);
}

inline __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float3 operator-(float3 a, float b) {
  return make_float3(a.x - b, a.y - b, a.z - b);
}

inline __device__ float3 operator-(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/* UNARY MINUS */

inline __device__ float2 operator-(float2 a) {
    return make_float2(-a.x, -a.y);
}

inline __device__ float3 operator-(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

/* MULTIPLICATION */

inline __device__ float3 operator*(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

/* DIVISION */

inline __device__ float2 operator/(float2 a, float b) {
  return make_float2(a.x / b, a.y / b);
}

inline __device__ float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

/* DOT */

inline __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}

inline __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

/* CROSS */

inline __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/* NORMALIZE */

inline __device__ float3 normalize(float3 a) {
  float invLen = 1.0f / sqrtf(dot(a, a));
  return a * invLen;
}

/* LEN */

inline __device__ float len(float3 a) {
  return sqrtf(dot(a, a));
}

/* DIST */

inline __device__ float dist(float3 a, float3 b) {
  return len(b - a);
}

/* NORMAL COLOR */

__device__ float3 normal_color(const float3 normal)
{
  /* Maps normal range (-1, 1) to (0, 1), which is important when normal is visualized in color */
  return (normal + 1.0) / 2.0;
}

/* PIXEL TO CAMERA */

__device__ float3 pixel_to_camera(int xs, int ys, float z)
{
  /*
   xs, ys: pixel coordinates (non-negative)
   z: depth (positive)
  */
  // Camera intrinsics for Kinect2
  const float fx = 368.096588;
  const float fy = 368.096588;
  const float ox = 261.696594;
  const float oy = 202.522202;

  // From Pixel to Camera Coordinates
  const float x = z * (xs - ox) / fx;
  const float y = - (z * (ys - oy) / fy);

  return make_float3(x, y, -z);
}

#endif