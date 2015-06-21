# Calculate the intensity with a point light

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


def show_image(image):

    image_copy = image.copy()

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Use grey colormap
    cm = plt.get_cmap("gray")

    # Show points that fall out of the region [0, 1] in pink
    cm.set_under('pink')
    cm.set_over('pink')

    # No ticks on image axis
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    minimum = min(image_copy.min(), 0.0)
    maximum = max(image_copy.max(), 1.0)

    im = plt.imshow(image_copy, interpolation="nearest", cmap=cm, vmin=minimum, vmax=maximum)

    # Set up the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.05)
    clb = plt.colorbar(im, cax)


mod = SourceModule("""
#include <cuda.h>
//#include <vector_types.h>
//#include <math.h>

texture<float, cudaTextureType2D, cudaReadModeElementType> depth_in;

/* ADDITION */

inline __device__ float3 operator+(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator+(float3 a, float b) {
  return make_float3(a.x + b, a.y + b, a.z + b);
}

/* SUBTRACTION */

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

inline __device__ float3 operator/(float3 a, float b) {
  return make_float3(a.x / b, a.y / b, a.z / b);
}

/* DOT, CROSS, LEN, NORMAL */

inline __device__ float dot(float3 a, float3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __device__ float3 cross(float3 a, float3 b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ float3 normalize(float3 a) {
  float invLen = 1.0f / sqrtf(dot(a, a));
  return a * invLen;
}

inline __device__ float len(float3 a) {
  return sqrtf(dot(a, a));
}

/* Light functions */

__device__ float3 light_directional() {
  return make_float3(0, 0, 1);
}

__device__ float3 light_point(float3 w) {
  return normalize(w);
}

__device__ float attenuation(float falloff, float distance) {
    return 1.0f / (1 + falloff * distance * distance);
}


extern "C"
__device__ float3 pixel_to_camera(int xs, int ys, float z)
{
  // Camera coordinates
  const float fx = 368.096588;
  const float fy = 368.096588;
  const float ox = 261.696594;
  const float oy = 202.522202;

  // From Pixel to Camera Coordinates
  const float x = -z * (xs - ox) / fx;
  const float y = - (-z * (ys - oy) / fy);

  return make_float3(x, y, z);
}

extern "C"
__global__ void intensity(float *normal_out, float *intensity_out)
{
  // Constants
  //const float diff_depth = 0.01;
  const float albedo = 0.8;
  const float falloff = 1.0;
  float3 light = light_directional();

  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  // Point e in World coordinates
  const float3 point_e = pixel_to_camera(x, y, tex2D(depth_in, x, y));

  // Find the neighbours in camera coords
  const float3 point_b = pixel_to_camera(x, y-1, tex2D(depth_in, x, y-1));
  const float3 point_d = pixel_to_camera(x-1, y, tex2D(depth_in, x-1, y));
  const float3 point_f = pixel_to_camera(x+1, y, tex2D(depth_in, x+1, y));
  const float3 point_h = pixel_to_camera(x, y+1, tex2D(depth_in, x, y+1));

  // Calculate Normal
  const float3 vectorDF = point_f - point_d;
  const float3 vectorHB = point_b - point_h;
  float3 normal = normalize(cross(vectorDF, vectorHB));
  //normal = (normal + 1.0) / 2.0;  // Map range (-1, 1) to (0, 1) when visualized in color

  normal_out[index * 3] = normal.x;
  normal_out[index * 3 + 1] = normal.y;
  normal_out[index * 3 + 2] = normal.z;

  light = light_point(point_e);

  intensity_out[index] = attenuation(falloff, len(point_e)) * albedo * dot(normal, light);
}
""", no_extern_c=True)
# Note: We need the n_extern_c=True so that we can have operator overloading

multiply_them = mod.get_function("intensity")
tex_in = mod.get_texref('depth_in')
tex_in.set_address_mode(1, drv.address_mode.WRAP)
tex_in.set_address_mode(2, drv.address_mode.WRAP)

image = Image.open("sphere_depth.tiff")
image = np.asarray(image, dtype=np.float32)
show_image(image)

image_arr = drv.matrix_to_array(image, 'C')
tex_in.set_array(image_arr)
normal = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

intensity = np.zeros_like(image)

multiply_them(
    drv.Out(normal), drv.Out(intensity),
    block=(16, 8, 1), grid=(32, 53))

show_image(intensity)

plt.show()



