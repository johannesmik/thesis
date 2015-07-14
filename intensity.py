# Calculate the intensity prime (using a point light)

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
import os

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

import utils

mod = SourceModule("""
#include <cuda.h>
//#include <vector_types.h>
#include <utils.cu>

texture<float, cudaTextureType2D, cudaReadModeElementType> depth_sensor;
texture<float, cudaTextureType2D, cudaReadModeElementType> depth_current;

/* Light functions */

inline __device__ float3 light_directional() {
  return make_float3(0, 0, 1);
}

inline __device__ float3 light_point(float3 w) {
  return normalize(w);
}

inline __device__ float attenuation(float falloff, float distance) {
    return 1.0f / (1 + falloff * distance * distance);
}

/* INTENSITY FUNCTIONS */

__device__ float intensity(const float3 &normal, const float3 &w) {

  const float albedo = 0.8;
  const float falloff = 1.0;
  const float3 light = light_directional();

  return attenuation(falloff, len(w)) * albedo * dot(normal, light);
}


extern "C"
__global__ void intensity_prime(float *normal_out, float *intensity_change_out)
{
  // Constants
  const float diff_depth = 0.00001;

  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  // Point e in World coordinates
  const float3 point_e = pixel_to_camera(x, y, tex2D(depth_sensor, x, y));

  // Adding diff_depth to the depth will change the intensity of four of it's neighbors.
  // 1. Find the intensity of those neighbors before adding the depth to the current point
  float intensity_before[4];
  for (int i = 0; i < 4; ++i) {
    int x_n, y_n;
    if (i == 0) { x_n = x;   y_n = y-1; }
    if (i == 1) { x_n = x-1; y_n = y ; }
    if (i == 2) { x_n = x+1; y_n = y; }
    if (i == 3) { x_n = x;   y_n = y+1; }

    // Find the neighbors of the neighbor in camera coords
    const float3 point_b = pixel_to_camera(x_n, y_n-1, tex2D(depth_sensor, x_n, y_n-1));
    const float3 point_d = pixel_to_camera(x_n-1, y_n, tex2D(depth_sensor, x_n-1, y_n));
    const float3 point_f = pixel_to_camera(x_n+1, y_n, tex2D(depth_sensor, x_n+1, y_n));
    const float3 point_h = pixel_to_camera(x_n, y_n+1, tex2D(depth_sensor, x_n, y_n+1));

    // Calculate Normal
    const float3 vectorDF = point_f - point_d;
    const float3 vectorHB = point_b - point_h;
    const float3 normal = normalize(cross(vectorDF, vectorHB));
    const float3 normal_color = (normal + 1.0) / 2.0;  // Map range (-1, 1) to (0, 1) when visualized in color

    intensity_before[i] = intensity(normal, point_e);
  }

  // 2. Find the intensity of those neighbors after adding depth to the current point
  float intensity_after[4];
  float3 normal_color;
  for (int i = 0; i < 4; ++i) {
    int x_n, y_n;
    if (i == 0) { x_n = x;   y_n = y-1; }
    if (i == 1) { x_n = x-1; y_n = y ; }
    if (i == 2) { x_n = x+1; y_n = y; }
    if (i == 3) { x_n = x;   y_n = y+1; }

    // Find the neighbors of the neighbor in camera coords
    float3 point_b = pixel_to_camera(x_n, y_n-1, tex2D(depth_sensor, x_n, y_n-1));
    float3 point_d = pixel_to_camera(x_n-1, y_n, tex2D(depth_sensor, x_n-1, y_n));
    float3 point_f = pixel_to_camera(x_n+1, y_n, tex2D(depth_sensor, x_n+1, y_n));
    float3 point_h = pixel_to_camera(x_n, y_n+1, tex2D(depth_sensor, x_n, y_n+1));

    if (i == 0) { point_h = pixel_to_camera(x, y, tex2D(depth_sensor, x, y) + diff_depth); }
    if (i == 1) { point_f = pixel_to_camera(x, y, tex2D(depth_sensor, x, y) + diff_depth); }
    if (i == 2) { point_d = pixel_to_camera(x, y, tex2D(depth_sensor, x, y) + diff_depth); }
    if (i == 3) { point_b = pixel_to_camera(x, y, tex2D(depth_sensor, x, y) + diff_depth); }

    // Calculate Normal
    const float3 vectorDF = point_f - point_d;
    const float3 vectorHB = point_b - point_h;
    const float3 normal = normalize(cross(vectorDF, vectorHB));
    normal_color = (normal + 1.0) / 2.0;  // Map range (-1, 1) to (0, 1) when visualized in color

    intensity_after[i] = intensity(normal, point_e);
  }

  intensity_change_out[index] = 0.0;
  for (int i = 0; i < 4; ++i) {
    intensity_change_out[index] = intensity_change_out[index] + (intensity_after[i] - intensity_before[i]);
  }
  intensity_change_out[index] = intensity_change_out[index] / diff_depth;
  normal_out[index * 3] = normal_color.x;
  normal_out[index * 3 + 1] = normal_color.y;
  normal_out[index * 3 + 2] = normal_color.z;
}
""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])
# Note: We need the n_extern_c=True so that we can have operator overloading

intensity_prime = mod.get_function("intensity_prime")

# Set up textures
tex_depth_noise = mod.get_texref('depth_sensor')
tex_depth_noise.set_address_mode(1, drv.address_mode.WRAP)
tex_depth_noise.set_address_mode(2, drv.address_mode.WRAP)
depth_image = Image.open("sphere_depth.tiff")
#depth_image = depth_image.filter(ImageFilter.GaussianBlur(2))
depth_image = np.asarray(depth_image, dtype=np.float32)
mu, sigma = 0, 0.0
depth_image = depth_image + sigma * np.random.randn(*depth_image.shape).astype(np.float32) + mu
depth_image_arr = drv.matrix_to_array(depth_image, 'C')
tex_depth_noise.set_array(depth_image_arr)

tex_depth_current = mod.get_texref('depth_current')
tex_depth_current.set_address_mode(1, drv.address_mode.WRAP)
tex_depth_current.set_address_mode(2, drv.address_mode.WRAP)
depth_current_arr = drv.matrix_to_array(depth_image, 'C')
tex_depth_current.set_array(depth_current_arr)


normal = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)

intensity_change = np.zeros_like(depth_image)

intensity_prime(
    drv.Out(normal), drv.Out(intensity_change),
    block=(16, 8, 1), grid=(32, 53))

utils.show_image(depth_image, title='depth image')
utils.show_image(intensity_change, title='intensity change')
utils.show_image(normal, title='normal')

plt.show()
