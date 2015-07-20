from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import utils

mod = SourceModule("""
#include <cuda.h>
#include "utils.cu"
#include "intensities.cu"
#include "normal.cu"

texture<float, cudaTextureType2D, cudaReadModeElementType> depth;
texture<float, cudaTextureType2D, cudaReadModeElementType> intensity_current;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_intensity;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_intensity_sensor;

__device__ void set_depth_neighborhood(int2 pos, float neighborhood[5][5])
{
  /* returns the 5x5 depth neighborhood around pos. */
  /* Loads texture memory into global memory (slow) */

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      neighborhood[j][i] = tex2D(depth, pos.x -2 + i, pos.y -2 + j);
    }
  }
}

__device__ float intensity_local(int2 center_pos, int2 change_pos, float adjustment){
  // Calculate the intensity around center, but adjust the depth of the pixel at change_pos
  // Center_pos: screen coords
  // Change_pos: screen coords

  float depth_neighborhood[5][5];
  set_depth_neighborhood(center_pos, depth_neighborhood);
  const float z = depth_neighborhood[3][3];

  // Adjust
  depth_neighborhood[change_pos.y - center_pos.y + 2][change_pos.x - center_pos.x + 2] += adjustment;

  const float3 normal = normal_cross(depth_neighborhood, center_pos);
  const float intensity_return = intensity(normal, pixel_to_camera(center_pos.x, center_pos.y, z));

  return intensity_return;
}


inline __device__ float intensity_local(int2 pos) {
  // Calculate the intensity at the midpoint of 5x5 depth neighborhood around pos
  return intensity_local(pos, pos, 0.0);
}

inline __device__ float intensity_local(int2 pos, float adjustment) {
  // Calculate the intensity around center, but adjust the depth of the pixel at center
  // pos: screen coords
  return intensity_local(pos, pos, adjustment);
}

extern "C"
__global__ void energy_prime(float *energy_change_out){

  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  const float h = 0.001;

  // Calculate changes in the intensity term

  // Intensity array before and from sensor
  float intensity_before[5][5];
  float intensity_sensor[5][5];

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      intensity_before[j][i] = tex2D(intensity_current, x -2 + i, y -2 + j);
      intensity_sensor[j][i] = tex2D(ir_intensity_sensor, x -2 + i, y -2 + j);
    }
  }

  float intensity_after[5][5];
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      int2 center = make_int2(x + i - 2, y + j - 2);
      int2 change = make_int2(x, y);
      intensity_after[j][i] = intensity_local(center, change, h);
    }
  }

  float intensity_term = 0;
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      intensity_term += pow(intensity_after[j][i] - intensity_sensor[j][i], 2) - pow(intensity_before[j][i] - intensity_sensor[j][i], 2);
    }
  }
  intensity_term /= h;

  // Calculate changes in the normal term

  // Return
  //energy_change_out[index] = intensity_before[3][3];
  energy_change_out[index] = intensity_term;

}

extern "C"
__global__ void intensity_image(float *intensity_out)
{
    // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  intensity_out[index] = intensity_local(make_int2(x, y), 0.0);
}

extern "C"
__global__ void energy_normal(float3 *normal, float *energy_normal_out)
{
  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  const float factor = 0.25;

  float3 normal_center = normal[index];

  int adjacent_indices[4] = {-1};
  adjacent_indices[0] = (index - elementPitch >= 0) ? index - elementPitch  : -1;
  adjacent_indices[1] = (index - 1 >= 0 && x - 1 > 0) ? index - 1 : -1;
  adjacent_indices[2] = (index + 1 < gridDim.y * blockDim.y * elementPitch && x + 1 < elementPitch) ? index + 1 : -1;
  adjacent_indices[3] = (index + elementPitch < gridDim.y * blockDim.y * elementPitch) ? index + elementPitch : -1;

  energy_normal_out[index] = 0;
  for (int i = 0; i < 4; ++i) {
    if (adjacent_indices[i] != -1 && normal[adjacent_indices[i]] != make_float3(0, 0, 0))
     energy_normal_out[index] += abs(len(normal[adjacent_indices[i]]  - normal_center));
  }
  energy_normal_out[index] *= factor;
}

extern "C"
__global__ void energy(float *energy_intensity_out, float *intensity_out, float3 *normal_out)
{
  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  float depth_neighborhood[5][5];
  set_depth_neighborhood(make_int2(x, y), depth_neighborhood);

  //float intensity_test = intensity_local(make_int2(x, y), depth_neighborhood);
  //float intensity_test = intensity_local(make_int2(x, y), 0.5);
  float intensity_test = intensity_local(make_int2(x, y), make_int2(x + 1, y), 0.01);

  float3 normal = normal_pca(depth_neighborhood, make_int2(x, y));
  float3 normal_c = normal_colorize(normal);

  float intensity_given = tex2D(ir_intensity, x, y);
  float intensity_new = intensity(normal, pixel_to_camera(x, y, tex2D(depth, x, y)));

  intensity_out[index] = intensity_test;
  energy_intensity_out[index] = pow(intensity_given - intensity_new, 2);
  normal_out[index] = normal_c;

}
""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])

energy_function = mod.get_function("energy")
energy_normal_function = mod.get_function("energy_normal")
energy_prime_function = mod.get_function("energy_prime")
intensity_function = mod.get_function("intensity_image")

use_testimage = False
if use_testimage:
    depth_image = Image.open("testimage.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32)
    depth_image = depth_image * 10.0 / 255.
else:
    depth_image = Image.open("sphere_depth.tiff")
    depth_image = Image.open("head_depth.tiff")
    #depth_image = depth_image.filter(ImageFilter.GaussianBlur(2))
    depth_image = np.asarray(depth_image, dtype=np.float32)
    depth_image = depth_image * 10

ir_intensity_image = Image.open("head_ir.tiff")
ir_intensity_image = np.asarray(ir_intensity_image, dtype=np.float32)
#ir_intensity_image = ir_intensity_image[:,:,0] / 255.

mu, sigma = 0, 0.0
depth_image = depth_image + sigma * np.random.randn(*depth_image.shape).astype(np.float32) + mu
print("min/max of depth_image", depth_image.min(), depth_image.max())

# Set up textures
depth_tex = mod.get_texref('depth')
depth_tex.set_address_mode(0, drv.address_mode.CLAMP)
depth_tex.set_address_mode(1, drv.address_mode.CLAMP)
depth_image_arr = drv.matrix_to_array(depth_image, 'C')
depth_tex.set_array(depth_image_arr)

# Set up texture: Intensity
ir_intensity_tex = mod.get_texref('ir_intensity')
ir_intensity_tex.set_address_mode(0, drv.address_mode.CLAMP)
ir_intensity_tex.set_address_mode(1, drv.address_mode.CLAMP)
ir_intensity_image_arr = drv.matrix_to_array(ir_intensity_image, 'C')
ir_intensity_tex.set_array(ir_intensity_image_arr)

# Set up texture: Intensity Sensor
intensity_sensor = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
intensity_sensor_tex = mod.get_texref('ir_intensity_sensor')
intensity_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
intensity_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
intensity_sensor_arr = drv.matrix_to_array(ir_intensity_image, 'C')
intensity_sensor_tex.set_array(intensity_sensor_arr)

# Set up texture: Intensity current
intensity_current = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
intensity_current_tex = mod.get_texref('intensity_current')
intensity_current_tex.set_address_mode(0, drv.address_mode.CLAMP)
intensity_current_tex.set_address_mode(1, drv.address_mode.CLAMP)
intensity_current_arr = drv.matrix_to_array(intensity_current, 'C')
intensity_current_tex.set_array(intensity_current_arr)

energy_intensity = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
energy_prime = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
intensity = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
energy_normal = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
normal = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)

intensity_change = np.zeros_like(depth_image)

for i in range(1):

    intensity_function(
        drv.Out(intensity_current),
        block=(16, 8, 1), grid=(32, 53))

    intensity_current_arr = drv.matrix_to_array(intensity_current, 'C')
    intensity_current_tex.set_array(intensity_current_arr)

    energy_prime_function(
        drv.Out(energy_prime),
        block=(16, 8, 1), grid=(32, 53))

    # energy_function(
    #     drv.Out(energy_intensity), drv.Out(intensity), drv.Out(normal),
    #     block=(16, 8, 1), grid=(32, 53))
    #
    # energy_normal_function(
    #     drv.In(normal), drv.Out(energy_normal),
    #     block=(16, 8, 1), grid=(32, 53))

utils.show_image(energy_prime, title="Energy Prime")
#utils.show_image(ir_intensity_image, title="IR Intensity Image")
#utils.show_image(energy_intensity, title="Energy Intensity")
# utils.show_image(energy_normal, title="Energy Normal")
# utils.show_image(intensity, title="Intensity")
# utils.show_image(normal, title="Normal")
print("energy:", energy_intensity.sum())

plt.show()
