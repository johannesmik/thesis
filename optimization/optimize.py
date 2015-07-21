from __future__ import print_function

# Todo Refactor this, especially the function names and arguments to make sense, otherwise it's a mess to debug
# FIXME

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import utils

mod = SourceModule("""
#include <cuda.h>
#include "utils.cu"
#include "intensities.cu"
#include "normal.cu"

texture<float, cudaTextureType2D, cudaReadModeElementType> depth_sensor;
texture<float, cudaTextureType2D, cudaReadModeElementType> depth_current;
texture<float, cudaTextureType2D, cudaReadModeElementType> intensity_current;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_intensity;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_sensor;

__device__ void set_depth_neighborhood(int2 pos, float neighborhood[5][5])
{
  /* returns the 5x5 depth neighborhood around pos. */
  /* Loads texture memory into global memory (slow) */

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      neighborhood[j][i] = tex2D(depth_current, pos.x -2 + i, pos.y -2 + j);
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
      intensity_sensor[j][i] = tex2D(ir_sensor, x -2 + i, y -2 + j);
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

  intensity_out[index] = intensity_local(make_int2(x, y));
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
  float intensity_new = intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor, x, y)));

  intensity_out[index] = intensity_test;
  energy_intensity_out[index] = pow(intensity_given - intensity_new, 2);
  normal_out[index] = normal_c;

}
""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])

class Optimizer(object):

    def __init__(self, depth_sensor_filename, ir_sensor_filename):

        self.eps = 0.0001
        self.max_iterations = 5

        self.iteration_counter = 0

        # Counter for debug reasons
        # Todo Delete at sometime :)
        self.debug_counter = 0

        # Setup CUDA
        self.energy_function = mod.get_function("energy")
        self.energy_normal_function = mod.get_function("energy_normal")
        self.energy_prime_function = mod.get_function("energy_prime")
        self.intensity_function = mod.get_function("intensity_image")

        # Setup and Upload texture Depth_sensor
        self.depth_sensor_image = Image.open(depth_sensor_filename)
        self.depth_sensor_image = np.asarray(self.depth_sensor_image, dtype=np.float32) * 10
        self.depth_sensor_tex = mod.get_texref('depth_sensor')
        self.depth_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.depth_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
        self.depth_sensor_arr = drv.matrix_to_array(self.depth_sensor_image, 'C')
        self.depth_sensor_tex.set_array(self.depth_sensor_arr)
        self.shape = self.depth_sensor_image.shape

        # Setup and Upload texture IR_sensor
        self.ir_sensor_image = Image.open(ir_sensor_filename)
        self.ir_sensor_image = np.asarray(self.ir_sensor_image, dtype=np.float32)
        self.ir_sensor_tex = mod.get_texref('ir_sensor')
        self.ir_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.ir_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
        self.ir_sensor_arr = drv.matrix_to_array(self.ir_sensor_image, 'C')
        self.ir_sensor_tex.set_array(self.ir_sensor_arr)

        # Set up texture: Depth_current
        depth_current_image = np.zeros(self.shape, dtype=np.float32)
        self.depth_current_tex = mod.get_texref('depth_current')
        self.depth_current_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.depth_current_tex.set_address_mode(1, drv.address_mode.CLAMP)
        depth_current_arr = drv.matrix_to_array(depth_current_image, 'C')
        self.depth_current_tex.set_array(depth_current_arr)

        # Set up texture: Intensity current
        self.ir_current_image = np.zeros(self.shape, dtype=np.float32)
        self.ir_current_tex = mod.get_texref('intensity_current')
        self.ir_current_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.ir_current_tex.set_address_mode(1, drv.address_mode.CLAMP)
        self.ir_current_arr = drv.matrix_to_array(self.ir_current_image, 'C')
        self.ir_current_tex.set_array(self.ir_current_arr)

        # Set up temporary variables
        energy_intensity = np.zeros(self.shape, dtype=np.float32)
        energy_prime = np.zeros(self.shape, dtype=np.float32)
        intensity = np.zeros(self.shape, dtype=np.float32)
        energy_normal = np.zeros(self.shape, dtype=np.float32)
        normal = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.float32)

    def add_noise(self, image, mu=0.0, sigma=0.0):
        image = image + sigma * np.random.randn(*image.shape).astype(np.float32) + mu

    def callback(self, xk):
        self.iteration_counter += 1
        print('callback', self.iteration_counter)

    def energy(self, depth_fimage_current):
        """
         Calculate energy(depth_image_current)
        """
        # Todo write this, otherwise it won't optimize I think

        # Fixme obviously this is wrong
        print ('current energy: ', -depth_fimage_current.max())
        return -depth_fimage_current.max()

    def energy_prime(self, depth_fimage_current):
        """
         Calculate energy'(depth_image_current)
        """

        ir = np.zeros(self.shape, dtype=np.float32)
        energy_prime = np.zeros(self.shape, dtype=np.float32)

        self.intensity_function(
              drv.Out(ir),
              block=(16, 8, 1), grid=(32, 53))

        # Update IR current
        ir_arr = drv.matrix_to_array(ir, 'C')
        self.ir_current_tex.set_array(ir_arr)

        self.energy_prime_function(
              drv.Out(energy_prime),
              block=(16, 8, 1), grid=(32, 53))
        return energy_prime.flatten()

    def optimize(self, depth_image_start):
        ' Calls fmin_cg '

        self.iteration_counter = 0

        xopt, fopt, func_calls, grad_calls, warnflag = optimize.fmin_cg(self.energy, depth_image_start.flatten(),
                                                                        fprime=self.energy_prime,
                                                                        #epsilon=self.eps,
                                                                        maxiter=self.max_iterations,
                                                                        callback=self.callback,
                                                                        full_output=True)

        utils.show_image(depth_image_start, 'start image')

        print('shape',self.reshape_flat(xopt).shape)
        utils.show_image(self.reshape_flat(xopt), 'end image')

        return 0

    def reshape_flat(self, img_flat):
        # Reshapes flat image into array of right dimensions
        third = img_flat.size / (self.shape[0] * self.shape[1])
        if third == 1:
            return img_flat.reshape((self.shape[0], self.shape[1]))
        else:
            return img_flat.reshape((self.shape[0], self.shape[1], third))

path = '../assets/optimization'

optimizer = Optimizer('%s/head_depth.tiff' % path, '%s/head_ir.tiff' % path)

depth_sensor_image = Image.open('%s/head_depth.tiff' % path)
depth_sensor_image = np.asarray(depth_sensor_image, dtype=np.float32) * 10
optimizer.optimize(depth_sensor_image)

plt.show()
