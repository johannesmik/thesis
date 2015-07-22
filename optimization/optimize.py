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
// Should be uneven 5 or 7 or 9
#define m_depth 5
#define m_ir 5

#include <cuda.h>
#include "utils.cu"
#include "intensities.cu"
#include "normal.cu"

texture<float, cudaTextureType2D, cudaReadModeElementType> depth_sensor_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> depth_current_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_current_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_sensor_tex;

__device__ void set_depth_neighborhood(int2 pos, float neighborhood[m_depth][m_depth])
{
  /* returns the (m, m) depth neighborhood around pos. */
  /* Loads texture memory into global memory (slow) */

  const int left_border = (-1 + (m_ir + 1 )/ 2);

  for (int i = 0; i < m_depth; ++i) {
    for (int j = 0; j < m_depth; ++j) {
      neighborhood[j][i] = tex2D(depth_current_tex, pos.x - left_border + i, pos.y - left_border + j);
    }
  }
}

__device__ float ir_local(int2 center_pos, int2 change_pos, float adjustment){
  // Calculate the ir around center based on 'depth_current_tex', but adjust the depth of the pixel at change_pos
  // Center_pos: screen coords
  // Change_pos: screen coords

  LambertianLightingModel lightingmodel = LambertianLightingModel();

  float depth_neighborhood[m_depth][m_depth];
  set_depth_neighborhood(center_pos, depth_neighborhood);
  const float z = depth_neighborhood[(m_depth + 1) / 2][(m_depth + 1) / 2];

  const int left_border = (-1 + (m_ir + 1 )/ 2);

  // Adjust
  depth_neighborhood[change_pos.y - center_pos.y + left_border][change_pos.x - center_pos.x + left_border] += adjustment;

  const float3 normal = normal_cross(depth_neighborhood, center_pos);
  const float ir_return = lightingmodel.intensity(normal, pixel_to_camera(center_pos.x, center_pos.y, z));

  return ir_return;
}

inline __device__ float ir_local(int2 pos) {
  // Calculate the ir intensity at the midpoint of (m, m) depth neighborhood around pos
  return ir_local(pos, pos, 0.0);
}

inline __device__ float ir_local(int2 pos, float adjustment) {
  // Calculate the ir intensity around center, but adjust the depth of the pixel at center
  // pos: screen coords
  return ir_local(pos, pos, adjustment);
}

extern "C"
__global__ void energy_prime(float *energy_change_out){

  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  const float h = 0.001;

  // Calculate changes in the ir term

  // ir array before and from sensor
  float ir_before[m_ir][m_ir];
  float ir_sensor[m_ir][m_ir];

  // m_ir = 5 -> 2, 7 -> 3
  const int left_border = (-1 + (m_ir + 1 )/ 2);

  for (int i = 0; i < m_ir; ++i) {
    for (int j = 0; j < m_ir; ++j) {
      ir_before[j][i] = tex2D(ir_current_tex, x - left_border + i, y - left_border + j);
      ir_sensor[j][i] = tex2D(ir_sensor_tex, x - left_border + i, y - left_border + j);
    }
  }

  float ir_after[m_ir][m_ir];
  for (int i = 0; i < m_ir; ++i) {
    for (int j = 0; j < m_ir; ++j) {
      int2 center = make_int2(x + i - left_border, y + j - left_border);
      int2 change = make_int2(x, y);
      ir_after[j][i] = ir_local(center, change, h);
    }
  }

  float ir_term = 0;
  for (int i = 0; i < m_ir; ++i) {
    for (int j = 0; j < m_ir; ++j) {
      ir_term += pow(ir_after[j][i] - ir_sensor[j][i], 2) - pow(ir_before[j][i] - ir_sensor[j][i], 2);
    }
  }

  ir_term /= h;

  // Calculate changes in the normal term

  // Return
  //energy_change_out[index] = ir_before[3][3];
  energy_change_out[index] = ir_term;

}

extern "C"
__global__ void ir_image(float *ir_out)
{
    // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  ir_out[index] = ir_local(make_int2(x, y));
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
__global__ void energy(float *energy_ir_out)
{
  // FIXME calculates the wrong energy

  /* Steps to do:
    - Calculate ir intensity
    - Calculate normal of current point and adjacent pixels
    - Calculate energy: ( ir intensity - ir_sensor )
  */

  LambertianLightingModel lightingmodel = LambertianLightingModel();

  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  float depth_neighborhood[m_depth][m_depth];
  set_depth_neighborhood(make_int2(x, y), depth_neighborhood);

  float ir_test = ir_local(make_int2(x, y));

  float3 normal = normal_pca(depth_neighborhood, make_int2(x, y));
  // float3 normal_c = normal_colorize(normal);

  float ir_given = tex2D(ir_sensor_tex, x, y);
  float ir_new = lightingmodel.intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor_tex, x, y)));

  energy_ir_out[index] = pow(ir_given - ir_new, 2);
  // ir_out[index] = ir_test;
  // normal_out[index] = normal_c;

}
""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])

class Optimizer(object):

    def __init__(self, depth_sensor_filename, ir_sensor_filename):

        self.eps = 0.0001
        self.max_iterations = 20

        self.iteration_counter = 0

        # Counter for debug reasons
        # Todo Delete at sometime :)
        self.debug_counter = 0

        # Setup CUDA
        self.energy_function = mod.get_function("energy")
        self.energy_normal_function = mod.get_function("energy_normal")
        self.energy_prime_function = mod.get_function("energy_prime")
        self.ir_function = mod.get_function("ir_image")

        # Setup and Upload texture Depth_sensor
        self.depth_sensor_image = Image.open(depth_sensor_filename)
        self.depth_sensor_image = np.asarray(self.depth_sensor_image, dtype=np.float32) * 10
        self.depth_sensor_tex = mod.get_texref('depth_sensor_tex')
        self.depth_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.depth_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
        self.depth_sensor_arr = drv.matrix_to_array(self.depth_sensor_image, 'C')
        self.depth_sensor_tex.set_array(self.depth_sensor_arr)
        self.shape = self.depth_sensor_image.shape

        # Setup and Upload texture IR_sensor
        self.ir_sensor_image = Image.open(ir_sensor_filename)
        self.ir_sensor_image = np.asarray(self.ir_sensor_image, dtype=np.float32)
        self.ir_sensor_tex = mod.get_texref('ir_sensor_tex')
        self.ir_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.ir_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
        self.ir_sensor_arr = drv.matrix_to_array(self.ir_sensor_image, 'C')
        self.ir_sensor_tex.set_array(self.ir_sensor_arr)

        # Set up texture: Depth_current
        depth_current_image = np.zeros(self.shape, dtype=np.float32)
        self.depth_current_tex = mod.get_texref('depth_current_tex')
        self.depth_current_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.depth_current_tex.set_address_mode(1, drv.address_mode.CLAMP)
        depth_current_arr = drv.matrix_to_array(depth_current_image, 'C')
        self.depth_current_tex.set_array(depth_current_arr)

        # Set up texture: Intensity current
        self.ir_current_image = np.zeros(self.shape, dtype=np.float32)
        self.ir_current_tex = mod.get_texref('ir_current_tex')
        self.ir_current_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.ir_current_tex.set_address_mode(1, drv.address_mode.CLAMP)
        self.ir_current_arr = drv.matrix_to_array(self.ir_current_image, 'C')
        self.ir_current_tex.set_array(self.ir_current_arr)

    def add_noise(self, image, mu=0.0, sigma=0.0):
        image = image + sigma * np.random.randn(*image.shape).astype(np.float32) + mu

    def callback(self, xk):
        self.iteration_counter += 1
        print('callback', self.iteration_counter)

    def energy(self, depth_fimage_current):
        """
         Calculate energy(depth_image_current)
        """
        # TODO way to log the single energy terms

        self.set_current_depth(self.reshape_flat(depth_fimage_current))

        energy_out = np.zeros(self.shape, dtype=np.float32)

        self.energy_function(
            drv.Out(energy_out),
            block=(16, 8, 1), grid=(32, 53))

        #utils.show_image(energy_out, title='energy out')
        print('energy', np.sum(energy_out) * 1000)

        return np.sum(energy_out) * 1000

    def energy_prime(self, depth_fimage_current):
        """
         Calculate energy'(depth_image_current)
        """

        self.set_current_depth(self.reshape_flat(depth_fimage_current))

        ir = np.zeros(self.shape, dtype=np.float32)
        energy_prime = np.zeros(self.shape, dtype=np.float32)

        self.ir_function(
              drv.Out(ir),
              block=(16, 8, 1), grid=(32, 53))

        # Update IR current
        ir_arr = drv.matrix_to_array(ir, 'C')
        self.ir_current_tex.set_array(ir_arr)

        self.energy_prime_function(
              drv.Out(energy_prime),
              block=(16, 8, 1), grid=(32, 53))

        energy_prime = energy_prime / 100

        print ('energy_prime, min: ', energy_prime.min(), ' max: ', energy_prime.max())

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

        utils.show_image(self.ir_sensor_image, 'ir sensor image')

        ir_start = self.relight_depth(depth_image_start)
        utils.show_image(ir_start, 'ir start')

        ir_end = self.relight_depth(self.reshape_flat(xopt))
        utils.show_image(ir_end, 'ir end')

        return 0

    def reshape_flat(self, img_flat):
        # Reshapes flat image into array of right dimensions
        third = img_flat.size / (self.shape[0] * self.shape[1])
        if third == 1:
            return img_flat.reshape((self.shape[0], self.shape[1]))
        else:
            return img_flat.reshape((self.shape[0], self.shape[1], third))

    def relight_depth(self, depth_image):

        # TODO candidate for function set_current_depth
        self.set_current_depth(depth_image)

        ir = np.zeros(self.shape, dtype=np.float32)

        self.ir_function(
              drv.Out(ir),
              block=(16, 8, 1), grid=(32, 53))

        return ir

    def set_current_depth(self, depth_image):
        depth_arr = drv.matrix_to_array(depth_image, 'C')
        self.depth_current_tex.set_array(depth_arr)

path = '../assets/optimization'

optimizer = Optimizer('%s/head_depth.tiff' % path, '%s/head_ir.tiff' % path)

depth_sensor_image = Image.open('%s/head_depth.tiff' % path)
depth_sensor_image = np.asarray(depth_sensor_image, dtype=np.float32) * 10
optimizer.optimize(depth_sensor_image)

plt.show()
