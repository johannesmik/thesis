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

def show_image(image, title=None):

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

    print("Show Image: Chosen min/max", minimum, maximum, " Real min/max", image_copy.min(), image_copy.max())

    if title:
        plt.title(title)

    im = plt.imshow(image_copy, interpolation="nearest", cmap=cm, vmin=minimum, vmax=maximum)

    # Set up the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.05)
    clb = plt.colorbar(im, cax)

mod = SourceModule("""
#include <cuda.h>
#include "utils.cu"
#include "intensities.cu"
#include "normal.cu"

texture<float, cudaTextureType2D, cudaReadModeElementType> depth;

__device__ void neighborhood(int2 pos, float *neighborhood)
{
  /* returns the 5x5 depth neighborhood around pos. */
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      neighborhood[j*5 + i] = tex2D(depth, pos.x -2 + i, pos.y -2 + j);
    }
  }
}

extern "C"
__global__ void normal(float *normal_out)
{
  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  float depth_neighborhood[25];
  neighborhood(make_int2(x, y), depth_neighborhood);

  float3 normal = normal_cross(depth_neighborhood, x, y);
  float3 normal_c = normal_colorize(normal);

  normal_out[index * 3] = normal_c.x;
  normal_out[index * 3 + 1] = normal_c.y;
  normal_out[index * 3 + 2] = normal_c.z;
}
""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])

normal_function = mod.get_function("normal")

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

mu, sigma = 0, 0.0
depth_image = depth_image + sigma * np.random.randn(*depth_image.shape).astype(np.float32) + mu
print("min/max of depth_image", depth_image.min(), depth_image.max())

# Set up textures
depth_tex = mod.get_texref('depth')
depth_tex.set_address_mode(0, drv.address_mode.CLAMP)
depth_tex.set_address_mode(1, drv.address_mode.CLAMP)
depth_image_arr = drv.matrix_to_array(depth_image, 'C')
depth_tex.set_array(depth_image_arr)

normal = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)

intensity_change = np.zeros_like(depth_image)

for i in range(1000):
    normal_function(
    drv.Out(normal),
    block=(16, 8, 1), grid=(32, 53))

show_image(depth_image, title="Depth Image")
show_image(normal, title="Normal Image")

plt.show()
