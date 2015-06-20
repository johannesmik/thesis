import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule


def show_image(image):

    image_copy = image.copy()

    fig = plt.figure(figsize=(4, 3))
    ax = plt.gca()

    # Use grey colormap
    cm = plt.get_cmap("gray")

    # Show points that fall out of the region [0, 1] in pink
    cm.set_under('pink')
    cm.set_over('pink')

    # No ticks on image axis
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    im = plt.imshow(image_copy, interpolation="nearest", cmap=cm, vmin=-1.0, vmax=1.0)

    # Set up the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.05)
    clb = plt.colorbar(im, cax)


mod = SourceModule("""
#include <stdint.h>
#include <cuda.h>
#include <surface_functions.h>
texture<float, cudaTextureType2D, cudaReadModeElementType> tex_in;

__global__ void normal(float *dest)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  //float value = tex2D(tex_in, x, y);

  // Find the neighbours
  const float value_b = tex2D(tex_in, x, y-1);
  //const float value_d = tex2D(tex_in, x-1, y);
  //const float value_f = tex2D(tex_in, x+1, y);
  const float value_h = tex2D(tex_in, x, y+1);

  dest[index] = value_h - value_b;

}
""")

multiply_them = mod.get_function("normal")
tex_in = mod.get_texref('tex_in')
tex_in.set_address_mode(1, drv.address_mode.WRAP)
tex_in.set_address_mode(2, drv.address_mode.WRAP)

image = Image.open("sphere_depth.tiff")
image = np.asarray(image, dtype=np.float32)
show_image(image)

image_arr = drv.matrix_to_array(image, 'C')
tex_in.set_array(image_arr)
dest = np.zeros_like(image)

multiply_them(
    drv.Out(dest),
    block=(16, 8, 1), grid=(32, 53))

show_image(dest)

plt.show()



