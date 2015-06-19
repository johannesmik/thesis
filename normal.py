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

    im = plt.imshow(image_copy, interpolation="nearest", cmap=cm, vmin=0.0, vmax=1.0)

    # Set up the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.05)
    clb = plt.colorbar(im, cax)


mod = SourceModule("""
__global__ void normal(float *dest, float *img)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  // Find the neighbours

  dest[index] = img[index];
}
""")

multiply_them = mod.get_function("normal")

image = Image.open("sphere_depth.tiff")
image = np.asarray(image, dtype=np.float32)
show_image(image)

dest = np.zeros_like(image)
multiply_them(
    drv.Out(dest), drv.In(image),
    block=(16, 8, 1), grid=(32, 53))

show_image(dest)

plt.show()



