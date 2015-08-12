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
#define m_depth 5
#define m_ir 5

#include "global_functions.cu"

""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])

normal_function = mod.get_function("normal_pca")

use_testimage = True
if use_testimage:
    depth_image = Image.open("../assets/optimization/testimage.tiff")
    depth_image = Image.open("../assets/clustering/SpecularSphere_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32)
    depth_image = depth_image * 10.0
else:
    depth_image = Image.open("../assets/testimages/sphere-depth.png")
    depth_image = Image.open("../assets/optimization/head_depth.tiff")
    #depth_image = depth_image.filter(ImageFilter.GaussianBlur(2))
    depth_image = np.asarray(depth_image, dtype=np.float32)
    depth_image = depth_image * 10

mu, sigma = 0, 0.0
depth_image = depth_image + sigma * np.random.randn(*depth_image.shape).astype(np.float32) + mu

# Set up textures
depth_tex = mod.get_texref('depth_current_tex')
utils.set_texture_border_mode(depth_tex, mode='clamp')
utils.update_texture(depth_tex, depth_image)

normal = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.float32)

intensity_change = np.zeros_like(depth_image)

for i in range(10):
    normal_function(
    drv.Out(normal),
    block=(16, 8, 1), grid=(32, 53))

utils.show_image(depth_image, title=None)
utils.show_image(normal, title=None, colorbar=False)

# image = Image.fromarray(np.uint8(normal * 255))
# #image = image.transpose(Image.FLIP_TOP_BOTTOM)
# with open('test_normal.tiff', 'w') as f:
#     image.save(f)

plt.show()
