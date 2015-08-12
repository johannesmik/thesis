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
#include "lights.cu"

#include "global_functions.cu"
""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])

energy_function = mod.get_function("energy")
#energy_normal_function = mod.get_function("energy_normal")
energy_prime_function = mod.get_function("energy_prime")
intensity_function = mod.get_function("intensity_image")

use_testimage = False
if use_testimage:
    depth_image = Image.open("testimage.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32)
    depth_image = depth_image * 10.0 / 255.
else:
    depth_image = Image.open("../assets/optimization/sphere_depth.tiff")
    depth_image = Image.open("../assets/optimization/head_depth.tiff")
    #depth_image = depth_image.filter(ImageFilter.GaussianBlur(2))
    depth_image = np.asarray(depth_image, dtype=np.float32)
    depth_image = depth_image * 10

depth_sensor_image = depth_image

ir_intensity_image = Image.open("../assets/optimization/head_ir.tiff")
ir_intensity_image = np.asarray(ir_intensity_image, dtype=np.float32)
#ir_intensity_image = ir_intensity_image[:,:,0] / 255.

mu, sigma = 0, 0.0
depth_image = depth_image + sigma * np.random.randn(*depth_image.shape).astype(np.float32) + mu
print("min/max of depth_image", depth_image.min(), depth_image.max())

# Set up textures
depth_tex = mod.get_texref('depth_current_tex')
depth_tex.set_address_mode(0, drv.address_mode.CLAMP)
depth_tex.set_address_mode(1, drv.address_mode.CLAMP)
depth_image_arr = drv.matrix_to_array(depth_image, 'C')
depth_tex.set_array(depth_image_arr)

depth_sensor_tex = mod.get_texref('depth_sensor_tex')
depth_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
depth_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
depth_sensor_image_arr = drv.matrix_to_array(depth_sensor_image, 'C')
depth_sensor_tex.set_array(depth_sensor_image_arr)

utils.show_image(depth_image, 'depth image')

# Set up texture: Intensity Sensor
intensity_sensor = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
intensity_sensor_tex = mod.get_texref('ir_sensor_tex')
intensity_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
intensity_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
intensity_sensor_arr = drv.matrix_to_array(ir_intensity_image, 'C')
intensity_sensor_tex.set_array(intensity_sensor_arr)

# Set up texture: Intensity current
intensity_current = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
intensity_current_tex = mod.get_texref('ir_current_tex')
intensity_current_tex.set_address_mode(0, drv.address_mode.CLAMP)
intensity_current_tex.set_address_mode(1, drv.address_mode.CLAMP)
intensity_current_arr = drv.matrix_to_array(intensity_current, 'C')
intensity_current_tex.set_array(intensity_current_arr)

# Set up texture: material current
k_d = 0.5 * np.ones((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
k_s = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
n = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
margin = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
material_current = np.dstack((k_d, k_s, n, margin))
material_current_tex = mod.get_texref('material_current_tex')
utils.set_texture_border_mode(material_current_tex, mode='clamp')
material_current_arr = drv.make_multichannel_2d_array(material_current, 'C')
material_current_tex.set_array(material_current_arr)

energy_intensity = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
energy_prime_depth = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
energy_prime_material = np.zeros((depth_image.shape[0], depth_image.shape[1], 4), dtype=np.float32)
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
         drv.Out(energy_prime_depth), drv.Out(energy_prime_material),
         block=(16, 8, 1), grid=(32, 53))

    energy_data_term = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
    energy_shading_constraint = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
    energy_shape_prior = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)
    energy_material_prior = np.zeros((depth_image.shape[0], depth_image.shape[1]), dtype=np.float32)

    energy_function(
         drv.Out(energy_data_term), drv.Out(energy_shading_constraint),
         drv.Out(energy_shape_prior), drv.Out(energy_material_prior),
         block=(16, 8, 1), grid=(32, 53))
    #
    # energy_normal_function(
    #     drv.In(normal), drv.Out(energy_normal),
    #     block=(16, 8, 1), grid=(32, 53))

utils.show_image(energy_prime_depth, title='energy prime depth')

# utils.show_image(energy_prime_material[:,:,0], title='Material 0')
# utils.show_image(energy_prime_material[:,:,1], title='Material 1')
# utils.show_image(energy_prime_material[:,:,2], title='Material 2')
# utils.show_image(energy_data_term, title="Data Term")
# utils.show_image(energy_shading_constraint, title='Shading Constraint')
# utils.show_image(energy_shape_prior, title='shape priour')
# utils.show_image(energy_material_prior, title='material prior')

#utils.show_image(ir_intensity_image, title="IR Intensity Image")
#utils.show_image(energy_intensity, title="Energy Intensity")
# utils.show_image(energy_normal, title="Energy Normal")
# utils.show_image(intensity, title="Intensity")
# utils.show_image(normal, title="Normal")
print("energy:", energy_intensity.sum())

plt.show()
