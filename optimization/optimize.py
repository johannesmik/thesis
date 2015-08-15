from __future__ import print_function

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

#include "global_functions.cu"

""", no_extern_c=True, include_dirs=[os.getcwd() + '/cuda'])

class Optimizer(object):

    def __init__(self, depth_sensor_image, ir_sensor_image, lightingmodel='diffuse', normalmodel='pca',
                 depth_variance=0.001, ir_variance=0.001, w_d=1, w_m=50):

        self.eps = 0.0001
        self.max_iterations = 20

        self.iteration_counter = 0

        # Counter for debug reasons
        # Todo Delete at sometime :)
        self.debug_counter = 0

        # Setup Lighting model
        lightingmodels = {'diffuse': 0, 'specular' : 1}
        self.lightingmodel = lightingmodels[lightingmodel]

        # Setup normalmodel
        normalmodels = {'cross': 0, 'pca' : 1}
        self.normalmodel = normalmodels[normalmodel]

        self.depth_variance = depth_variance
        self.ir_variance = ir_variance
        self.w_d = w_d
        self.w_m = w_m

        # Setup CUDA functions
        self.energy_function = mod.get_function("energy")
        self.energy_prime_function = mod.get_function("energy_prime")
        self.ir_function = mod.get_function("intensity_image")
        self.normal_function = mod.get_function("normal_pca")

        # Setup and Upload texture Depth_sensor
        self.depth_sensor_image = np.asarray(depth_sensor_image, dtype=np.float32)
        self.depth_sensor_tex = mod.get_texref('depth_sensor_tex')
        self.depth_sensor_tex.set_address_mode(0, drv.address_mode.CLAMP)
        self.depth_sensor_tex.set_address_mode(1, drv.address_mode.CLAMP)
        self.depth_sensor_arr = drv.matrix_to_array(self.depth_sensor_image, 'C')
        self.depth_sensor_tex.set_array(self.depth_sensor_arr)
        self.shape = self.depth_sensor_image.shape

        # Setup and Upload texture IR_sensor
        self.ir_sensor_image = np.asarray(ir_sensor_image, dtype=np.float32)
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

        # Set up texture: material current
        self.material_current = np.zeros((depth_current_image.shape[0], depth_current_image.shape[1], 4), dtype=np.float32)
        self.material_current_tex = mod.get_texref('material_current_tex')
        utils.set_texture_border_mode(self.material_current_tex, mode='clamp')
        self.material_current_arr = drv.make_multichannel_2d_array(self.material_current, 'C')
        self.material_current_tex.set_array(self.material_current_arr)

        self.energy_log = []

    def add_noise(self, image, mu=0.0, sigma=0.0):
        image = image + sigma * np.random.randn(*image.shape).astype(np.float32) + mu

    def callback(self, xk):
        dt, sc, sp, mp = self.energy(xk, return_terms=True)
        self.iteration_counter += 1
        self.energy_log.append([dt, sc, sp, mp])
        print('callback', self.iteration_counter)
        print('energy data sc sp mp', dt, sc, sp, mp)

    def energy(self, depth_material_fimage_current, return_terms=False):
        """
         Calculate energy(depth_image_current)

         :param return_terms: If set to true then a list of [dataterm, shading constraint, shape prior, material prior]
                              is returned.
        """

        depth_image = self.reshape_flat(depth_material_fimage_current[:(512*424)])
        material_image = self.reshape_flat(depth_material_fimage_current[(512*424):])

        self.set_current_depth(depth_image)
        self.set_current_material(material_image)

        energy_data_term = np.zeros(self.shape, dtype=np.float32)
        energy_shading_constraint = np.zeros(self.shape, dtype=np.float32)
        energy_shape_prior = np.zeros(self.shape, dtype=np.float32)
        energy_material_prior = np.zeros(self.shape, dtype=np.float32)


        self.energy_function(
             np.int32(self.lightingmodel), np.int32(self.normalmodel),
             np.float32(self.depth_variance), np.float32(self.ir_variance),
             np.float32(self.w_d), np.float32(self.w_m),
             drv.Out(energy_data_term), drv.Out(energy_shading_constraint),
             drv.Out(energy_shape_prior), drv.Out(energy_material_prior),
             block=(16, 8, 1), grid=(32, 53))

        energy_total = np.sum(energy_data_term + energy_shading_constraint + energy_shape_prior + energy_material_prior)
        #utils.show_image(energy_out, title='energy out')

        if return_terms:
            return [np.sum(energy_data_term),
              np.sum(energy_shading_constraint), np.sum(energy_shape_prior), np.sum(energy_material_prior)]
        else:
            return energy_total

    def energy_prime(self, depth_material_fimage_current):
        """
         Calculate energy'(depth_image_current)
        """

        depth_image = self.reshape_flat(depth_material_fimage_current[:(512*424)])
        material_image = self.reshape_flat(depth_material_fimage_current[(512*424):])

        self.set_current_depth(depth_image)
        self.set_current_material(material_image)

        # TODO: do i really need to update the IR here?
        ir = np.zeros(self.shape, dtype=np.float32)

        self.ir_function(
              np.int32(self.lightingmodel),
              np.int32(self.normalmodel),
              drv.Out(ir),
              block=(16, 8, 1), grid=(32, 53))

        # Update IR current
        ir_arr = drv.matrix_to_array(ir, 'C')
        self.ir_current_tex.set_array(ir_arr)
        # TODO end

        depth_prime = np.zeros(self.shape, dtype=np.float32)
        material_prime = np.zeros((self.shape[0], self.shape[1], 4), dtype=np.float32)
        self.energy_prime_function(
               np.int32(self.lightingmodel), np.int32(self.normalmodel),
               np.float32(self.depth_variance), np.float32(self.ir_variance),
               np.float32(self.w_d), np.float32(self.w_m),
               drv.Out(depth_prime), drv.Out(material_prime),
               block=(16, 8, 1), grid=(32, 53))

        energy_prime = depth_prime / 100

        #print ('depth_prime, min/max', depth_prime.min(), depth_prime.max())
        #print ('material_prime, min/max', material_prime.min(), material_prime.max())

        return np.concatenate((depth_prime.flatten(), material_prime.flatten()))

    def optimize(self, depth_image_start, material_image_start):
        ' Calls fmin_cg '

        self.iteration_counter = 0

        xstart = np.concatenate((depth_image_start.flatten(), material_image_start.flatten()))
        self.callback(xstart)

        xopt, fopt, func_calls, grad_calls, warnflag = optimize.fmin_cg(self.energy,
                                                                        xstart,
                                                                        fprime=self.energy_prime,
                                                                        # epsilon=self.eps,
                                                                        maxiter=self.max_iterations,
                                                                        callback=self.callback,
                                                                        full_output=True)

        # Sensor images
        utils.show_image(self.ir_sensor_image, 'ir sensor image')
        utils.show_image(self.depth_sensor_image, 'depth sensor image')

        # Start images
        utils.show_image(depth_image_start, 'start depth image')
        utils.show_image(material_image_start[:,:,0], 'start material $k_d$ image')
        utils.show_image(material_image_start[:,:,1], 'start material $k_s$ image')
        utils.show_image(material_image_start[:,:,2], 'start material $n$ image')

        # Optimal (found) Depth and Material image
        depth_image_opt = self.reshape_flat(xopt[:(512*424)])
        material_image_opt = self.reshape_flat(xopt[(512*424):])

        utils.show_image(depth_image_opt, 'depth image found optimum')
        utils.show_image(material_image_opt[:,:,0], 'material $k_d$ found optimum')
        utils.show_image(material_image_opt[:,:,1], 'material $k_s$ found optimum')
        utils.show_image(material_image_opt[:,:,2], 'material $n$ found optimum')

        # Infrared images
        ir_start = self.relight_depth(depth_image_start, material_image_start)
        utils.show_image(ir_start, 'ir start')

        ir_end = self.relight_depth(depth_image_opt, material_image_opt)
        utils.show_image(ir_end, 'ir end')

        # Normal images
        normal_start = self.calculate_normal(depth_image_start)
        utils.show_image(normal_start, 'normal start')

        normal_end = self.calculate_normal(depth_image_opt)
        utils.show_image(normal_end, 'normal end (found opt)')

        energy_log = np.array(self.energy_log)
        if len(energy_log) > 0:

            plt.figure()
            plt.title('Data term')
            plt.plot(energy_log[:,0])
            plt.figure()
            plt.title('Shading constraint term')
            plt.plot(energy_log[:,1])
            plt.figure()
            plt.title('Shape prior term')
            plt.plot(energy_log[:,2])
            plt.figure()
            plt.title('Material prior term')
            plt.plot(energy_log[:,3])
            plt.figure()
            plt.title('Total energy')
            plt.plot(energy_log[:,0] + energy_log[:,1] + energy_log[:,2] + energy_log[:,3])

        return 0

    def reshape_flat(self, img_flat):
        # Reshapes flat image into array of right dimensions
        third = img_flat.size / (self.shape[0] * self.shape[1])
        if third == 1:
            return img_flat.reshape((self.shape[0], self.shape[1]))
        else:
            return img_flat.reshape((self.shape[0], self.shape[1], third))

    def relight_depth(self, depth_image, material_image):

        self.set_current_depth(depth_image)
        self.set_current_material(material_image)


        ir = np.zeros(self.shape, dtype=np.float32)

        self.ir_function(
              np.int32(self.lightingmodel),
              np.int32(self.normalmodel),
              drv.Out(ir),
              block=(16, 8, 1), grid=(32, 53))

        return ir

    def calculate_normal(self, depth_image):

        self.set_current_depth(depth_image)

        normal = np.zeros((424, 512, 3), dtype=np.float32)

        self.normal_function(
            drv.Out(normal),
            block=(16, 8, 1), grid=(32, 53))

        return normal


    def set_current_depth(self, depth_image):
        depth_arr = drv.matrix_to_array(depth_image, 'C')
        self.depth_current_tex.set_array(depth_arr)

    def set_current_material(self, material_image):
        """
        :param material_image: a 424x512x4 array
        """
        self.material_current = material_image
        self.material_current_arr = drv.make_multichannel_2d_array(self.material_current, 'C')
        self.material_current_tex.set_array(self.material_current_arr)

