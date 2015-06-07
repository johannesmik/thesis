__author__ = 'johannes'

import numpy as np
import pcl
import matplotlib.pyplot as plt
from scipy import misc
from  scipy import optimize
from skimage.filters import sobel, sobel_h, sobel_v, median, scharr_h, scharr_v
from skimage.morphology import disk
import cProfile
import time
import context, scenes, cameras, meshes, materials, lights

"""
    Naming convention:

        ___map = Image (not flattened)
"""

class Minimize(object):

    def __init__(self, sensor_depthmap, sensor_intensitymap, ground_depthmap, ground_intensitymap, normal='sobel',
                 glcontext=None, camera=None, scene=None, maxIterations=10):

        self.sensor_depthmap = sensor_depthmap
        self.sensor_intensitymap = sensor_intensitymap
        self.ground_depthmap = ground_depthmap
        self.ground_intensitymap = ground_intensitymap
        self.normal = normal

        self.height, self.width = sensor_depthmap.shape

        if not glcontext:
            self.context = context.Context(width=self.width, height=self.height, show_framerate=False)
        else:
            self.context = glcontext

        if not camera:
            self.camera = cameras.PerspectiveCamera()
        else:
            self.camera = camera

        if not scene:
            # Initialize Scene #TODO
                self.scene = scenes.NormalTexture(lighttype="point", material="lambertian")
        else:
            self.scene = scene

        self.scene_depth = scenes.DepthTexture(lighttype="point", material="normal")

        self.maxIterations = maxIterations

        self.material = self.scene.meshes[0].material # Material of first mesh

        self.ksearch = 15  # For PCL normal finding

        self.init_plot_variables()
        self.init_result_variables()

        # Calculate other variables
        self.sensor_normalmap = self.normals(sensor_depthmap)

    def init_plot_variables(self):
        ' (Re-) initializes the plot variables to zero '
        self.energy_total_plot = []
        self.energy_depth_plot = []
        self.energy_intensity_plot = []
        self.energy_normal_plot = []

    def init_result_variables(self):
        ' (Re-) initializes the result variables to zero '
        self.result_depthmap = np.zeros((self.height, self.width))
        self.result_normalmap = np.zeros((self.height, self.width))
        self.result_intensitymap = np.zeros((self.height, self.width))

        self.result_rmse_0 = 0
        self.result_rmse_1 = 0
        self.result_rmse_2 = 0
        self.result_rmse_3 = 0

    def callback(self, xk):
        ' Called after one iteration of the optimization. '

        energy_terms = self.energy(xk, returnall=True)

        print "iteration: ", energy_terms

        # Save those energys for plotting
        self.energy_total_plot.append(sum(energy_terms))
        self.energy_depth_plot.append(energy_terms[0])
        self.energy_intensity_plot.append(energy_terms[1])
        self.energy_normal_plot.append(energy_terms[2])

    def energy(self, depth_flat, returnall=False):

        depthmap = depth_flat.reshape((self.height, self.width))

        normals = self.normals(depthmap)

        intensity = self.intensity(normals)

        energy_terms = self._energy(intensity, depthmap, normals)

        if returnall:
            return energy_terms
        else:
            return sum(energy_terms)

    def _energy(self, intensity, depthmap, normal):

        energy_depth = np.sum((depthmap - self.sensor_depthmap) ** 2, axis=(0, 1))

        energy_intensity = np.sum((intensity - self.sensor_intensitymap) ** 2, axis=(0, 1))

        energy_normal = np.sum((sobel(normal[:,:,0]))**2)
        energy_normal += np.sum((sobel(normal[:,:,1]))**2)
        energy_normal += np.sum((sobel(normal[:,:,2]))**2)

        return energy_depth, energy_intensity, energy_normal

    def intensity(self, normalmap, depthmap=None):
        " Calculate the intensity from a normalmap (and a depthmap). [Depthmap for light attenuation reasons] "

        height, width, colors = normalmap.shape
        self.material.overwrite_texture('normalmap', np.flipud(normalmap).astype('uint8').tobytes(), width, height)
        self.context._render(self.scene, self.camera)
        intensity = self.context.current_buffer()[:,:,0]

        return intensity

    def normals(self, depthmap):
        if self.normal == 'sobel':
            normal = self.normals_from_sobel(depthmap)
            normal = ((normal + 1) / 2.) * 255
        elif self.normal == 'pcl':
            normal = self.normals_from_pcl(depthmap)
            normal = ((normal + 1) / 2.) * 255
        elif self.normal == 'opengl':
            normal = self.normals_from_opengl(depthmap)
        return normal

    def normals_from_pcl(self, depthmap):
        HEIGHT, WIDTH = depthmap.shape
        # Reprojection
        n, f = 1.0, 20.0
        l, r = -1.0, 1.0
        t, b = 1.0, -1.0
        xndc, yndc = np.meshgrid(np.linspace(-1, 1, WIDTH), np.linspace(1, -1, HEIGHT))
        zndc = 2 * (depthmap / 255.) - 1
        zeye = 2 * f * n / (zndc*(f-n)-(f+n))
        xeye = -zeye*(xndc*(r-l)+(r+l))/(2.0*n)
        yeye = -zeye*(yndc*(t-b)+(t+b))/(2.0*n)
        a = np.dstack((xeye, yeye, zeye))
        a = np.reshape(a, (WIDTH * HEIGHT, 3))
        a = a.astype(np.float32)

        # Calculate normals
        pointcloud = pcl.PointCloud()
        pointcloud.from_array(a)
        pclnormals = pointcloud.calc_normals(ksearch=self.ksearch)

        normals = np.ones((HEIGHT, WIDTH, 4))
        normals[:,:,:3] = pclnormals.reshape(WIDTH, HEIGHT, 3)
        return normals

    def normals_from_sobel(self, depthmap):
        ' Calculate the normals with sobel '
        HEIGHT, WIDTH = depthmap.shape
        normals = 255 * np.ones((HEIGHT, WIDTH, 4))
        normals[:,:,0] = depthmap * sobel_v(depthmap)
        normals[:,:,1] = depthmap * -sobel_h(depthmap)
        normals[:,:,2] = 255

        # Normalize such that the length is one, and the values range between -1 and 1
        normals = normals / 255.
        normals[:,:,:3] = normals[:,:,:3] / np.linalg.norm(normals[:,:,:3], axis=2)[:,:,np.newaxis]
        #normals[:,:,3] = 1
        return normals

    def normals_from_scharr(self, depthmap):
        # TODO
        print "normals from scharr not implemented yet"

    def normals_from_opengl(self, depthmap):

        height, width = depthmap.shape

        normal = 255 * np.ones((height, width, 4), dtype='float64')

        self.scene_depth.meshes[0].material.overwrite_texture_bw('depthmap', np.flipud(depthmap).astype('uint8').tobytes(), width, height)
        self.context._render(self.scene_depth, self.camera)
        normal[:,:,:3] = self.context.current_buffer()[:,:,:]

        return normal

    def optimize(self, start_depthmap):
        " Optimize start value and recalculate results. "

        # (Re-) Calculate start values
        self.start_depthmap = start_depthmap
        self.start_normalmap = self.normals(start_depthmap)
        self.start_intensitymap = self.intensity(self.start_normalmap)
        self.start_time = time.clock()

        # Reset the plots
        self.init_plot_variables()
        self.callback(start_depthmap.flatten())

        # (Re-) Calculate results

        xopt, fopt, func_calls, grad_calls, warnflag = optimize.fmin_cg(self.energy, start_depthmap.flatten(), epsilon=1, maxiter=self.maxIterations, callback=self.callback, full_output=True)

        self.result_depthmap = xopt.reshape((self.height, self.width))
        self.result_normalmap = self.normals(self.result_depthmap)
        self.result_intensitymap = self.intensity(self.result_normalmap)

        self.result_time = time.clock()
        self.result_seconds = self.result_time - self.start_time
        self.result_energy = fopt
        self.result_func_calls = func_calls
        self.result_grad_calls = grad_calls

        self.result_rmse_0 = np.sqrt( np.sum( (self.ground_depthmap - self.sensor_depthmap)**2, axis=(0,1) ) / (self.height * self.width) )
        self.result_rmse_1 = np.sqrt( np.sum( (self.ground_depthmap - self.start_depthmap)**2, axis=(0,1) ) / (self.height * self.width) )
        self.result_rmse_2 = np.sqrt( np.sum( (self.ground_depthmap - self.result_depthmap)**2, axis=(0,1) ) / (self.height * self.width) )
        self.result_rmse_3 = np.sqrt( np.sum( (self.ground_intensitymap - self.result_intensitymap)**2, axis=(0,1) ) / (self.height * self.width) )

    def save_results(self, name):
        ' Save results to files called "name_..." . '

        np.save('%s_start_depthmap.npy' % name, self.start_depthmap)
        np.save('%s_start_intensitymap.npy' % name, self.start_intensitymap)
        np.save('%s_result_depthmap.npy' % name, self.result_depthmap)

        misc.imsave("%s_start_depthmap.png" % name, self.start_depthmap)
        misc.imsave("%s_result_depthmap.png" % name, self.result_depthmap)

        text = "%s" % name
        text += "\n"
        text += "\nNormals: " + self.normal
        text += "\n"
        text += "\n Time needed: " + str(self.result_seconds)
        text += "\n Result energy: " + str(self.result_energy)
        text += "\n Func calls: " + str(self.result_func_calls)
        text += "\n Grad calls: " + str(self.result_grad_calls)
        text += "\n Time per func call: " + str(float(self.result_time) / self.result_func_calls)
        text += "\n"
        text += "\nRMSE(R_GT, R_noise): " + str(self.result_rmse_0)
        text += "\nRMSE(R_GT, R_start): " + str(self.result_rmse_1)
        text += "\nRMSE(R_GT, R_result): " + str(self.result_rmse_2)
        text += "\nRMSE(I_GT, I_result): " + str(self.result_rmse_3)

        with open("%s_result.txt" % name, 'w') as f:
            f.write(text)

    def show_results(self, plottitle="Plot: Optimization Results"):
        " Show results as a matplotlib plot "

        print "Time needed (seconds): ", self.result_seconds

        print "RMSE(R_GT, R_noise)", self.result_rmse_0
        print "RMSE(R_GT, R_start)", self.result_rmse_1
        print "RMSE(R_GT, R_result)", self.result_rmse_2
        print "RMSE(I_GT, I_result)", self.result_rmse_3

        # Lets plot the energies first
        plt.figure()
        plt.title("%s - Energys" % plottitle)
        plt.plot(self.energy_normal_plot)
        plt.plot(self.energy_depth_plot)
        plt.plot(self.energy_total_plot)
        plt.plot(self.energy_intensity_plot)
        plt.xlabel("iterations")
        plt.ylabel("energy")
        plt.legend(["normal", "depth", "total", "intensity"])

        # Now the images
        # Plots will be plotted in order
        results_to_plot = []

        results_to_plot.append((self.ground_depthmap, "Ground Depth"))
        results_to_plot.append((self.sensor_depthmap, "Sensor Depth"))
        results_to_plot.append((self.start_depthmap, "Start Depth"))
        results_to_plot.append((self.result_depthmap, "Result Depth"))
        results_to_plot.append((np.abs(self.start_depthmap - self.result_depthmap), "|Start - Result|"))

        results_to_plot.append((self.ground_intensitymap, "Ground Intensity"))
        results_to_plot.append((self.sensor_intensitymap, "Sensor Intensity"))
        results_to_plot.append((self.start_intensitymap, "Start Intensity"))
        results_to_plot.append((self.result_intensitymap, "Result Intensity"))
        results_to_plot.append((np.abs(self.start_intensitymap.astype("int64") - self.result_intensitymap.astype("int64")).astype("uint8"), "|Start - Result|"))

        results_to_plot.append((self.sensor_normalmap.astype('uint8')[:,:,:3], "Sensor Normal"))
        results_to_plot.append((self.start_normalmap.astype('uint8')[:,:,:3], "Start Normal"))
        results_to_plot.append((self.result_normalmap.astype('uint8')[:,:,:3], "Result Normal"))

        rows, columns = 3, 5

        plt.figure()
        plt.suptitle(plottitle)

        for i, result_to_plot in enumerate(results_to_plot):

            plot, title = result_to_plot

            plt.subplot(rows, columns, i+1)
            plt.title(title)
            plt.imshow(plot, cmap=plt.cm.gray, interpolation='nearest')
            plt.colorbar()


def open_as_array(filename):
    """ Opens image as float-array """
    img = misc.imread(filename)
    print "Opened ", filename, type(img), img.shape, img.dtype
    array = img.astype(float)
    return array

def sphere(glcontext):
    " Use glcontext to render an intensity and depth image of a Sphere "

    # Create Scene to render images
    scene = scenes.Scene(backgroundcolor=np.array([1, 1, 1, 1]))
    sphere_geometry = meshes.IcosphereGeometry(subdivisions=4)
    sphere_material = materials.LambertianMaterial()
    sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
    scene.add(sphere)

    square_material = materials.LambertianMaterial()
    square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -3]), geometry=meshes.SquareGeometry(), material=square_material)
    square.size = 3
    scene.add(square)

    light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)
    scene.add(light)

    camera = cameras.PerspectiveCamera()

    glcontext._render(scene, camera)
    intensitymap = glcontext.current_buffer()[:,:,0]

    square.material = materials.DepthMaterial()
    sphere.material = materials.DepthMaterial()
    glcontext._render(scene, camera)
    depthmap = glcontext.current_buffer()[:,:,0]

    return intensitymap.astype("float64"), depthmap.astype("float64")

def squares(glcontext):
    " Use glcontext to render an intensity and depth image of a Sphere "

    # Create Scene to render images
    scene = scenes.Scene(backgroundcolor=np.array([1, 1, 1, 1]))

    square_material = materials.LambertianMaterial()
    square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -3]), geometry=meshes.SquareGeometry(), material=square_material)
    square.size = 3
    scene.add(square)

    square2 = meshes.Mesh(name='Square 2', position=np.array([0, 0, -2.5]), geometry=meshes.SquareGeometry(), material=square_material)
    scene.add(square2)

    square3 = meshes.Mesh(name='Square 3', position=np.array([1, -1, -2.4]), geometry=meshes.SquareGeometry(), material=square_material)
    scene.add(square3)

    light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)
    scene.add(light)

    camera = cameras.PerspectiveCamera()

    glcontext._render(scene, camera)
    intensitymap = glcontext.current_buffer()[:,:,0]

    square.material = materials.DepthMaterial()
    square2.material = materials.DepthMaterial()
    square3.material = materials.DepthMaterial()
    glcontext._render(scene, camera)
    depthmap = glcontext.current_buffer()[:,:,0]

    return intensitymap.astype("float64"), depthmap.astype("float64")


if __name__ == '__main__':

    test = "1"

    if test == "1":

        # Fixed width and height
        # Different filtered input: Non-filtered, Median, 127-Start

        width, height = 50, 50

        glcontext = context.Context(width=width, height=height, show_framerate=False)

        intensitymap, depthmap = sphere(glcontext)
        # intensitymap = open_as_array("human-face/intensity_small.png")
        # depthmap = open_as_array("human-face/depth_small.png")
        ground_intensity = intensitymap.copy()
        ground_depth = depthmap.copy() # Ground True

        # Add some noise to depth
        mu, sigma = 0, 5
        sensor_depthmap = depthmap + sigma * np.random.randn(*depthmap.shape) + mu

        # Add some noise to intensity
        mu, sigma = 0, 5
        sensor_intensitymap = intensitymap + sigma * np.random.randn(*intensitymap.shape) + mu

        smooth_normal_minimizer = Minimize(sensor_depthmap, sensor_intensitymap, ground_depth, ground_intensity,
                                           normal='opengl', glcontext=glcontext, camera=None, scene=None, maxIterations=5)
        # Unfiltered Start Depth
        start_depthmap = sensor_depthmap.copy()

        smooth_normal_minimizer.optimize(start_depthmap)
        smooth_normal_minimizer.show_results(plottitle="Unfiltered Start Depth")
        smooth_normal_minimizer.save_results("results/unfiltered")

        #Median Filtered Start Depth
        start_depthmap = sensor_depthmap.copy()
        start_depthmap = np.clip(start_depthmap, 0, 255) / 255.0
        median(start_depthmap, disk(2), out=start_depthmap)

        smooth_normal_minimizer.optimize(start_depthmap)
        smooth_normal_minimizer.show_results(plottitle="Median Filtered Start Depth")
        smooth_normal_minimizer.save_results("results/medianfiltered")

        # 127 Start Depth
        start_depthmap = 127 * np.ones((height, width))

        smooth_normal_minimizer.optimize(start_depthmap)
        smooth_normal_minimizer.show_results(plottitle="127 Start Depth")
        smooth_normal_minimizer.save_results("results/zeros")

    elif test == "2":

        widths = [30, 60, 120, 512]
        heights = [30, 60, 120, 424]
        for width, height in zip(widths, heights):

            glcontext = context.Context(width=width, height=height, show_framerate=False)

            intensitymap, depthmap = sphere(glcontext)
            ground_intensity = intensitymap.copy()
            ground_depth = depthmap.copy() # Ground True

            mu, sigma = 0, 5
            sensor_depthmap = depthmap + sigma * np.random.randn(*depthmap.shape) + mu

            mu, sigma = 0, 5
            sensor_intensitymap = intensitymap + sigma * np.random.randn(*intensitymap.shape) + mu

            smooth_normal_minimizer = Minimize(sensor_depthmap, sensor_intensitymap, ground_depth, ground_intensity,
                                               normal='pcl', glcontext=glcontext, camera=None, scene=None, maxIterations=30)
            # Unfiltered Start Depth
            start_depthmap = sensor_depthmap.copy()

            smooth_normal_minimizer.optimize(start_depthmap)
            smooth_normal_minimizer.show_results(plottitle="%dx%d" % (width, height))
            smooth_normal_minimizer.save_results("results/%dx%d" % (width, height))



    plt.show()