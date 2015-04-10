__author__ = 'johannes'

import numpy as np
import pcl
import matplotlib.pyplot as plt
from scipy import misc
from  scipy import optimize
from skimage.filters import sobel, sobel_h, sobel_v, median, scharr_h, scharr_v
from skimage.morphology import disk
import cProfile
import context, scenes, cameras

def show_info(depth, depth_o, intensity, intensity_o, normal):
    plt.figure()
    plt.subplot(3, 2, 1)
    plt.title('Received Depth')
    plt.imshow(depth, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.subplot(3, 2, 2)
    plt.title('Intensity Difference')
    plt.imshow(np.abs(intensity_o - intensity), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.subplot(3, 2, 3)
    plt.title('Original Depth')
    plt.imshow(depth_o, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.subplot(3, 2, 4)
    plt.title('Original Intensity')
    plt.imshow(intensity_o, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.subplot(3, 2, 5)
    plt.title('Normal (PCL)')
    plt.imshow(normal.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.subplot(3, 2, 6)
    plt.title('Intensity from OpenGL')
    plt.imshow(intensity, cmap=plt.cm.gray, interpolation='nearest')

def opt_callback(xk):
    ' xk: current value of x '

    energy_total, energy_depth, energy_intensity, energy_normal = energy(xk, returnall=True)
    # Save those energys for plotting
    print "iteration: ", energy_total, energy_depth, energy_intensity, energy_normal
    global energy_total_plot, energy_depth_plot, energy_intensity_plot, energy_normal_plot
    energy_total_plot.append(energy_total)
    energy_depth_plot.append(energy_depth)
    energy_intensity_plot.append(energy_intensity)
    energy_normal_plot.append(energy_normal)

def energy(depth, returnall=False):

    depth = depth.reshape((WIDTH, HEIGHT))

    global TIMESTEP, depth_gt, intensity_gt, glcontext, smoothnormals
    TIMESTEP += 1

    #normal = normals_from_depth_pcl(depth, ksearch=15)
    normal = normals_from_depth_sobel(depth)
    normal = ((normal + 1) / 2.) * 255

    intensity = intensity_from_normals(normal, glcontext)

    energy_depth = np.sum((depth - depth_gt) ** 2, axis=(0, 1))
    energy_intensity = np.sum((intensity - intensity_gt) ** 2, axis=(0, 1))
    energy_normal = np.sum((sobel(normal[:,:,0]))**2)
    energy_normal += np.sum((sobel(normal[:,:,1]))**2)
    energy_normal += np.sum((sobel(normal[:,:,2]))**2)

    if TIMESTEP == 1 or TIMESTEP == 400:
        #show_info(depth, depth_gt, intensity, intensity_gt, normal)
        pass

    energy = 0

    energy += energy_depth

    energy += energy_intensity

    if smoothnormals:
        energy += energy_normal

    if returnall:
        return energy, energy_depth, energy_intensity, energy_normal
    else:
        return energy

def intensity_from_normals(normals, glcontext):

    glcontext.material.overwrite_texture(glcontext.normalmap_id, np.flipud(normals).astype('uint8').tobytes(), WIDTH, HEIGHT)
    glcontext._render(glcontext.scene, glcontext.camera)
    intensity = glcontext.current_buffer()[:,:,0]

    return intensity

def normals_from_depth_pcl(depth, ksearch=10):
    ''' Calculate normals with pcl '''

    # Reprojection
    n, f = 1.0, 20.0
    l, r = -1.0, 1.0
    t, b = 1.0, -1.0
    xndc, yndc = np.meshgrid(np.linspace(-1, 1, WIDTH), np.linspace(1, -1, HEIGHT))
    zndc = 2 * (depth / 255.) - 1
    zeye = 2 * f * n / (zndc*(f-n)-(f+n))
    xeye = -zeye*(xndc*(r-l)+(r+l))/(2.0*n)
    yeye = -zeye*(yndc*(t-b)+(t+b))/(2.0*n)
    a = np.dstack((xeye, yeye, zeye))
    a = np.reshape(a, (WIDTH * HEIGHT, 3))
    a = a.astype(np.float32)

    # Calculate normals
    pointcloud = pcl.PointCloud()
    pointcloud.from_array(a)
    pclnormals = pointcloud.calc_normals(ksearch=ksearch)

    normals = np.ones((HEIGHT, WIDTH, 4))
    normals[:,:,:3] = pclnormals.reshape(WIDTH, HEIGHT, 3)
    return normals

def normals_from_depth_sobel(depth):
    ' Calculate the normals with sobel '
    # TODO not correct yet
    HEIGHT, WIDTH = depth.shape
    normals = 127 * np.ones((HEIGHT, WIDTH, 4))
    normals[:,:,0] = sobel_v(depth)
    normals[:,:,1] = sobel_h(depth)

    # Normalize such that the length is one, and the values range between -1 and 1
    normals = normals / 255.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,np.newaxis]
    normals[:,:,3] = 1

    return normals

def normals_from_depth_scharr(depth):
    ' Calculate the normals with sobel '
    # TODO not correct yet
    depth /= 255
    normals = np.ones((HEIGHT, WIDTH, 4))
    normals[:,:,0] =  scharr_v(depth)
    normals[:,:,1] =  scharr_h(depth)
    depth *= 255

    return normals

def open_as_array(filename):
    """ Opens image as array """
    img = misc.imread(filename)
    print filename, type(img), img.shape, img.dtype
    array = img.astype(float)
    return array

if __name__ == '__main__':


    intensity = open_as_array("reconstruction-scene8/intensity.png")
    print "intensity", np.min(intensity), np.max(intensity)
    intensity_gt = intensity.copy()
    depth = open_as_array("reconstruction-scene8/depth.png")
    print "depth", np.min(depth), np.max(depth)
    depth_gt = depth.copy() # Ground True

    WIDTH, HEIGHT = intensity.shape

    # Add some noise to depth
    mu, sigma = 0, 5
    depth += sigma * np.random.randn(*depth.shape) + mu

    # Add some noise to intensity
    mu, sigma = 0, 5
    intensity += sigma * np.random.randn(*intensity.shape) + mu

    # Initial value for depth to begin optimization
    depth_x0 = depth.copy()
    depth_x0 = np.clip(depth_x0, 0, 255)
    print np.min(depth_x0), np.max(depth_x0)
    depth_x0 = depth_x0 / 255.0
    median(depth_x0, disk(2), out=depth_x0)
    print np.min(depth_x0), np.max(depth_x0)
    #depth_x0  = depth_x0 * 255
    # Set start image to zero
    #depth_x0 = 0 * depth_x0

    print np.min(depth_x0), np.max(depth_x0)

    # Create OpenGL context
    glcontext = context.Context(width=WIDTH, height=HEIGHT, show_framerate=False)

    glcontext.scene = scenes.exampleScene7()
    glcontext.material = glcontext.scene.meshes[0].material
    glcontext.normalmap_id = glcontext.material.textures['normalmap']
    glcontext.camera = cameras.PerspectiveCamera()

    for title in ("Smooth Gradients", "No Smooth Gradients"):

        energy_total_plot, energy_depth_plot, energy_intensity_plot, energy_normal_plot = [], [], [], []
        TIMESTEP = 0

        if title == "Smooth Gradients":
            smoothnormals = True
        else:
            smoothnormals = False

        # For start energy:
        opt_callback(depth_x0.flatten())


        xk_depth = optimize.fmin_cg(energy, depth_x0.flatten(), epsilon=1, maxiter=5, callback=opt_callback)

        result_depth = xk_depth.reshape((WIDTH, HEIGHT))

        sensor_normal = normals_from_depth_sobel(depth.reshape((WIDTH, HEIGHT)))
        sensor_normal = ((sensor_normal + 1) / 2.) * 255

        start_normal = normals_from_depth_sobel(depth_x0.reshape((WIDTH, HEIGHT)))
        start_normal = ((start_normal + 1) / 2.) * 255
        start_intensity = intensity_from_normals(start_normal, glcontext)

        xk_normal = normals_from_depth_sobel(result_depth.reshape((WIDTH, HEIGHT)))
        xk_normal = ((xk_normal + 1) / 2.) * 255
        xk_intensity = intensity_from_normals(xk_normal, glcontext)

        # Calculate root mean squared error
        rmse_0 = np.sqrt( np.sum( (depth_gt - depth)**2, axis=(0,1) ) / (WIDTH * HEIGHT) )
        rmse_1 = np.sqrt( np.sum( (depth_gt - depth_x0)**2, axis=(0,1) ) / (WIDTH * HEIGHT) )
        rmse_2 = np.sqrt( np.sum( (depth_gt - result_depth)**2, axis=(0,1) ) / (WIDTH * HEIGHT) )
        rmse_3 = np.sqrt( np.sum( (intensity_gt - xk_intensity)**2, axis=(0,1) ) / (WIDTH * HEIGHT) )

        print "RMSE(R_GT, R_noise)", rmse_0
        print "RMSE(R_GT, R_start)", rmse_1
        print "RMSE(R_GT, R_result)", rmse_2
        print "RMSE(I_GT, I_result)", rmse_3

        plt.figure()
        plt.suptitle(title)
        plt.subplot(3, 5, 1)
        plt.title('Ground Depth')
        plt.imshow(depth_gt.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()
        plt.subplot(3, 5, 2)
        plt.title('"Sensor" Depth')
        plt.imshow(depth.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()
        plt.subplot(3, 5, 3)
        plt.title('Start Depth')
        plt.imshow(depth_x0.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()
        plt.subplot(3, 5, 4)
        plt.title('Result Depth')
        plt.imshow(result_depth.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()
        plt.subplot(3, 5, 5)
        plt.title('|Start - Result|')
        plt.imshow(np.abs(result_depth.reshape((WIDTH, HEIGHT)) - depth_x0), cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()

        plt.subplot(3, 5, 6)
        plt.title('Ground Intensity')
        plt.imshow(intensity_gt.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
        plt.colorbar()
        plt.subplot(3, 5, 7)
        plt.title('"Sensor" Int.')
        plt.imshow(intensity.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
        plt.colorbar()
        plt.subplot(3, 5, 8)
        plt.title('Start Int.')
        plt.imshow(start_intensity, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
        plt.colorbar()
        plt.subplot(3, 5, 9)
        plt.title('Result Int.')
        plt.imshow(xk_intensity, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
        plt.colorbar()
        plt.subplot(3, 5, 10)
        plt.title('|Start - Result|')
        plt.imshow(np.abs(xk_intensity.astype("int64") - start_intensity.astype("int64")).astype("uint8"), cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()

        plt.subplot(3, 5, 12)
        plt.title('Normal')
        plt.imshow(sensor_normal.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()
        plt.subplot(3, 5, 13)
        plt.title('start normal')
        plt.imshow(start_normal.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()
        plt.subplot(3, 5, 14)
        plt.title('result normal')
        plt.imshow(xk_normal.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
        plt.colorbar()

        plt.figure()
        plt.title(title)
        plt.plot(energy_normal_plot)
        plt.plot(energy_depth_plot)
        plt.plot(energy_total_plot)
        plt.plot(energy_intensity_plot)
        plt.xlabel("iterations")
        plt.ylabel("energy")
        plt.legend(["normal", "depth", "total", "intensity"])

    plt.show()