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

def energy(depth, depth_o, intensity_o, glcontext):

    depth = depth.reshape((WIDTH, HEIGHT))
    #depth = depth.clip(0, 255)

    global TIMESTEP
    TIMESTEP += 1

    normal = normals_from_depth_pcl(depth, ksearch=15)
    #normal = normals_from_depth_sobel(depth)
    normal = ((normal + 1) / 2.) * 255

    intensity = intensity_from_normals(normal, glcontext)

    energy = 0
    energy += 5
    energy += np.sum((intensity - intensity_o) ** 2, axis=(0, 1))

    #print energy

    if TIMESTEP == 1 or TIMESTEP == 400:
        show_info(depth, depth_o, intensity, intensity_o, normal)

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
    depth /= 255
    normals = np.ones((HEIGHT, WIDTH, 4))
    normals[:,:,0] = sobel_v(depth)
    normals[:,:,1] = sobel_h(depth)
    depth *= 255

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


    intensity = open_as_array("reconstruction-scene4/intensity.png")
    depth = open_as_array("reconstruction-scene4/depth.png")
    depth_gt = depth.copy() # Ground True

    WIDTH, HEIGHT = intensity.shape
    TIMESTEP = 0


    # Add some noise to depth
    mu, sigma = 0, 5
    depth += sigma * np.random.randn(*depth.shape) + mu

    # Add some noise to intensity
    mu, sigma = 0, 5
    intensity += sigma * np.random.randn(*intensity.shape) + mu

    # Initial value for depth to begin optimization
    depth_x0 = depth.copy()


    # Create OpenGL context
    c = context.Context(width=WIDTH, height=HEIGHT)

    c.scene = scenes.exampleScene3()
    c.material = c.scene.meshes[0].material
    c.normalmap_id = c.material.textures['normalmap']
    c.camera = cameras.PerspectiveCamera()

    xk = optimize.fmin_cg(energy, depth_x0.flatten(), args=(depth, intensity, c), epsilon=1, maxiter=30)

    # Calculate root mean squared error
    rmse_1 = np.sqrt( np.sum( (depth_gt - depth_x0)**2, axis=(0,1) ) / (WIDTH * HEIGHT) )
    rmse_2 = np.sqrt( np.sum( (depth_gt - xk.reshape((WIDTH, HEIGHT)))**2, axis=(0,1) ) / (WIDTH * HEIGHT) )

    print "RMSE(R_GT, R_0)", rmse_1
    print "RMSE(R_GT, R)", rmse_2

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.title('Start Depth')
    plt.imshow(depth_x0.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.subplot(2, 2, 2)
    plt.title('Result Depth')
    plt.imshow(xk.reshape((WIDTH, HEIGHT)), cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.subplot(2, 2, 3)
    plt.title('Difference')
    plt.imshow(np.abs(xk.reshape((WIDTH, HEIGHT)) - depth_x0), cmap=plt.cm.gray, interpolation='nearest')
    plt.subplot(2, 2, 4)
    plt.title('Result Intensity')
    normal = normals_from_depth_pcl(xk.reshape((WIDTH, HEIGHT)), ksearch=15)
    normal = ((normal + 1) / 2.) * 255
    plt.imshow(intensity_from_normals(normal, c), cmap=plt.cm.gray, interpolation='nearest')
    plt.show()
