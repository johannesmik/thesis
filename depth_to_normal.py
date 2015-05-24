__author__ = 'johannes'

import numpy as np
import pcl
from scipy import misc
import cProfile
from skimage.filters import sobel, sobel_h, sobel_v, median, scharr_h, scharr_v
import matplotlib.pyplot as plt

import reconstruction, context

' Here I want to test different methods to calculate the normal map from a depth map '

def open_as_array(filename):
    """ Opens image as array """
    img = misc.imread(filename)
    array = img.astype(float)
    return array

def normals_from_depth_sobel(depth):
    ' Calculate the normals with sobel '
    # TODO not correct yet
    HEIGHT, WIDTH = depth.shape
    normals = 255 * np.ones((HEIGHT, WIDTH, 4))
    normals[:,:,0] = sobel_v(depth)
    normals[:,:,1] = sobel_h(depth)

    return normals

def normals_from_depth_pcl(depth, ksearch=10):
    ''' Calculate normals with pcl '''

    # Camera parameters
    n, f = 1.0, 20.0
    l, r = -1.0, 1.0
    t, b = 1.0, -1.0

    HEIGHT, WIDTH = depth.shape

    # Warp into 3D Space
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
    normals[:,:,:3] = pclnormals.reshape(HEIGHT, WIDTH, 3)
    return normals

def normals_from_depth_sobel(depth):
    ' Calculate the normals with sobel '
    # TODO not correct yet
    HEIGHT, WIDTH = depth.shape
    normals = np.ones((HEIGHT, WIDTH, 4))

    normals[:,:,0] = 0.1 * (1 + depth) * sobel_v(depth)
    normals[:,:,1] = 0.1 * (1 + depth) * -sobel_h(depth)
    normals[:,:,2] = (1 + depth)

    # Normalize such that the length is one, and the values range between -1 and 1
    normals = normals / 255.
    normals = normals / np.linalg.norm(normals[:,:,:3], axis=2)[:,:,np.newaxis]
    #normals[:,:,3] = 1
    return normals

def normals_from_depth_sobel2(depth):
    ' Calculate the normals with sobel '
    # TODO not correct yet
    HEIGHT, WIDTH = depth.shape
    normals = 255 * np.ones((HEIGHT, WIDTH, 4))
    normals[:,:,0] = depth * sobel_v(depth)
    normals[:,:,1] = depth * -sobel_h(depth)
    normals[:,:,2] = 1

    # Normalize such that the length is one, and the values range between -1 and 1
    normals = normals / 255.
    normals[:,:,:3] = normals[:,:,:3] / np.linalg.norm(normals[:,:,:3], axis=2)[:,:,np.newaxis]
    normals[:,:,3] = 1
    return normals

def normals_from_depth_sobel3(depth):
    HEIGHT, WIDTH = depth.shape
    normals = np.cross

if __name__ == '__main__':

    # Sphere from Image

    depth1 = open_as_array("images/reconstruction-scene8/depth.png")
    normal1 = open_as_array("images/reconstruction-scene8/normal.png")

    normal_pcl = normals_from_depth_pcl(depth1)
    normal_pcl = ((normal_pcl + 1) / 2.) * 255

    normal_sobel = normals_from_depth_sobel2(depth1)
    normal_sobel = ((normal_sobel + 1) / 2.) * 255

    plt.subplot(3,3,1)
    plt.title("Depthmap")
    plt.imshow(depth1, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.colorbar()

    plt.subplot(3,3,2)
    plt.title("Normal PCL")
    plt.imshow(normal_pcl.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
    plt.colorbar()

    plt.subplot(3,3,3)
    plt.title("Normal (sobel)")
    plt.imshow(normal_sobel.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
    plt.colorbar()

    # Sphere from OpenGL context

    width, height = 600, 512
    glcontext = context.Context(width=width, height=height, show_framerate=True)

    intensity2, depth2 = reconstruction.sphere(glcontext)

    cProfile.run("normal_pcl2 = normals_from_depth_pcl(depth2)")
    normal_pcl2 = ((normal_pcl2 + 1) / 2.) * 255

    cProfile.run("normal_sobel2 = normals_from_depth_sobel2(depth2)")
    normal_sobel2 = ((normal_sobel2 + 1) / 2.) * 255

    plt.subplot(3,3,4)
    plt.imshow(depth2, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.colorbar()

    plt.subplot(3,3,5)
    plt.imshow(normal_pcl2.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
    plt.colorbar()

    plt.subplot(3,3,6)
    plt.imshow(normal_sobel2.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
    plt.colorbar()


    # Squares

    intensity3, depth3 = reconstruction.squares(glcontext)

    normal_pcl3 = normals_from_depth_pcl(depth3)
    normal_pcl3 = ((normal_pcl3 + 1) / 2.) * 255

    normal_sobel3 = normals_from_depth_sobel2(depth3)
    normal_sobel3 = ((normal_sobel3 + 1) / 2.) * 255

    plt.subplot(3,3,7)
    plt.imshow(depth3, cmap=plt.cm.gray, interpolation='nearest', vmin=0, vmax=255)
    plt.colorbar()

    plt.subplot(3,3,8)
    plt.imshow(normal_pcl3.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
    plt.colorbar()

    plt.subplot(3,3,9)
    plt.imshow(normal_sobel3.astype('uint8')[:,:,:3], cmap=plt.cm.gray, interpolation='nearest')
    plt.colorbar()

    plt.savefig('results/depth_to_normal.png', dpi=300)
    plt.show()
