"""
Experiment 3 was about finding the variance of the intensity and depth using static scenes

Each scene consists of 10 RGB-D-IR images.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle

import utils

def plot_variance_histogram(mean, variance):

    mean = mean.flatten()
    variance = variance.flatten()

    plt.figure()
    plt.title('Histogram!')

    h, bedges = np.histogram(mean, 200, normed=True)

    weights = np.zeros_like(mean)

    for i in range(200):
        if i == 0:
            weights[np.where((0 <= mean) & (mean < bedges[i]))] = 1 / (1 + h[i])
        else:
            weights[np.where((bedges[i-1] <= mean) & (mean < bedges[i]))] = 1 / (1 + h[i])

    H, xedges, yedges = np.histogram2d(mean, variance,
                                       bins=200,
                                       range=[[0.001, 0.6], [0, 0.0001]],
                                       normed=True,
                                       weights=weights)

    print H
    X, Y = np.meshgrid(xedges, yedges)
    plt.pcolormesh(X, Y, H.T)
    plt.colorbar()
    #plt.set_aspect('equal')


scenes = ['data-livingroom', 'data-kitchen', 'data-balcony']

scenes = ['data-livingroom']

for scene in scenes:

    print "Scene: ", scene

    ir_array = None
    depth_array = None

    path = os.path.expanduser('~/workspace/dataset-experiment3/%s' % scene)

    for i in range(10):
        ir = utils.read_blob("%s/ir_%d.blob" % (path, i), blob_image_type='ir')
        depth = utils.read_blob("%s/depth_%d.blob" % (path, i), blob_image_type='depth')

        # Stack on top of each other
        if ir_array != None:
            ir_array = np.dstack((ir_array, ir))
            depth_array = np.dstack((depth_array, depth))
        else:
            ir_array = ir
            depth_array = depth

    ir_mean = np.mean(ir_array, axis=2)
    ir_variance = np.var(ir_array, axis=2)
    depth_variance = np.var(depth_array, axis=2)

    print ir_array[158, 237, :], ir_mean[158, 237],  ir_variance[158, 237]
    print ir_array[187, 295, :]
    print ir_array[225, 110, :]

    utils.show_image(ir_mean, 'ir mean scene %s' % scene)
    utils.show_image(ir_variance, 'ir variance scene %s' % scene, minmax=(0, ir_variance.max()))
    utils.show_image(depth_variance, 'depth variance scene %s' % scene, minmax=(0, depth_variance.max()))

    plt.figure()
    plt.scatter(ir_mean.flatten(), ir_variance.flatten(), c='red', s=1, alpha=0.3, edgecolors='none')

    a = shuffle(ir_mean.flatten(), random_state=0)[:10000]
    b = shuffle(ir_variance.flatten(), random_state=0)[:10000]
    print "a size", a.size
    # plot_variance_histogram(ir_mean[np.where(0 < ir_mean)], ir_variance[np.where(0 < ir_mean)])
    plot_variance_histogram(ir_mean, ir_variance)

    plt.figure()
    plt.title('histogram test')
    plt.hist(ir_mean.flatten(), 30, normed=1, facecolor='green', alpha=0.5)

    # Load color image
    color = utils.read_blob("%s/color_1.blob" % path, blob_image_type='color')
    utils.show_image(color, 'color scene %s' % scene)

plt.show()