import time
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.patches as mpatches

from sklearn.cluster import MiniBatchKMeans
import scipy.optimize

cluster_cmap = matplotlib.cm.jet

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((h, w, d))
    label_idx = 0

    for i in range(h):
        for j in range(w):
            image[i, j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

def phong_error_func(params, angles, intensities):

    intensities_calculated = phong2(angles, *params)

    error = 0

    # Squared sum error
    error += np.sum((intensities_calculated - intensities)**2)

    # penalize error
    penalizer = 500
    if not (0 <= params[0] <= 1):
        error += penalizer
    if not (0 <= params[1] <= 1) < 0:
        error += penalizer
    if not(0 <= params[0] + params[1] <= 1):
        error += penalizer
    if not (1 <= params[2] <= 1000):
        error += 2 * penalizer

    return error

def phong(angle, k_diffuse, k_specular, n):
    """ Phong model """

    angle = np.deg2rad(angle)
    diffuse_term = k_diffuse * np.cos(angle)
    specular_term = k_specular * np.power(np.cos(2 * angle), n)
    return diffuse_term + specular_term

def phong2(angle, k_diffuse, k_specular, n):
    """ Phong model but clips the cosine to 0, 1 """

    angle = np.deg2rad(angle)
    diffuse_term = k_diffuse * np.clip(np.cos(angle), 0, 1)
    specular_term = k_specular * np.power(np.clip(np.cos(2 * angle), 0, 1), n)
    return diffuse_term + specular_term

def get_codebook(centers, features=None):
    n_clusters = centers.shape[0]

    if not features:
        codebook = np.linspace(0, 1, n_clusters).reshape(n_clusters, 1)
    else:
        codebook = centers[:, features]

    return codebook

def plot_inertias_ratio(inertias, ratios, minmax, threshold=0.8):

    x1 = range(minmax[0], minmax[1] + 1)

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.plot(x1, inertias, 'bo', markersize=6)
    ax1.set_xlabel('number of clusters $k$')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('inertia $I_k$', color='b')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('blue')
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'ratio $\frac{I_{k+1}}{I_{k}}$', color='r')
    ax2.set_ylim((0, 1.05))
    ax2.set_xlim((minmax[0] - 0.5, minmax[1] + 0.5))
    ax2.spines['right'].set_color('red')
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.set_ticks_position('right')
    plt.plot(x1, ratios, 'ro', markersize=6)
    plt.plot([minmax[0] - 0.5, minmax[1] + 0.5], [threshold, threshold], 'r--')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')

def plot_clusters(centers, labels, features=None, title=''):
    # Plot result

    n_clusters = centers.shape[0]

    codebook = get_codebook(centers, features)

    image_show = recreate_image(codebook, labels, 512, 424)
    image_show = image_show.squeeze()

    plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    if title == '':
        ax = plt.axes([0, 0, 1, 1])
    else:
        plt.title(title)
    plt.axis('off')
    plt.imshow(image_show, vmin=np.min(codebook), vmax=np.max(codebook), cmap=cluster_cmap)

def plot_image(image, title='', vmin=0, vmax=255):
    plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    if title == '':
        ax = plt.axes([0, 0, 1, 1])
    else:
        plt.title(title)
    plt.axis('off')
    plot = plt.imshow(image, vmin=vmin, vmax=vmax)
    plot.set_cmap('gray')

def plot_angle_intensities(angles_per_cluster, intensities_per_angle_per_cluster, centers, clusters=None, features=None,
                           title='', legend=False):

    # Todo use clusters
    n_clusters = centers.shape[0]

    codebook = get_codebook(centers, features)

    plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.ylim([0, 1.05])
    plt.xlim([0, 95])
    plt.ylabel('intensity', fontsize=18)
    plt.xlabel('angle $\Theta_h$', fontsize=18)
    if title is not None:
        plt.title(title)

    recs = []
    classes = []

    for i, angles, intensities in zip(range(len(angles_per_cluster)), angles_per_cluster, intensities_per_angle_per_cluster):
        plt.scatter(angles, intensities, c=[codebook[i,0]] * len(angles), vmin=0, vmax=1, cmap=cluster_cmap, label='a')
        recs.append(mpatches.Rectangle((0,0),1,1, fc=cluster_cmap(codebook[i,0])))
        classes.append('Cluster %d' % i)

    if legend:
        plt.legend(recs,classes,loc=1)

def cluster(color_image, depth_image, ir_image, lambdas, threshold=0.8, verbose=False, k_min=1, k_max=20, all_clusters=False):
    """

    :param color_image:
    :param depth_image:
    :param lambdas: 4 values [screen_coordinates, color_image, depth_image, ir_image]
    :param threshold:
    :param verbose:
    :param k_min:
    :param k_max:
    :param all_clusters:
    :return:
    """

    # Todo which line?
    shape_x, shape_y = color_image.shape[:2]
    y, x = np.meshgrid(np.linspace(0, shape_y, shape_y), np.linspace(0, shape_x, shape_x))
    # y, x = np.meshgrid(np.linspace(0, 255, shape_y), np.linspace(0, 255, shape_x))

    r = color_image[:,:,0].flatten()
    g = color_image[:,:,1].flatten()
    b = color_image[:,:,2].flatten()
    d = depth_image.flatten()
    ir = ir_image.flatten()

    lambda1, lambda2, lambda3, lambda4 = lambdas

    if verbose:
        print "Shape_y, Shape_x", shape_y, shape_x
        print "Red:", r.shape, "min/max", r.min(), r.max()
        print "Green:", g.shape, "min/max", g.min(), g.max()
        print "Blue:", b.shape, "min/max", b.min(), b.max()
        print "Depth Shape", d.shape, "min/max", d.min(), d.max()
        print "Ir Shape", ir.shape, 'min/max', ir.min(), ir.max()

    X = np.vstack([lambda1 * x.flatten(), lambda1 * y.flatten(),
               lambda2 * r, lambda2 * g, lambda2 * b,
               lambda3 * d,
               lambda4 * ir])
    X = X.T

    ##############################################################################
    # Compute clustering with MiniBatchKMeans

    last_inertia = 0

    inertias = []
    ratios = []

    ratio = 0

    for i in range(k_min, k_max + 1):
        n_clusters = i
        batch_size = 200

        mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                              n_init=50, init_size=2000, max_no_improvement=20, verbose=0)
        mbk.fit(X)

        next_means_labels = mbk.labels_
        next_means_cluster_centers = mbk.cluster_centers_
        next_inertia = mbk.inertia_

        if last_inertia != 0:
            ratio = next_inertia / last_inertia
            ratios.append(ratio)

        inertias.append(mbk.inertia_)
        last_inertia = next_inertia

        if verbose:
            print '-----'
            print 'Next clusters:', n_clusters
            print "Ratio", ratio
            print "Next Inertia:", next_inertia

        if not all_clusters and ratio > threshold:
            break

        means_cluster_centers = next_means_cluster_centers
        means_labels = next_means_labels

    # Run kmeans one more time to finish the ratios
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k_max + 2, batch_size=batch_size,
                          n_init=100, max_no_improvement=10, verbose=0)
    mbk.fit(X)
    ratio = mbk.inertia_ / last_inertia
    ratios.append(ratio)
    return means_cluster_centers, means_labels, inertias, ratios

def extract_angles(ir_image, centers, labels, binrange=(0, 90), bins=20, normals=None, verbose=False):

    if normals is not None:
        ## TODO calculate normal image
        print "warning: not implemented yet"
        pass

    n_clusters = centers.shape[0]

    v = np.meshgrid(range(-256, 256), range(212, -212, -1))
    v = np.dstack((v[0], v[1], -368.096588 * np.ones((424, 512))))
    v = v / np.linalg.norm(v, axis=2)[:,:,None]
    v = -v

    # Angles between normal and v, dot-product
    angle = np.sum(normals * v, axis=2)
    angle_rad = np.arccos(angle)
    angle_deg = np.rad2deg(angle_rad)

    if verbose:
        print 'vshape', v.shape
        print 'normals shape', normals.shape

    return_angles = []
    return_intensities = []

    # Discretize the labels into bins and take the average
    for i in range(n_clusters):
        n, n_edges = np.histogram(angle_deg.flatten()[np.where(labels.flatten() == i)], bins=bins, range=binrange)

        n_midpoints = np.array([(n_edges[j] + n_edges[j+1]) / 2. for j in range(len(n_edges) - 1)])

        sy, sy_edges = np.histogram(angle_deg.flatten()[np.where(labels.flatten() == i)], bins=bins, range=binrange,
                                    weights=ir_image.flatten()[np.where(labels.flatten() == i)])

        x = n_midpoints[np.where(n > 0)]
        y = sy[np.where(n > 0)] / n[np.where(n > 0)]

        return_angles.append(x)
        return_intensities.append(y)

    return return_angles, return_intensities

def fit_clusters_to_phong(angles_per_cluster, intensities_per_angle_per_cluster):
    popts = []
    for angles, intensities in zip(angles_per_cluster, intensities_per_angle_per_cluster):
        popt = fit_to_phong(angles, intensities)
        popts.append(popt)

    return popts

def fit_to_phong(angles, values):
    angles = np.array(angles)
    values = np.array(values)
    popt = scipy.optimize.fmin(phong_error_func, np.array([0.5, 0.0, 50]), args=(angles, values), disp=False)
    return popt


if __name__ == '__main__':

    pass
