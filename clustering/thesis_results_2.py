__author__ = 'johannes'

"""
Images for the results in Section Material Estimation -- Experiments

Those is a try out for the k-noise-RMSE_d error plot

Will open a lot of figure plots.

You can choose a scene by setting the 'scene' variable.
"""

# TODO write for the different scenes

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import estimate

scene = 'ThreeSpheres'

if scene == 'ThreeSpheres':
    color_image = Image.open("../assets/clustering/ThreeSpheres_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("../assets/clustering/ThreeSpheres_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("../assets/clustering/ThreeSpheres_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("../assets/clustering/ThreeSpheres_normal_pca.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]
    kd_image = Image.open("../assets/clustering/ThreeSpheres_kd.tiff")
    kd_image = np.asarray(kd_image, dtype=np.float32)
    ks_image = Image.open("../assets/clustering/ThreeSpheres_ks.tiff")
    ks_image = np.asarray(ks_image, dtype=np.float32)
    n_image = Image.open("../assets/clustering/ThreeSpheres_n.tiff")
    n_image = np.asarray(n_image, dtype=np.float32)

elif scene == 'ThreeBoxes':
    color_image = Image.open("../assets/clustering/ThreeBoxes_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("../assets/clustering/ThreeBoxes_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("../assets/clustering/ThreeBoxes_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("../assets/clustering/ThreeBoxes_normal_pca.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]
    kd_image = Image.open("../assets/clustering/ThreeBoxes_kd.tiff")
    kd_image = np.asarray(kd_image, dtype=np.float32)
    ks_image = Image.open("../assets/clustering/ThreeBoxes_ks.tiff")
    ks_image = np.asarray(ks_image, dtype=np.float32)
    n_image = Image.open("../assets/clustering/ThreeBoxes_n.tiff")
    n_image = np.asarray(n_image, dtype=np.float32)

elif scene == 'SpecularSphere':
    color_image = Image.open("../assets/clustering/SpecularSphere_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("../assets/clustering/SpecularSphere_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("../assets/clustering/SpecularSphere_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("../assets/clustering/SpecularSphere_normal_pca.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]
    kd_image = Image.open("../assets/clustering/SpecularSphere_kd.tiff")
    kd_image = np.asarray(kd_image, dtype=np.float32)
    ks_image = Image.open("../assets/clustering/SpecularSphere_ks.tiff")
    ks_image = np.asarray(ks_image, dtype=np.float32)
    n_image = Image.open("../assets/clustering/SpecularSphere_n.tiff")
    n_image = np.asarray(n_image, dtype=np.float32)

elif scene == 'Monkey':
    color_image = Image.open("../assets/clustering/MonkeySuzanne_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("../assets/clustering/MonkeySuzanne_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("../assets/clustering/MonkeySuzanne_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("../assets/clustering/MonkeySuzanne_normal_pca.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]
    kd_image = Image.open("../assets/clustering/MonkeySuzanne_kd.tiff")
    kd_image = np.asarray(kd_image, dtype=np.float32)
    ks_image = Image.open("../assets/clustering/MonkeySuzanne_ks.tiff")
    ks_image = np.asarray(ks_image, dtype=np.float32)
    n_image = Image.open("../assets/clustering/MonkeySuzanne_n.tiff")
    n_image = np.asarray(n_image, dtype=np.float32)


datapoints = []
n_clusters = range(1, 101, 1)
variances = [0, 5, 10, 15, 20, 25, 30, 35, 40]

threshold = .99

for n_cluster in n_clusters:
    for variance in variances:
        centers, labels, inertias, ratios = estimate.cluster(color_image, depth_image, ir_image,
                                            lambdas=[.5, .5, .5, .0], k_min=n_cluster, k_max=n_cluster, all_clusters=False, verbose=True)

        # Extract
        angles_per_cluster, intensities_per_angle = estimate.extract_angles(ir_image, centers, labels, bins=360, binrange=(0, 80),normals=normals)

        # Binned
        angles_per_cluster, intensities_per_angle = estimate.extract_angles(ir_image, centers, labels, bins=30, binrange=(0, 80), normals=normals)

        # Fit
        popts = estimate.fit_clusters_to_phong(angles_per_cluster, np.array(intensities_per_angle) / 255.)

        kd_image_found = np.zeros((424 * 512))
        ks_image_found = np.zeros((424 * 512))
        n_image_found = np.zeros((424 * 512))

        for i in range(np.max(labels)+1):
             kd_image_found[np.where(labels == i)] = popts[i][0]
             ks_image_found[np.where(labels == i)] = popts[i][1]
             n_image_found[np.where(labels == i)] = popts[i][2]

        kd_image_found = kd_image_found.reshape((424, 512))
        ks_image_found = ks_image_found.reshape((424, 512))
        n_image_found = n_image_found.reshape((424, 512))

        # Calculate the RMSD of k_d, k_s, and n
        rmse_kd = np.sum(np.sqrt((kd_image - kd_image_found)**2)) / (424 * 512)
        rmse_ks = np.sum(np.sqrt((ks_image - ks_image_found)**2)) / (424 * 512)
        rmse_n = np.sum(np.sqrt((n_image - n_image_found)**2)) / (424 * 512)

        print 'RMSE k_d',  rmse_kd
        print 'RMSE k_s', rmse_ks
        print 'RMSE n', rmse_n
        print 'ration', ratios[0]

        datapoints.append([n_cluster, variance, rmse_kd, rmse_ks, rmse_n, ratios[0]])

        break

datapoints = np.array(datapoints)

fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax.set_xlabel('number of clusters $k$')
ax.set_ylabel('$RMSE_{kd}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.plot(datapoints[:, 0], datapoints[:, 2], 'r-')

fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax.set_xlabel('number of clusters $k$')
ax.set_ylabel('$RMSE_{ks}$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.plot(datapoints[:, 0], datapoints[:, 3], 'b-')

fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax.set_xlabel('number of clusters $k$')
ax.set_ylabel('RMSE_{n}')
ax.plot(datapoints[:, 0], datapoints[:, 4], 'b-')

fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax.set_xlabel('number of clusters $k$')
ax.set_ylabel(r'ratio $\frac{I_{k+1}}{I_{k}}$')
ax.plot(datapoints[:, 0], datapoints[:, 5], 'b-')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, - depth_image, c='r', marker='.', alpha=0.25)
# ax.scatter(initial_clusters[:,0], initial_clusters[:,1], - initial_clusters[:,2] + 0.05, s=30, alpha=1)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('- Z')

# Cluster Image

plt.show()