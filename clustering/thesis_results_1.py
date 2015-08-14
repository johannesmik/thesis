"""
Images for the results in Section Material Estimation -- Experiments

Will open a lot of figure plots.

You can choose a scene by setting the 'scene' variable.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import estimate

scene = 'Monkey'

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

# Input images
estimate.plot_image(color_image / 255.)
estimate.plot_image(depth_image)

# Cluster Image
threshold = 0.99
centers, labels, inertias, ratios = estimate.cluster(color_image, depth_image, ir_image,
                                            lambdas=[.5, .5, .5, .0], threshold=threshold, k_min=1, k_max=50, all_clusters=False, verbose=True)

estimate.plot_clusters(centers, labels)

print 'Chosen number of clusters', len(centers)

estimate.plot_inertias_ratio(inertias, ratios, minmax=(1,len(inertias)), threshold=threshold)

# Extract
angles_per_cluster, intensities_per_angle = estimate.extract_angles(ir_image, centers, labels, bins=360, binrange=(0, 80),normals=normals)
estimate.plot_angle_intensities(angles_per_cluster, np.array(intensities_per_angle) / 255., centers, legend=False)

# Binned
angles_per_cluster, intensities_per_angle = estimate.extract_angles(ir_image, centers, labels, bins=30, binrange=(0, 80), normals=normals)
estimate.plot_angle_intensities(angles_per_cluster, np.array(intensities_per_angle) / 255., centers, legend=False)

# Fit
popts = estimate.fit_clusters_to_phong(angles_per_cluster, np.array(intensities_per_angle) / 255.)

print 'popts',  popts

plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xlabel('angle $\Theta_h$', fontsize=18)
plt.ylabel('intensity', fontsize=18)
plt.ylim([0, 1.05])
plt.xlim([0, 95])
codebook = estimate.get_codebook(centers)
# Plot the found functions
for i, popt in enumerate(popts):
    angles_cont = np.linspace(0, 90, 120)
    plt.plot(angles_cont, estimate.phong2(angles_cont, *popt), color=estimate.cluster_cmap(codebook[i,0]))


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


# Plot k_d, k_s, n

# estimate.plot_image(kd_image, '$k_d$ original', vmin=0, vmax=1)
# estimate.plot_image(kd_image_found, '$k_d$ found', vmin=0, vmax=1)
#
# estimate.plot_image(ks_image, '$k_s$ original', vmin=0, vmax=1)
# estimate.plot_image(ks_image_found, '$k_s$ found', vmin=0, vmax=1)
#
# estimate.plot_image(n_image, 'n original', vmin=0, vmax=200)
# estimate.plot_image(n_image_found, 'n found', vmin=0, vmax=200)

# Plots with no title
estimate.plot_image(kd_image, vmin=0, vmax=1)
estimate.plot_image(kd_image_found, vmin=0, vmax=1)

estimate.plot_image(ks_image, vmin=0, vmax=1)
estimate.plot_image(ks_image_found, vmin=0, vmax=1)

estimate.plot_image(n_image, vmin=0, vmax=200)
estimate.plot_image(n_image_found, vmin=0, vmax=200)

# Calculate the RMSD of k_d, k_s, and n
rmse_kd = np.sum(np.sqrt((kd_image - kd_image_found)**2)) / (424 * 512)
rmse_ks = np.sum(np.sqrt((ks_image - ks_image_found)**2)) / (424 * 512)
rmse_n = np.sum(np.sqrt((n_image - n_image_found)**2)) / (424 * 512)

print 'RMSE k_d',  rmse_kd
print 'RMSE k_s', rmse_ks
print 'RMSE n', rmse_n

estimate.plot_image(ir_image)

plt.show()