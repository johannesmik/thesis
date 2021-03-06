__author__ = 'johannes'
"""
Images for the results in Section Material Estimation -- Experiments

Will open a lot of figure plots.

You can choose a scene by setting the 'scene' variable.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import estimate

#### Params

save_images = False

####

color_image = Image.open("../assets/optimization/greenball/color_transformed.tiff")
color_image = np.asarray(color_image, dtype=np.float32)
depth_image = Image.open("../assets/optimization/greenball/depth.tiff")
depth_image = np.asarray(depth_image, dtype=np.float32) * 255
ir_image = Image.open("../assets/optimization/greenball/ir.tiff")
ir_image = np.asarray(ir_image, dtype=np.float32) * 255
normal_image = Image.open("../assets/optimization/greenball/normal_pca.tiff")
normal_image = np.asarray(normal_image, dtype=np.float32)
normals = (normal_image / 255.) * 2. - 1.
normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]

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

kd_image_found = np.clip(kd_image_found.reshape((424, 512)), 0, 1)
ks_image_found = np.clip(ks_image_found.reshape((424, 512)), 0, 1)
n_image_found = np.clip(n_image_found.reshape((424, 512)), 1, 200)

# Plots with no title
estimate.plot_image(normals)

estimate.plot_image(kd_image_found, vmin=0, vmax=1)

estimate.plot_image(ks_image_found, vmin=0, vmax=1)

estimate.plot_image(n_image_found, vmin=0, vmax=200)

# Save images
if save_images:
    images = [kd_image_found, ks_image_found, n_image_found]
    names = ['kd', 'ks', 'n']
    for image, name in zip(images, names):
        image = image.astype(np.float32)
        img = Image.fromarray(image, mode='F')
        with open('../assets/optimization/greenball/%s.tiff' % name, 'w') as f:
            img.save(f)

estimate.plot_image(ir_image)

plt.show()