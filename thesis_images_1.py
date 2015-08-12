"""
Images for Section: Material Estimation -- clustering

Will open a lot of figure plots.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import estimate

scene = 'SpecularSphere'

if scene == 'ThreeSpheres':
    color_image = Image.open("assets/ThreeSpheresSpecular_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("assets/ThreeSpheresSpecular_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("assets/ThreeSpheresSpecular_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("assets/ThreeSpheresSpecular_normal.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]
elif scene == 'SpecularSphere':
    color_image = Image.open("assets/SpecularSphere_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("assets/SpecularSphere_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("assets/SpecularSphere_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("assets/SpecularSphere_normal.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]

# Input images
estimate.plot_image(color_image / 255.)
estimate.plot_image(depth_image)

# # Cluster Image
# for i in [1, 2, 3, 4, 5, 50]:
#     centers, labels, inertias, ratios = kmeans.cluster(color_image, depth_image, ir_image,
#                                         lambdas=[0.5, 0.5/3.0, 0.5, 0], threshold=0.95,
#                                         k_min=i, k_max=i, all_clusters=True)
#
#     kmeans.plot_clusters(centers, labels)
#
# # Inertia, Ratio plot
# centers, labels, inertias, ratios = kmeans.cluster(color_image, depth_image, ir_image,
#                                         lambdas=[0.5, 0.5/3., 0.5, 0], threshold=0.95,
#                                         k_min=1, k_max=20, all_clusters=True)
#
# kmeans.plot_inertias_ratio(inertias, ratios, minmax=(1,20), threshold=0.8)

# Cluster Image
centers, labels, inertias, ratios = estimate.cluster(color_image, depth_image, ir_image,
                                            lambdas=[.5, .5/3, .5, 0], threshold=0.8, k_min=1, k_max=20, all_clusters=False)

estimate.plot_clusters(centers, labels)

# Extract
angles_per_cluster, intensities_per_angle = estimate.extract_angles(ir_image, centers, labels, bins=360, binrange=(0, 90), normals=normals)
estimate.plot_angle_intensities(angles_per_cluster, np.array(intensities_per_angle) / 255., centers)

# Binned
angles_per_cluster, intensities_per_angle = estimate.extract_angles(ir_image, centers, labels, bins=20, binrange=(0, 90), normals=normals)
estimate.plot_angle_intensities(angles_per_cluster, np.array(intensities_per_angle) / 255., centers)

# Fit
popts = estimate.fit_clusters_to_phong(angles_per_cluster, np.array(intensities_per_angle) / 255.)

print 'popts',  popts

plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xlabel('angle $\Theta_h$')
plt.ylabel('intensity')
plt.ylim([0, 1.05])
plt.xlim([0, 95])
codebook = estimate.get_codebook(centers)
# Plot the found functions
for i, popt in enumerate(popts):
    angles_cont = np.linspace(0, 90, 120)
    plt.plot(angles_cont, estimate.phong2(angles_cont, *popt), color=estimate.cluster_cmap(codebook[i,0]))

plt.show()