"""
Images for Section: Material Estimation -- clustering

Will open a lot of figure plots.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import kmeans

scene = 'SpecularSphere'

if scene == 'ThreeSpheres':
    color_image = Image.open("assets/ThreeSpheresSpecular_0000_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("assets/ThreeSpheresSpecular_0000_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("assets/ThreeSpheresSpecular_0000_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("assets/ThreeSpheresSpecular_0000_normal.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]
elif scene == 'SpecularSphere':
    color_image = Image.open("assets/SpecularSphere_0000_color.tiff")
    color_image = np.asarray(color_image, dtype=np.float32)
    depth_image = Image.open("assets/SpecularSphere_0000_depth.tiff")
    depth_image = np.asarray(depth_image, dtype=np.float32) * 255
    ir_image = Image.open("assets/SpecularSphere_0000_ir.tiff")
    ir_image = np.asarray(ir_image, dtype=np.float32) * 255
    normal_image = Image.open("assets/SpecularSphere_0000_normal.tiff")
    normal_image = np.asarray(normal_image, dtype=np.float32)
    normals = (normal_image / 255.) * 2. - 1.
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]

# Input images
kmeans.plot_image(color_image / 255.)
kmeans.plot_image(depth_image)

# Cluster Image
for i in [1, 2, 3, 4, 5, 50]:
    centers, labels, inertias, ratios = kmeans.cluster(color_image, depth_image, ir_image,
                                        lambdas=[0.5, 0.5/3.0, 0.5, 0], threshold=0.95,
                                        k_min=i, k_max=i, all_clusters=True)

    kmeans.plot_clusters(centers, labels)

# Inertia, Ratio plot
centers, labels, inertias, ratios = kmeans.cluster(color_image, depth_image, ir_image,
                                        lambdas=[0.5, 0.5/3., 0.5, 0], threshold=0.95,
                                        k_min=1, k_max=20, all_clusters=True)

kmeans.plot_inertias_ratio(inertias, ratios, minmax=(1,20), threshold=0.8)

print "inertia", inertias
print "ration", ratios

plt.show()