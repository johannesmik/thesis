print(__doc__)

import time
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

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

##############################################################################
# Generate sample data
np.random.seed(0)

color_image = Image.open("assets/ThreeSpheresSpecular_0000_color.tiff")
color_image = np.asarray(color_image, dtype=np.float32)
print color_image.shape
depth_image = Image.open("assets/ThreeSpheresSpecular_0000_depth.tiff")
depth_image = np.asarray(depth_image, dtype=np.float32) * 255


shape_x, shape_y = color_image.shape[:2]
print shape_y, shape_x

lambda1 = 0.5
lambda2 = 0.05
lambda3 = 1.1
y, x = np.meshgrid(np.linspace(0, 255, shape_y), np.linspace(0, 255, shape_x))
r = color_image[:,:,0].flatten()
g = color_image[:,:,1].flatten()
b = color_image[:,:,2].flatten()
d = depth_image.flatten()
print x.flatten().shape
print y.flatten().shape
print "Red:", r.shape, "min/max", r.min(), r.max()
print g.shape
print b.shape
print "Depth Shape", d.shape, "min/max", d.min(), d.max()

X = np.vstack([lambda1 * x.flatten(), lambda1 * y.flatten(),
               lambda2 * r, lambda2 * g, lambda2 * b,
               lambda3 * d])
X = X.T
print X.shape

##############################################################################
# Compute clustering with MiniBatchKMeans

n_clusters = 7
batch_size = 200

mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                      n_init=100, max_no_improvement=10, verbose=0)
t0 = time.time()
mbk.fit(X)
t_mini_batch = time.time() - t0
mbk_means_labels = mbk.labels_
mbk_means_cluster_centers = mbk.cluster_centers_

print "Inertia:", mbk.inertia_

print "centers",  mbk_means_cluster_centers
mbk_means_labels_unique = np.unique(mbk_means_labels)

##############################################################################
# Plot result

codebook = np.linspace(0, 1, n_clusters).reshape(n_clusters, 1)

print "codebook shape", codebook.shape

depth_codebook = mbk_means_cluster_centers[:, 5] / 255.
depth_codebook = depth_codebook.reshape((-1, 1))
print depth_codebook.shape

color_codebook = mbk_means_cluster_centers[:, (2, 3, 4)] / 255.

print codebook

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
image_show = recreate_image(codebook, mbk_means_labels, 512, 424)
print  image_show.shape
image_show = image_show.squeeze()
plt.imshow(image_show)

plt.show()
