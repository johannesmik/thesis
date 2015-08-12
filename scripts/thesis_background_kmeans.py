__author__ = 'johannes'

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from PIL import Image


depth_image = Image.open("../assets/testimages/sphere-depth-small.png")
depth_image = np.asarray(depth_image, dtype=np.float32)

shape_x, shape_y = depth_image.shape[:2]
y, x = np.meshgrid(np.linspace(0, shape_y, shape_y), np.linspace(0, shape_x, shape_x))

X = np.vstack([x.flatten(), y.flatten(), depth_image.flatten()])
X = X.T

initial_clusters = np.array([X[25 * 50 + 15], X[25 * 50 + 35], X[15 * 50 + 25], X[35 * 50 + 25], X[25 * 50 + 25]])

print initial_clusters
print "X", X[25 * 50 + 25]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, - depth_image, c='r', marker='.', alpha=0.25)
ax.scatter(initial_clusters[:,0], initial_clusters[:,1], - initial_clusters[:,2] + 0.05, s=30, alpha=1)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('- Z')

for i in [1, 2, 5, 20]:

    mbk = KMeans(max_iter=i, n_clusters=5,  init=initial_clusters, verbose=0, n_init=1)
    mbk.fit(X)

    centers = mbk.cluster_centers_

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, - depth_image, c='r', marker='.', alpha=0.25)
    ax.scatter(centers[:,0], centers[:,1], - centers[:,2] + 0.05, s=30, alpha=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('- Z')
    plt.title('Iteration %d ' % i)

plt.show()