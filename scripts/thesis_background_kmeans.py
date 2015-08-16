import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from PIL import Image

depth_image = Image.open("../assets/testimages/sphere-depth-small.png")
depth_image = np.asarray(depth_image, dtype=np.float32)

depth_cut = ((255 - depth_image[25,:]) - 76 )/ 4.

print depth_cut
x = np.linspace(1, 50, 50)

X = np.vstack([x, depth_cut])
X = X.T

initial_clusters = np.array([X[15], X[25], X[35], X[40], X[45]])

for i in [0, 1, 2, 5, 20]:

    if i == 0:
        nn = NearestNeighbors(1, 1)
        nn.fit(initial_clusters)
        centers = initial_clusters
        dist, labels = nn.kneighbors(X)
    else:
        mbk = KMeans(max_iter=i, n_clusters=5,  init=initial_clusters, verbose=0, n_init=1)
        mbk.fit(X)
        centers = mbk.cluster_centers_
        labels = mbk.labels_


    fig, ax = plt.subplots(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.xlim([0, 51])
    plt.ylim([-1, 50])
    ax.set_xlabel('X', fontsize=22)
    ax.set_ylabel('Z', fontsize=22)
    plt.title('Iteration %d ' % i, fontsize=24)
    ax.scatter(x, depth_cut, c=labels / 4., marker='o', alpha=1, s=90, edgecolors='none', cmap=plt.get_cmap("rainbow"))
    ax.scatter(centers[:,0], centers[:,1], c=np.linspace(0, 1, 5), s=550, marker='*', alpha=1, cmap=plt.get_cmap("rainbow"))

plt.show()