print(__doc__)

import time
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import scipy.optimize

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

def blinn2(angle, k_diffuse, k_specular, n):

    #if k_diffuse + k_specular > 1.0 or k_diffuse < 0 or k_specular < 0:
    #    return 10000

    angle = np.deg2rad(angle)
    print np.min(np.cos(2 * angle)), np.max(np.cos(2 * angle))
    diffuse_term = k_diffuse * np.clip(np.cos(angle), 0, 1)
    specular_term = k_specular * np.power(np.clip(np.cos(2 * angle), 0, 1), n)
    return diffuse_term + specular_term

def blinn(angle, k_diffuse, k_specular, n):

    #if k_diffuse + k_specular > 1.0 or k_diffuse < 0 or k_specular < 0:
    #    return 10000

    angle = np.deg2rad(angle)
    print np.min(np.cos(2 * angle)), np.max(np.cos(2 * angle))
    diffuse_term = k_diffuse * np.cos(angle)
    specular_term = k_specular * np.power(np.cos(2 * angle), n)
    return diffuse_term + specular_term

def fit_to_blinn(angles, values):
    popt, pcov = scipy.optimize.curve_fit(blinn, angles.astype(np.float64), values, p0=np.array([0.7, 0.2, 50], dtype=np.float64))
    return popt

##############################################################################
# Generate sample data
np.random.seed(0)

color_image = Image.open("assets/ThreeSpheresSpecular_0000_color.tiff")
color_image = Image.open("assets/SpecularSphere_0000_color.tiff")
color_image = np.asarray(color_image, dtype=np.float32)
depth_image = Image.open("assets/ThreeSpheresSpecular_0000_depth.tiff")
depth_image = Image.open("assets/SpecularSphere_0000_depth.tiff")
depth_image = np.asarray(depth_image, dtype=np.float32) * 255
ir_image = Image.open("assets/ThreeSpheresSpecular_0000_ir.tiff")
ir_image = Image.open("assets/SpecularSphere_0000_ir.tiff")
ir_image = np.asarray(ir_image, dtype=np.float32) * 255
normal_image = Image.open("assets/ThreeSpheresSpecular_0000_normal.tiff")
normal_image = Image.open("assets/SpecularSphere_0000_normal.tiff")
normal_image = np.asarray(normal_image, dtype=np.float32)
normals = (normal_image / 255.) * 2. - 1.
normals = normals / np.linalg.norm(normals, axis=2)[:,:,None]

shape_x, shape_y = color_image.shape[:2]
print shape_y, shape_x

lambda1 = 0.5
lambda2 = 0.5 / 3.
lambda3 = 0.5
lambda4 = .0
y, x = np.meshgrid(np.linspace(0, 255, shape_y), np.linspace(0, 255, shape_x))
r = color_image[:,:,0].flatten()
g = color_image[:,:,1].flatten()
b = color_image[:,:,2].flatten()
d = depth_image.flatten()
ir = ir_image.flatten()
print x.flatten().shape
print y.flatten().shape
print "Red:", r.shape, "min/max", r.min(), r.max()
print g.shape
print b.shape
print "Depth Shape", d.shape, "min/max", d.min(), d.max()
print "Ir Shape", ir.shape, 'min/max', ir.min(), ir.max()

X = np.vstack([lambda1 * x.flatten(), lambda1 * y.flatten(),
               lambda2 * r, lambda2 * g, lambda2 * b,
               lambda3 * d,
               lambda4 * ir])
X = X.T
print X.shape

##############################################################################
# Compute clustering with MiniBatchKMeans

last_inertia = 4250476262

x1 = range(1, 20)
inertias = []
ratios = []

for i in range(1, 20):
    print '-----'
    print 'clusters:', i
    n_clusters = i
    batch_size = 200

    mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                          n_init=100, max_no_improvement=10, verbose=0)
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0
    mbk_means_labels = mbk.labels_
    mbk_means_cluster_centers = mbk.cluster_centers_

    print "Inertia:", mbk.inertia_
    ratio = mbk.inertia_ / last_inertia
    print "Ratio", ratio
    last_inertia = mbk.inertia_

    inertias.append(mbk.inertia_)
    ratios.append(ratio)

    #print "centers",  mbk_means_cluster_centers
    mbk_means_labels_unique = np.unique(mbk_means_labels)

    ##############################################################################
    # Plot result

    codebook = np.linspace(0, 1, n_clusters).reshape(n_clusters, 1)

    #print "codebook shape", codebook.shape

    depth_codebook = mbk_means_cluster_centers[:, 5] / 255.
    depth_codebook = depth_codebook.reshape((-1, 1))
    #print depth_codebook.shape

    color_codebook = mbk_means_cluster_centers[:, (2, 3, 4)] / 255.

    print codebook

    image_show = recreate_image(codebook, mbk_means_labels, 512, 424)
    print image_show[200, (50, 250, 480)]
    #print  image_show.shape
    image_show = image_show.squeeze()


    if ratio > 0.7:
        break

# fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
# plt.plot(x1, inertias, 'bo', markersize=6)
# ax1.set_xlabel('number of clusters $k$')
# # Make the y-axis label and tick labels match the line color.
# ax1.set_ylabel('inertia $i_k$', color='b')
# for tl in ax1.get_yticklabels():
#     tl.set_color('b')
# ax2 = ax1.twinx()
# ax2.set_ylabel(r'ratio $\frac{i_k}{i_{k-1}}$', color='r')
# plt.plot(x1, ratios, 'ro', markersize=6)
# plt.plot([0, 20], [0.8, 0.8], 'r--')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')

plt.figure()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (%d colors, K-Means)' % n_clusters)
plt.imshow(image_show)

v = np.meshgrid(range(-256, 256), range(212, -212, -1))
print 'vshape 1', v[0].shape, v[1].shape
v = np.dstack((v[0], v[1], -368.096588* np.ones((424, 512))))
print 'vshape', v.shape
v = v / np.linalg.norm(v, axis=2)[:,:,None]
v = -v
print "vshape 3", v.shape

# Angles between normal and v, dot-product
print 'normals shape', normals.shape
angle = np.sum(normals * v, axis=2)
angle_rad = np.arccos(angle)
angle_deg = np.rad2deg(angle_rad)

t_image = Image.open("assets/SpecularSphere_0000_n.tiff")
t_image = np.asarray(depth_image, dtype=np.float32)
plt.figure()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
#plt.title('angle values')
plot = plt.imshow(t_image)
#plt.colorbar()
plot.set_cmap('gray')

plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
#plt.title('scatter plot')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
#plt.ylabel('intensity')
#plt.xlabel(r'$\varphi$')
ax.set_xlim([0,90])
ax.set_ylim([0,1.1])

plt.scatter(angle_deg.flatten()[np.where(mbk_means_labels.flatten() == 0)][::1250], ir_image.flatten()[np.where(mbk_means_labels.flatten() == 0)][::1250] / 255., c='b', s=75)
plt.scatter(angle_deg.flatten()[np.where(mbk_means_labels.flatten() == 1)][::700], ir_image.flatten()[np.where(mbk_means_labels.flatten() == 1)][::700] / 255., c='r', s=75)


plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
#plt.ylabel('intensity')
#plt.xlabel(r'$\varphi$')
ax.set_xlim([0,90])
ax.set_ylim([0,1.1])

print "fit to blinn"
for i in range(n_clusters):


    angles = angle_deg.flatten()[np.where(mbk_means_labels.flatten() == i)]
    intensities = ir_image.flatten()[np.where(mbk_means_labels.flatten() == i)] / 255.

    x = np.linspace(0, 40, 50)

    sortarg = np.argsort(angles)
    y = np.interp(x, angles[sortarg], intensities[sortarg])

    print np.all(np.diff(angles[sortarg]) > 0)

    popt = fit_to_blinn(x, y)
    print i, popt

    if i == 0:
        color = 'b'
    elif i == 1:
        color = 'r'
    else:
        color = 'r'

    angles_cont = np.linspace(0, 90, 120)
    print angles_cont.shape
    print blinn2(angles_cont, *popt).shape
    plt.plot(angles_cont, blinn2(angles_cont, *popt), color)
    #plt.scatter(x, y, c='r')
    #plt.scatter(angle_deg.flatten()[np.where(mbk_means_labels.flatten() == i)],
     #           ir_image.flatten()[np.where(mbk_means_labels.flatten() == i)] / 255.,
     #           c='b')




plt.show()
