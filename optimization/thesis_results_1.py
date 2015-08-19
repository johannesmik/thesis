"""

Optimization for the Specular Sphere.

Depth Sensor: Noisy Ground Truth
IR Sensor: Noisy Ground Truth

Depth Start: Noisy Ground Truth, gaussian filtered
Material Start: Results from the material estimation.

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import optimize
import utils

from scipy.ndimage.filters import gaussian_filter

path = '../assets/clustering'
scenename= 'ThreeSpheres'

# Parameters
sigma_d = 0.01
sigma_i = 0.001
w_d = 1.0
w_m = 5

## Choose one:
#lm = 'diffuse'
lm = 'specular'

# Makes experiments repeatable with the same noise
# (If I call the script again I get the same random values)
np.random.seed(0)

depth_ground = Image.open('%s/%s_depth.tiff' % (path, scenename))
depth_ground = np.asarray(depth_ground, dtype=np.float32) * 4.5

depth_observed = depth_ground + sigma_d * np.random.randn(*depth_ground.shape) + 0
depth_observed = depth_observed.astype(np.float32)

infrared_ground = Image.open('%s/%s_ir.tiff' % (path, scenename))
infrared_ground = np.asarray(infrared_ground, dtype=np.float32)

infrared_observed = infrared_ground + sigma_i * np.random.randn(*infrared_ground.shape) + 0
infrared_observed = infrared_observed.astype(np.float32)

k_d = 1 * np.asarray(Image.open('%s/results/%s_kd.tiff' % (path, scenename)), dtype=np.float32)
k_s = 1 * np.asarray(Image.open('%s/results/%s_ks.tiff' % (path, scenename)), dtype=np.float32)
n = np.asarray(Image.open('%s/results/%s_n.tiff' % (path, scenename)), dtype=np.float32)

k_d_ground = np.asarray(Image.open('%s/%s_kd.tiff' % (path, scenename)), dtype=np.float32)
k_s_ground = np.asarray(Image.open('%s/%s_ks.tiff' % (path, scenename)), dtype=np.float32)

margin = np.zeros((424, 512), dtype=np.float32)
material_image = np.dstack((k_d, k_s, n, margin))

# Diffuse/Specular Optimization
optimizer = optimize.Optimizer(depth_observed, infrared_observed,
                                        lightingmodel=lm,
                                        normalmodel='pca',
                                        depth_variance=sigma_d, ir_variance=sigma_i,
                                        w_d=w_d, w_m=w_m, pca_radius=5.0,
                                        max_iterations=150)

depth_sensor_image_filtered = gaussian_filter(depth_ground, 1)
optimizer.optimize(depth_sensor_image_filtered, material_image)

optimizer.plot_results()
optimizer.print_results()

utils.show_image(np.abs(depth_ground - optimizer.depth_image_start_),
                 'depth difference start', vmax=0.5, cmap='jet')
utils.show_image(np.abs(depth_ground - optimizer.depth_image_opt_),
                 'depth difference opt', vmax=0.5, cmap='jet')
utils.show_image(k_d_ground,
                 'kd ground truth')
utils.show_image(k_s_ground,
                 'ks ground truth')

print ('Root mean square error Depth Start' )
print (np.mean(np.sqrt((depth_ground - optimizer.depth_image_start_)**2)))
print ('Root mean square error Depth Opt' )
print (np.mean(np.sqrt((depth_ground - optimizer.depth_image_opt_)**2)))

print ('Root mean square error Infrared Start' )
print (np.mean(np.sqrt((infrared_ground - optimizer.ir_image_start_)**2)))
print ('Root mean square error Infrared Opt' )
print (np.mean(np.sqrt((infrared_ground - optimizer.ir_image_opt_)**2)))

print ('Root mean square error initial k_d' )
print (np.mean(np.sqrt((k_d_ground - optimizer.material_image_start_[:,:,0])**2)))
print ('Root mean square error optimial k_d' )
print (np.mean(np.sqrt((k_d_ground - optimizer.material_image_opt_[:,:,0])**2)))

print ('Root mean square error initial k_s' )
print (np.mean(np.sqrt((k_s_ground - optimizer.material_image_start_[:,:,1])**2)))
print ('Root mean square error optimial k_s' )
print (np.mean(np.sqrt((k_s_ground - optimizer.material_image_opt_[:,:,1])**2)))

plt.show()
