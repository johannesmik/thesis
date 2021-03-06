"""

Optimization for the Green Ball

Observed Depth: depth image
Observed Ir: infrared image

Depth Start: Gaussian filtered depth image
Material Start: From the material estimation step

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import optimize
import utils
from scipy.ndimage.filters import gaussian_filter

# Makes experiments repeatable with the same noise
# (If I call the script again I get the same results again)
np.random.seed(0)

depth_sensor_image = Image.open('../assets/optimization/greenball/depth.tiff')
depth_sensor_image = np.asarray(depth_sensor_image, dtype=np.float32) * 4.5

# Small trick to lift the middle depth artifact from 0 just a little bit
depth_sensor_image[178:220, 236:276][np.where(depth_sensor_image[178:220, 236:276] == 0)] = .01

ir_sensor_image = Image.open('../assets/optimization/greenball/ir.tiff')
ir_sensor_image = np.asarray(ir_sensor_image, dtype=np.float32)

k_d = Image.open('../assets/optimization/greenball/kd.tiff')
k_d = np.asarray(k_d, dtype=np.float32)
# k_d = 0.6 * np.ones((424, 512), dtype=np.float32)

# Test with k_d = ir
#k_d = np.clip(ir_sensor_image, 0.0, 0.4)

k_s = Image.open('../assets/optimization/greenball/ks.tiff')
k_s = np.asarray(k_s, dtype=np.float32)
# k_s = 0.1 * np.ones((424, 512), dtype=np.float32)

#n = Image.open('../assets/optimization/greenball/n.tiff')
#n = np.asarray(n, dtype=np.float32)
n = 50 * np.ones((424, 512), dtype=np.float32)

margin = np.zeros((424, 512), dtype=np.float32)
material_image = np.dstack((k_d, k_s, n, margin))

# Diffuse Optimization
optimizer = optimize.Optimizer(depth_sensor_image, ir_sensor_image,
                                        lightingmodel='specular', normalmodel='pca',
                                        depth_variance=0.0001, ir_variance=0.0001,
                                        w_d=10, w_m=5.0,
                                        pca_radius=1.25,
                                        max_iterations=150)

depth_sensor_image_filtered = gaussian_filter(depth_sensor_image, 2)
optimizer.optimize(depth_sensor_image_filtered, material_image)

optimizer.material_image_opt_[:,:,1] = np.clip(optimizer.material_image_opt_[:,:,1], 0, 1)
optimizer.plot_results()

utils.show_image(depth_sensor_image - optimizer.depth_image_opt_, 'depth difference opt')

plt.show()


