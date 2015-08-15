"""

Optimization for the Monkey,

Depth Sensor: Noisy Ground Truth
IR Sensor: Noisy Ground Truth

Depth Start: Noisy Ground Truth
Material Start: Ground Truth

"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import optimize
import utils

path = '../assets/clustering'
scenename= 'MonkeySuzanne'

# Makes experiments repeatable with the same noise
# (If I call the script again I get the same results again)
np.random.seed(0)

depth_sensor_image = Image.open('%s/%s_depth.tiff' % (path, scenename))
depth_sensor_image = np.asarray(depth_sensor_image, dtype=np.float32) * 4.5

mu, sigma = 0, 0.003
depth_sensor_image_noise = depth_sensor_image + sigma * np.random.randn(*depth_sensor_image.shape) + mu
depth_sensor_image_noise = depth_sensor_image_noise.astype(np.float32)

mu, sigma = 0, 0.003
depth_sensor_image_noise2 = depth_sensor_image + sigma * np.random.randn(*depth_sensor_image.shape) + mu
depth_sensor_image_noise2 = depth_sensor_image_noise2.astype(np.float32)

ir_sensor_image = Image.open('%s/%s_ir.tiff' % (path, scenename))
ir_sensor_image = np.asarray(ir_sensor_image, dtype=np.float32)

mu, sigma = 0, 0.01
ir_sensor_image_noise = ir_sensor_image + sigma * np.random.randn(*ir_sensor_image.shape) + mu
ir_sensor_image_noise = ir_sensor_image_noise.astype(np.float32)

k_d = np.asarray(Image.open('%s/%s_kd.tiff' % (path, scenename)), dtype=np.float32)
k_s = np.asarray(Image.open('%s/%s_ks.tiff' % (path, scenename)), dtype=np.float32)
n = np.asarray(Image.open('%s/%s_n.tiff' % (path, scenename)), dtype=np.float32)
margin = np.zeros((424, 512), dtype=np.float32)
material_image = np.dstack((k_d, k_s, n, margin))

optimizer = optimize.Optimizer(depth_sensor_image_noise, ir_sensor_image_noise, lightingmodel='specular')
optimizer.optimize(depth_sensor_image_noise2, material_image)

optimizer.plot_results()

# Save depth start and depth end
utils.save_bw_image(optimizer.depth_image_start_, 'depth_start.tiff')
utils.save_bw_image(optimizer.depth_image_opt_, 'depth_opt.tiff')

utils.show_image(depth_sensor_image - optimizer.depth_image_start_, 'depth difference start')
utils.show_image(depth_sensor_image - optimizer.depth_image_opt_, 'depth difference opt')

plt.show()


