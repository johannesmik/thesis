# Linear fit from RGB to IR mean values.

import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import utils
from  scipy.stats import  linregress

import yaml

# Needed to load numpy arrays correctly from yaml
def array_constructor(loader, node):
    return np.array(loader.construct_sequence(node))
yaml.add_constructor('!array', array_constructor)

colors = ['yellow', 'orange', 'red', 'purple', 'green', 'lblue', 'blue', 'brown', 'black', 'white']

path = os.path.expanduser('~/dataset-experiment2')
with open(path + '/results-experiment2-mean-variance.yaml') as f:
    result = yaml.load(f)

color_mean = result['color_mean']
color_variance = result['color_variance']
ir_mean = result['ir_mean']
ir_var = result['ir_variance']

x, y = [], []
for color in colors:
    x.append(color_mean[color]['daylight'])
    y.append(ir_mean[color]['daylight'])

x = np.array(x)[:,:3]
y = np.array(y)

result, residues, rank, s = np.linalg.lstsq(x, y)

print "result:\t", result, "\nresidues:\t", residues, "\nrank:\t", rank, "\ns:\t", s

# Show one image
color_image = utils.read_blob("%s/color_2_daylight.blob" % (path), blob_image_type='color')[:,:,:3]
utils.show_image(color_image, title='Kinect Color')

kinect_ir_image = utils.read_blob("%s/ir_2_daylight.blob" % (path), blob_image_type='ir')
utils.show_image(kinect_ir_image, title='Kinect IR')

result = result.reshape((3, 1))

print "result shape", result.shape, "color_image shape", color_image.shape

ir_image = np.tensordot(result, color_image, (0 ,2))
ir_image = np.clip(ir_image, 0, 1)
ir_image = ir_image.reshape((1080, 1920))
print "ir_image shape", ir_image.shape
utils.show_image(ir_image, title='Calc. Ir Image')


kinect_color_matrix = np.array([ 1.0599465578241038e+03, 0., 9.5488326677588441e+02,
                           0., 1.0539326808799726e+03, 5.2373858291060583e+02,
                           0., 0., 1.  ]).reshape(3, 3)

kinect_color_dist = np.array([ 5.6268441170930321e-02, -7.4199141308694802e-02,
       1.4250797540545752e-03, -1.6951722389720336e-03,
       2.4107681263086548e-02 ])

# Undistort Color image
dst = cv2.undistort(color_image, kinect_color_matrix, kinect_color_dist)
utils.show_image(dst, title='undistorted')
plt.show()

