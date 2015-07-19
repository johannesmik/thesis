import os
import math
import numpy as np
import matplotlib.pyplot as plt
import yaml

import utils

color_arrays = []
depth_arrays = []
ir_arrays = []

colors = ['yellow', 'orange', 'red', 'purple', 'green', 'lblue', 'blue', 'brown', 'black', 'white']

path = '~/dataset-experiment2'
path = os.path.expanduser(path)

for i in range(20):

    # Only read every 2nd picture
    if i % 2 == 1:
        continue

    for light in ['daylight', 'roomlight']:

        color_array = utils.read_blob("%s/color_%d_%s.blob" % (path, i, light), blob_image_type='color')
        color_array = color_array[505:625, 830:1010, :].copy()
        ir_array = utils.read_blob("%s/ir_%d_%s.blob" % (path, i, light), blob_image_type='ir')
        ir_array = ir_array[190:230, 202:262].copy()

        color = colors[int(math.floor(i / 2.0))]

        color_arrays.append((color, light, color_array))
        ir_arrays.append((color, light, ir_array))

color_mean, color_variance = {}, {}

for color, light, color_array in color_arrays:
    if not color in color_mean:
        color_mean[color] = {}
    if not color in color_variance:
        color_variance[color] = {}

    color_mean[color][light] = np.mean(color_array, axis=(0, 1))
    color_variance[color][light] = np.var(color_array, axis=(0, 1))

ir_mean, ir_var = {}, {}

for color, light, ir_array in ir_arrays:
    if not color in ir_mean:
        ir_mean[color] = {}
    if not color in ir_var:
        ir_var[color] = {}

    ir_mean[color][light] = np.mean(ir_array)
    ir_var[color][light] = np.var(ir_array)

def rgb_float_list_to_int_string(rgb):
    rgb = rgb * 255
    rgb = rgb.astype(np.uint8)

    return "(%d, %d, %d)" % (rgb[0], rgb[1], rgb[2])


# Print results
print "Color \t Color Mean  \t Color Var \t IR Mean \t IR Var"
for color in colors:

    print color, '\t', rgb_float_list_to_int_string(color_mean[color]['daylight']), '\t\t',\
          rgb_float_list_to_int_string(color_variance[color]['daylight']), '\t',\
          ir_mean[color]['daylight'], '\t', ir_var[color]['daylight']

print ""
print "Roomlight"
print "Color \t Color Mean  \t Color Var \t IR Mean \t IR Var"
for color in colors:

    print color, '\t', rgb_float_list_to_int_string(color_mean[color]['roomlight']), '\t\t',\
          rgb_float_list_to_int_string(color_variance[color]['roomlight']), '\t',\
          ir_mean[color]['roomlight'], '\t', ir_var[color]['roomlight']

# Save results as .yaml
def array_representer(dumper, data):
    return dumper.represent_sequence('!array', data.tolist())

def numpy_scalar_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:float', str(data.item()))

yaml.add_representer(np.ndarray, array_representer)
yaml.add_multi_representer(np.generic, numpy_scalar_representer)

with open(path + '/results-experiment2-mean-variance.yaml', 'w') as f:
    f.write(yaml.dump({'color_variance' : color_variance, 'color_mean' : color_mean, 'ir_variance' : ir_var, 'ir_mean' : ir_mean}))

plt.show()