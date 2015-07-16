import os

import utils

color_arrays = []
depth_arrays = []
ir_arrays = []

path = '~/dataset-experiment2'

for i in range(20):
    color_array = utils.read_blob("%s/color_%d.blob" % (path, i), blob_image_type='color')
    ir_array = utils.read_blob("%s/color_%d.blob" % (path, i), blob_image_type='ir')

    color_arrays.append(color_array)
    ir_arrays.append(ir_array)

colors = ['yellow', 'orange', 'red', 'purple', 'green', 'lightblue', 'blue', 'brown', 'black', 'white']

# TODO