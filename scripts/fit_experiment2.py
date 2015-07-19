# Linear fit from RGB to IR mean values.

import os
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
