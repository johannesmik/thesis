"""
Experiment 4 tests if the material properties (specularity) can be detected using the kinect
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.optimize
import math

import utils


def blinn(angle, k_diffuse, k_specular, n):

    angle = np.deg2rad(angle)
    diffuse_term = k_diffuse * np.cos(angle)
    specular_term = k_specular * np.power(np.cos(angle * 2), n)
    return diffuse_term + specular_term

def fit_to_blinn(angles, values):
    popt, pcov = scipy.optimize.curve_fit(blinn, angles.astype(np.float64), values, p0=np.array([1.0, 0.0, 50], dtype=np.float64))
    return popt

angles = np.array(range(0, 50, 5))
aluminium_means = []
paper_means = []

path = os.path.expanduser('~/workspace/dataset-experiment4')

for i in range(len(angles)):

    ir = utils.read_blob("%s/ir_%d.blob" % (path, i), blob_image_type='ir')

    aluminium_mean = np.mean(ir[178:193, 253:259])
    paper_mean = np.mean(ir[153:168, 253:259])

    # get the mean values of the border

    aluminium_means.append(aluminium_mean)
    paper_means.append(paper_mean)

# plot_variance_histogram(ir_mean[np.where(0 < ir_mean)], ir_variance[np.where(0 < ir_mean)])
plt.figure(figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.plot(angles, paper_means, 'ro', label='Paper')
plt.plot(angles, aluminium_means, 'bo', label="Aluminium")


paper_popt = fit_to_blinn(angles, paper_means)
aluminium_popt = fit_to_blinn(angles, aluminium_means)

print "aluminium popt", aluminium_popt
print "paper popt", paper_popt

angles_cont = np.linspace(0, 60, 120)
plt.plot(angles_cont, blinn(angles_cont, *aluminium_popt), c='b', label=None)
plt.plot(angles_cont, blinn(angles_cont, *paper_popt), c='r',)

plt.legend()
plt.xlabel("Angle $\\varphi$ between camera axis / light direction and surface normal")
plt.ylabel("mean IR Intensity")
plt.show()