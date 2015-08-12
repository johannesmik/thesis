from __future__ import print_function
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pycuda.driver as drv

def show_image(image, title=None, colorbar=True):

    image_copy = image.copy()

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # Use grey colormap
    cm = plt.get_cmap("gray")

    # Show points that fall out of the region [0, 1] in pink
    cm.set_under('pink')
    cm.set_over('pink')

    # No ticks on image axis
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    minimum = min(image_copy.min(), 0.0)
    maximum = max(image_copy.max(), 1.0)

    print("Show Image (", title ,"): Chosen min/max", minimum, maximum, " Real min/max", image_copy.min(), image_copy.max())

    if title:
        plt.title(title)

    im = plt.imshow(image_copy, interpolation="nearest", cmap=cm, vmin=minimum, vmax=maximum)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.15, pad=0.05)
        clb = plt.colorbar(im, cax)

def set_texture_border_mode(texture, mode='clamp'):
    """
    :param texture:
    :param mode: 'clamp' or 'border'
    """

    if mode == 'clamp':
        address_mode = drv.address_mode.CLAMP
    elif mode == 'border':
        address_mode = drv.address_mode.BORDER

    texture.set_address_mode(0, address_mode)
    texture.set_address_mode(1, address_mode)

def update_texture(texture, numpy_array):
    texture.set_array(drv.matrix_to_array(numpy_array, 'C'))
