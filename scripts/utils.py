from __future__ import print_function

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_blob(blob_filename, blob_image_type):
    """
    Returns an np.array containing the blob data

    :param blob_image_type: color, depth, ir
    """

    if blob_image_type == "color":
        dimensions = (1080, 1920, 4)
        dtype = np.uint8
        mode = 'RGBA'
    if blob_image_type == "depth":
        dimensions = (424, 512)
        dtype = np.float32
        mode = 'L'
    if blob_image_type == "ir":
        dimensions = (424, 512)
        dtype = np.float32
        mode = 'L'

    a = np.fromfile(blob_filename, dtype=dtype)
    a = a.reshape(dimensions)

    if blob_image_type == "color":
        # Swaps BGRA to RGBA
        a[:, :, [0, 1, 2, 3]] = a[:, :, [2, 1, 0, 3]]
    if blob_image_type == "depth":
        a = a / 4500.0
        a = np.clip(a, 0, 1) * 255
        a = a.astype(np.uint8)
    if blob_image_type == "ir":
        a = a / 20000.0
        a = np.clip(a, 0, 1) * 255
        a = a.astype(np.uint8)

    return a

def convert_blob(blob_filename, png_filename, image_type):
    """

    :param image_type: color, depth, ir
    :return:
    """

    if image_type == "color":
        mode = 'RGBA'
    if image_type == "depth":
        mode = 'L'
    if image_type == "ir":
        mode = 'L'

    a = read_blob(blob_filename, image_type)
    b = Image.fromarray(a, mode=mode)
    b.save(png_filename)

    return

def show_image(image, title=None):

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

    # Set up the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.05)
    clb = plt.colorbar(im, cax)