from __future__ import print_function

from PIL import Image
import numpy as np


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
        a = a.astype(np.int8)
    if blob_image_type == "ir":
        a = a / 20000.0
        a = np.clip(a, 0, 1) * 255
        a = a.astype(np.int8)

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
