__author__ = 'johannes'

import numpy as np

def add_depth_noise(image, depth_scale_factor):
    """
    :param image: numpy array with values between 0 and 1
    :param depth_scale_factor: how many meters is value 1? (Something around 7,5)
    :return:
    """

    sigma = (0.0018 * np.pow(image, -2) + 0.1) * 10**(-5)

    return image + sigma * np.random.randn(*image.shape).astype(image.dtype)

def add_ir_noise(image):
    """
    :param image: numpy array with values between 0 and 1
    :return:
    """

    sigma = 0.7 * 10**-5 * np.ones_like(image)
    sigma[np.where(image >= 0.16)] = 0.8 * 10**-4

    return image + sigma * np.random.randn(*image.shape).astype(image.dtype)
