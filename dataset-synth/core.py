__author__ = 'johannes'

import numpy as np


class Object3D(object):
    # TODO cache matrices
    def __init__(self, name=None, position=None, rotation=None, size=1):

        self.name = name if not None else "Unnamed Object"

        self.position = position if isinstance(position, np.ndarray) else np.array([0, 0, 0], 'float')

        self.rotation = rotation if isinstance(rotation, np.ndarray) else np.array([0, 0, 0], 'float')

        self.size_x = size
        self.size_y = size
        self.size_z = size

        self.visible = True

    def __repr__(self):
        return self.__class__.__name__ + " (name: " + self.name + ")"

    @property
    def translationmatrix(self):
        translation = np.eye(4)
        translation[0:3, 3] = self.position
        return translation

    @property
    def rotationmatrix(self):
        x, y, z = self.rotation
        zrotation = np.array([[np.cos(z), np.sin(z), 0, 0],
                              [-np.sin(z), np.cos(z), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        yrotation = np.array([[np.cos(y), 0, np.sin(y), 0],
                              [0, 1, 0, 0],
                              [-np.sin(y), 0, np.cos(y), 0],
                              [0, 0, 0, 1]])
        xrotation = np.array([[1, 0, 0, 0],
                              [0, np.cos(x), np.sin(x), 0],
                              [0, -np.sin(x), np.cos(x), 0],
                              [0, 0, 0, 1]])
        xyz_rotation = np.dot(xrotation, np.dot(yrotation, zrotation))
        return xyz_rotation

    @property
    def scalematrix(self):
        scale = np.array([[self.size_x, 0, 0, 0],
                          [0, self.size_y, 0, 0],
                          [0, 0, self.size_z, 0],
                          [0, 0, 0, 1]])
        return scale

    @property
    def modelmatrix(self):
        return np.dot(np.dot(self.translationmatrix, self.rotationmatrix), self.scalematrix)

    def set_size(self, *args):
        if len(args) == 1:
            self.size_x = args[0]
            self.size_y = args[0]
            self.size_z = args[0]
        elif len(args) == 3:
            self.size_x = args[0]
            self.size_y = args[1]
            self.size_z = args[2]

    def toggle_visibility(self):
        if self.visible:
            self.visible = False
        else:
            self.visible = True