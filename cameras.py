__author__ = 'johannes'

import numpy as np


class Camera(object):
    """
      Camera baseclass
    """

    def __init__(self, name="Camera", position=None, rotation=None):
        """
        A simple camera. You can change (XYZ) position and (XYZ) rotation.
        """
        self.name = name

        self.position = position if isinstance(position, np.ndarray) else np.array([0, 0, 0], 'float')

        # Initially, the camera looks in the -Z direction, and up in the Y direction (0, 0, 0)
        self.rotation = rotation if isinstance(rotation, np.ndarray) else np.array([0, 0, 0], 'float')

        self.frames = {}

    def set_frame(self, time, position, rotation):
        """ Set keyframe. Translation and rotation both 3-vectors. """
        self.frames[time] = {'rotation' : rotation, 'position' : position}

    def remove_frame(self, time):
        if self.frames.get(time):
            self.frames.pop(time)

    def set_current_frame(self, time):
        """ Set translation and rotation to the values in time. Linearly interpolate between keyframes. """
        if self.frames.get(time):
            frame = self.frames.get(time)
            self.position = frame['position']
            self.rotation = frame['rotation']
        elif time < min(self.frames.keys()):
            # Before the first frame starts
            frame = self.frames.get(min(self.frames.keys()))
            self.position = frame['position']
            self.rotation = frame['rotation']
        elif time > max(self.frames.keys()):
            # After the first frame ended
            frame = self.frames.get(max(self.frames.keys()))
            self.position = frame['position']
            self.rotation = frame['rotation']
        else:
            # Mix the previous and next frame together
            timeA = min(self.frames.keys(), key=lambda k: 7000 if -k+time <= 0 else -k+time)
            timeB = min(self.frames.keys(), key=lambda k: 7000 if k-time <= 0 else k - time)
            frameA = self.frames[timeA]
            frameB = self.frames[timeB]
            factorA = (timeB - time) / float(timeB - timeA)
            factorB = (time - timeA) / float(timeB - timeA)
            self.position = factorA * frameA['position'] + factorB * frameB['position']
            self.rotation = factorA * frameA['rotation'] + factorB * frameB['rotation']

    def move_forward(self, length):
        self.position += np.dot(self.rotationmatrix.T, np.array([0, 0, -length, 1]))[0:3]

    def move_backward(self, length):
        self.move_forward(-length)

    def move_right(self, length):
        self.position += np.dot(self.rotationmatrix.T, np.array([length, 0, 0, 1]))[0:3]

    def move_left(self, length):
        self.move_right(-length)

    def rotate_left(self, angle):
        self.rotation[1] += angle

    def rotate_right(self, angle):
        self.rotation[1] -= angle

    def rotate_up(self, angle):
        self.rotation[0] -= angle

    def rotate_down(self, angle):
        self.rotation[0] += angle

    @property
    def translationmatrix(self):
        translation = np.eye(4)
        translation[0:3, 3] = - self.position
        return translation

    @property
    def rotationmatrix(self):
        x, y, z = -self.rotation
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
    def viewmatrix(self):
        return np.dot(self.rotationmatrix, self.translationmatrix)

    @property
    def projectionmatrix(self):
        """ To be implemented by the subclasses """
        return np.eye(4)

    def reshape(self, width, height):
        """ Called when the dimensions of the viewport have changed """
        pass


class PerspectiveCamera(Camera):
    # IDEA: cache matrix entries

    def __init__(self, name=None, position=None, rotation=None):
        super(PerspectiveCamera, self).__init__(name=name, position=position, rotation=rotation)

        self.init_frustum()

    def init_frustum(self):
        self.near, self.far = 1.0, 20
        self.left, self.right = -1.0, 1.0
        self.top, self.bottom = 1.0, -1.0

    def reshape(self, width, height):
        aspect = float(height) / width
        self.top, self.bottom = aspect, -aspect

    @property
    def projectionmatrix(self):
        # See http://www.songho.ca/opengl/gl_projectionmatrix.html
        # Or 'Red Book' chapter 5
        return np.array([[self.near / self.right, 0, 0, 0],
                         [0, self.near / self.top, 0, 0],
                         [0, 0, -(self.far + self.near) / (self.far - self.near),
                          -2 * self.far * self.near / (self.far - self.near)],
                         [0, 0, -1, 0]])


class PerspectiveCamera2(Camera):
    """
    Use another Projection Matrix, such that there is a linear relationship between Z (world) and z_c (clip)
    """

    def __init__(self, name=None, width=None, height=None, position=None, rotation=None):
        super(PerspectiveCamera2, self).__init__(name=name, position=position, rotation=rotation)

        # For the Kinect depth Image
        self.fx = 368.096588
        self.fy = 368.096588
        self.ox = 261.696594
        self.oy = 202.522202
        self.near, self.far = 0.4, 8.

        self.right = (512 * self.near) / (2 * self.fx)
        self.top = (424 * self.near) / (2 * self.fy)

    def reshape(self, width, height):
        pass

    @property
    def projectionmatrix(self):
        return np.array([[self.near / self.right, 0, 0, 0],
                         [0, self.near / self.top, 0, 0],
                         [0, 0, -(self.far + self.near) / (self.far - self.near),
                          -2 * self.far * self.near / (self.far - self.near)],
                         [0, 0, -1, 0]])


class PerspectiveCamera3(Camera):
    def __init__(self, name=None, position=None, rotation=None):
        super(PerspectiveCamera3, self).__init__(name=name, position=position, rotation=rotation)

        self.ox, self.oy = 50, 50
        self.fx, self.fy = 1, 1
        self.sx, self.sy = 100, 100

    def reshape(self, width, height):
        pass

    @property
    def projectionmatrix(self):
        # TODO double check this projection matrix. Consult Ma et al, chapter 3.

        pi = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])

        k = np.array([[self.fx * self.sx, 0, self.ox],
                      [0, self.fy * self.sy, self.oy],
                      [0, 0, 1]])

        return np.dot(k, pi)
