__author__ = 'johannes'

from OpenGL.GL import *
import numpy as np


class Light(object):
    """ Baseclass for a light.
    """

    counter = 0

    def __init__(self, color):
        self.color = color
        self.uniforms = {}
        self.lighttype = 'baselights'
        self.name = "Light"
        self.number = 0

    def put_up_uniforms(self, shader):
        for uniform_name in self.uniforms:
            uniformvalue, number_of_values = self.uniforms[uniform_name]
            location = glGetUniformLocation(shader.program, '%s[%i].%s' % (self.lighttype, self.number, uniform_name))
            if number_of_values == 1:
                glUniform1f(location, uniformvalue)
            elif number_of_values == 2:
                glUniform2f(location, *uniformvalue)
            elif number_of_values == 3:
                glUniform3f(location, *uniformvalue)
            elif number_of_values == 4:
                glUniform4f(location, *uniformvalue)


class DirectionalLight(Light):
    def __init__(self, name='Light', position=None, color=None, falloff=None, direction=None):
        self.lighttype = 'directionlights'
        self.name = name
        self.position = position
        self.color = color
        self.falloff = falloff
        self.direction = direction / np.linalg.norm(direction)

        self.uniforms = {'color': [color, 4], 'direction': [direction, 3]}

        self.number = DirectionalLight.counter
        DirectionalLight.counter += 1


class PointLight(Light):
    def __init__(self, position, color, falloff, name='Light'):
        self.lighttype = 'pointlights'
        self.name = name
        self.uniforms = {'color': [color, 4], 'position': [position, 3], 'falloff': [falloff, 1]}

        self.number = PointLight.counter
        PointLight.counter += 1


class AmbientLight(Light):
    def __init__(self, color, name='Light'):
        self.lighttype = 'ambientlights'
        self.name = name
        self.uniforms = {'color': [color, 4]}

        self.number = AmbientLight.counter
        AmbientLight.counter += 1


class SpotLight(Light):
    def __init__(self, position, direction, cone_angle, color, falloff, name='Light', ):
        self.lighttype = 'spotlights'
        self.name = name
        self.uniforms = {'color': [color, 4], 'position': [position, 3], 'direction': [direction, 3],
                         'cone_angle': [cone_angle, 1], 'falloff': [falloff, 1]}

        self.number = SpotLight.counter
        SpotLight.counter += 1
