__author__ = 'johannes'

from OpenGL.GL import *
import numpy as np

class Light(object):
    """ Baseclass for a light.
    """
    def __init__(self, color):
        self.color = color
        self.uniforms = {}
        self.lighttype = 'baselights'

    def put_up_uniforms(self, shader, lightnumber):
        for uniform_name in self.uniforms:
            uniformvalue, number_of_values = self.uniforms[uniform_name]
            location = glGetUniformLocation(shader.program, '%s[%i].%s' % (self.lighttype, lightnumber, uniform_name))
            if number_of_values == 1:
                glUniform1f(location, uniformvalue)
            elif number_of_values == 2:
                glUniform2f(location, *uniformvalue)
            elif number_of_values == 3:
                glUniform3f(location, *uniformvalue)
            elif number_of_values == 4:
                glUniform4f(location, *uniformvalue)

class DirectionalLight(Light):

    def __init__(self, position=None, color=None, falloff=None, direction=None):
        self.lighttype = 'directionlights'
        self.position = position
        self.color = color
        self.falloff = falloff
        self.direction = direction / np.linalg.norm(direction)

        self.uniforms = {}
        self.uniforms['color'] = [color, 4]
        self.uniforms['direction'] = [ direction, 3]

class PointLight(Light):
    def __init__(self, position, color, falloff):
        self.lighttype = 'pointlights'
        self.uniforms = {}
        self.uniforms['color'] = [color, 4]
        self.uniforms['position'] = [position, 3]
        self.uniforms['falloff'] = [falloff, 1]

class AmbientLight(Light):
    def __init__(self, color):
        self.lighttype = 'ambientlights'
        self.uniforms = {}
        self.uniforms['color'] = (color, 4)

class SpotLight(Light):
    pass