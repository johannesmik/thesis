__author__ = 'johannes'

from OpenGL.GL import *
import re
import numpy as np


class Light(object):
    """ Baseclass for a light.
    """

    counter = 0
    registered_names = []

    def __init__(self, color):
        self.color = color
        self.uniforms = {}
        self.lighttype = 'baselights'
        self._register_name('Light')
        self.number = 0

    def put_up_uniforms(self, shader):
        for uniform_name, uniform_value in self.uniforms.items():
            location = glGetUniformLocation(shader.program, '%s[%i].%s' % (self.lighttype, self.number, uniform_name))
            if type(uniform_value) is list:
                uniform_value_len = len(uniform_value)
            elif type(uniform_value) is np.ndarray:
                uniform_value_len = uniform_value.size
            elif type(uniform_value) is int or float:
                uniform_value_len = 1

            if uniform_value_len == 1:
                glUniform1f(location, uniform_value)
            elif uniform_value_len == 2:
                glUniform2f(location, *uniform_value)
            elif uniform_value_len == 3:
                glUniform3f(location, *uniform_value)
            elif uniform_value_len == 4:
                glUniform4f(location, *uniform_value)

    def _register_name(self, name):

        if name in Light.registered_names:
            oldname = name
            match = re.search(r'(.*?)(\d+)$', oldname)
            if match:
                new_index = int(match.group(2)) + 1
                newname = "%s%03d" % (match.group(1), new_index)
            else:
                newname = oldname + "001"
            print "Reconsider light name", oldname, " to ", newname
            self._register_name(newname)
            return

        else:
            self.name = name
            Light.registered_names.append(name)

    def set_parameters(self, **kwargs):
        for parameter_name, parameter_value in kwargs.iteritems():
            if parameter_name in self.uniforms:
                self.uniforms[parameter_name] = parameter_value
            else:
                print self.__class__.__name__, 'has not attribute', parameter_name

    def set_position(self, position):
        if 'position' in self.uniforms:
            self.uniforms['position'] = position
        else:
            print self.__class__.__name__, 'has no attribute position'

    def __repr__(self):
        return self.__class__.__name__ + " (name: " + self.name + ")"

class DirectionalLight(Light):
    def __init__(self, direction, name='Directionallight', position=None, color=None, falloff=None):
        self.lighttype = 'directionlights'
        self._register_name(name)
        self.position = position
        self.color = color
        self.falloff = falloff if falloff else 0
        self.direction = direction / np.linalg.norm(direction)

        self.uniforms = {'color': color, 'direction': direction}

        self.number = DirectionalLight.counter
        DirectionalLight.counter += 1


class PointLight(Light):
    def __init__(self, position, color, falloff, name='Pointlight'):
        self.lighttype = 'pointlights'
        self._register_name(name)
        self.uniforms = {'color': color, 'position': position, 'falloff': falloff}

        self.number = PointLight.counter
        PointLight.counter += 1


class AmbientLight(Light):
    def __init__(self, color, name='Ambientlight'):
        self.lighttype = 'ambientlights'
        self._register_name(name)
        self.uniforms = {'color': color}

        self.number = AmbientLight.counter
        AmbientLight.counter += 1


class SpotLight(Light):
    def __init__(self, position, direction, cone_angle, color, falloff, name='Spotlight', ):
        self.lighttype = 'spotlights'
        self._register_name(name)
        self.uniforms = {'color': color, 'position': position, 'direction': direction,
                         'cone_angle': cone_angle, 'falloff': falloff}

        self.number = SpotLight.counter
        SpotLight.counter += 1
