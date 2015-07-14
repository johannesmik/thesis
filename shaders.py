__author__ = 'johannes'

from OpenGL.GL import *
from OpenGL.GL import shaders


class Shader:
    def __init__(self, vertexshader, fragmentshader, attributes, uniforms, verbose=False):
        """ Initialize the Shader and get locations for the GLSL variables """

        with open(vertexshader) as f:
            vertex = shaders.compileShader(f.read(), GL_VERTEX_SHADER)
        with open(fragmentshader) as f:
            fragment = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex, fragment)

        self.attribute_locations = {}
        for attribute in attributes:
            location = glGetAttribLocation(self.program, attribute)
            if location is None:
                raise Exception("Could not get location of attribute %s" % attribute)
            elif location == -1 and verbose:
                print "Attribute %s is not used in the shader (%s or %s)" % (attribute, vertexshader, fragmentshader)
            self.attribute_locations[attribute] = location

        self.uniform_locations = {}
        for uniform in uniforms:
            location = glGetUniformLocation(self.program, uniform)
            if location is None:
                raise Exception("Could not get location of uniform %s" % uniform)
            elif location == -1 and verbose:
                print "Uniform %s is not used in the shader (%s or %s)" % (uniform, vertexshader, fragmentshader)
            self.uniform_locations[uniform] = location

        self.locations = {}
        self.locations.update(self.attribute_locations)
        self.locations.update(self.uniform_locations)


class ShaderLib:
    """
    Prepares all available shaders

     - Ambient Shader
     - Lambert Shader
     - Blinn Phong Shader
     - Specular Shader
     - Depth Shader
     - Normal Shader
    """

    def __init__(self):
        self.ambient = Shader('./shaders/vertex.glsl', './shaders/fragment-ambient.glsl',
                              ['position', 'texcoords'],
                              ['PMatrix', 'MMatrix', 'VMatrix'])

        self.depth = Shader('shaders/vertex.glsl', 'shaders/fragment-depth.glsl',
                            ['position'],
                            ['PMatrix', 'MMatrix', 'VMatrix'])

        self.normal = Shader('shaders/vertex.glsl', 'shaders/fragment-normal.glsl',
                             ['position', 'color', 'normal', 'texcoords'],
                             ['PMatrix', 'MMatrix', 'VMatrix', 'colormap', 'normalmap', 'depthmap', 'basecolor',
                              'use_normalmap', 'use_depthmap'])

        self.lambertian = Shader('shaders/vertex.glsl', 'shaders/fragment-lambertian.glsl',
                                 ['position', 'color', 'normal', 'texcoords'],
                                 ['PMatrix', 'MMatrix', 'VMatrix', 'colormap', 'normalmap', 'depthmap', 'basecolor',
                                  'use_normalmap', 'use_colormap', 'use_depthmap'])

        self.blinnphong = Shader('shaders/vertex.glsl', 'shaders/fragment-blinnphong.glsl',
                                 ['position', 'color', 'normal', 'texcoords'],
                                 ['PMatrix', 'MMatrix', 'VMatrix', 'colormap', 'normalmap', 'depthmap', 'basecolor',
                                  'use_normalmap', 'use_colormap', 'specularity', 'specular_color'])
