__author__ = 'johannes'

from OpenGL.GL import *
from PIL import Image
import numpy as np

class BaseMaterial:

    def __init__(self):

        self.uniforms = {}
        self.textures = {}
        self.textures['colormap'] = self.empty_texture(color="white")

        self.uniforms = {}
        self.uniforms['basecolor'] = (np.array([1., 1., 1.]), 3)

    def put_up_uniforms(self, shader):
        for uniform_name in self.uniforms:
            uniformvalue, number_of_values = self.uniforms[uniform_name]
            location = shader.locations.get(uniform_name, -1)
            if number_of_values == 1:
                glUniform1f(location, uniformvalue)
            elif number_of_values == 2:
                glUniform2f(location, *uniformvalue)
            elif number_of_values == 3:
                glUniform3f(location, *uniformvalue)
            elif number_of_values == 4:
                glUniform4f(location, *uniformvalue)
            # print uniform_name, location

    def empty_texture(self, color='white'):
        "Create a one-by-one pixel texture with a given color and return it's id"
        if color == 'black':
            imagebytes = '\x00\x00\x00'
        else:
            imagebytes = '\xff\xff\xff' # white

        id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, imagebytes)
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return id

    def add_colormap(self, filename):
        self.add_texture('colormap', filename)

    def add_normalmap(self, filename):
        self.add_texture('normalmap', filename)
        self.uniforms['use_normalmap'] = (True, 1)

    def overwrite_texture(self, id, imagebytes, width, height):
        ''' Overwrite the texture id with imagebytes '''
        glBindTexture(GL_TEXTURE_2D, id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagebytes)

    def add_texture(self, name, filename):
        im = Image.open(filename)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        ix, iy, imagebytes = im.size[0], im.size[1], im.tobytes("raw")
        print "opened", filename, ix, iy
        id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagebytes)

        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self.textures[name] = id

    def put_up_textures(self, shader):

        for i, texture_name in enumerate(self.textures):
            id = self.textures[texture_name]
            glActiveTexture(GL_TEXTURE0 + id)
            glBindTexture(GL_TEXTURE_2D, id)
            glUniform1i(shader.locations.get(texture_name, -1), id)

class AmbientMaterial(BaseMaterial):
    pass

class LambertianMaterial(BaseMaterial):
    pass

class BlinnPhongMaterial(BaseMaterial):
    pass

class BRDFMaterial(BaseMaterial):
    pass

class NormalMaterial(BaseMaterial):
    pass

class DepthMaterial(BaseMaterial):
    pass