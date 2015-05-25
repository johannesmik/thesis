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
        self.uniforms['basecolor'] = np.array([1., 1., 1.])

    def put_up_uniforms(self, shader):
        for uniform_name, uniform_value in self.uniforms.items():
            location = shader.locations.get(uniform_name, -1)
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
            print uniform_name, location

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
        self.uniforms['use_normalmap'] = True

    def add_depthmap(self, filename):
        self.add_texture('depthmap', filename)
        self.uniforms['use_depthmap'] = True

    # TODO make static, or let it use name instead of id and use id = self.textures[name]
    def overwrite_texture(self, id, imagebytes, width, height):
        ''' Overwrite the texture id with imagebytes '''
        glBindTexture(GL_TEXTURE_2D, id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagebytes)

    # TODO does not give correct result yet (called from reconstruction.py)
    def overwrite_texture_bw(self, id, imagebytes, width, height):
        ''' Overwrite the texture id with imagebytes '''
        glBindTexture(GL_TEXTURE_2D, id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, imagebytes)

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
            if texture_name == 'colormap':
                texture_unit = 0
            elif texture_name == 'normalmap':
                texture_unit = 1
            elif texture_name == 'depthmap':
                texture_unit = 2
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, id)
            glUniform1i(shader.locations.get(texture_name, -1), texture_unit)

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

class MaterialA(BaseMaterial):
    pass