__author__ = 'johannes'

from OpenGL.GL import *
from PIL import Image
import numpy as np


class BaseMaterial(object):
    def __init__(self):

        self.uniforms = {}
        self.textures = {'colormap': self.empty_texture(color="white")}

        self.uniforms = {'basecolor': np.array([1., 1., 1.])}

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

    @staticmethod
    def empty_texture(color='white'):
        """Create a one-by-one pixel texture with a given color and return it's id"""
        if color == 'black':
            imagebytes = '\x00\x00\x00'
        else:
            imagebytes = '\xff\xff\xff'  # white

        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, imagebytes)
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        return texture_id

    def add_colormap(self, filename):
        self.add_texture('colormap', filename)
        self.uniforms['use_colormap'] = True

    def add_normalmap(self, filename):
        self.add_texture('normalmap', filename)
        self.uniforms['use_normalmap'] = True

    def add_depthmap(self, filename):
        self.add_texture('depthmap', filename, repeat_s=False, repeat_t=False)
        self.uniforms['use_depthmap'] = True

    def overwrite_texture(self, name, imagebytes, width, height):
        """ Overwrite the texture id with imagebytes """
        texture_id = self.textures[name]
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, imagebytes)

    def overwrite_texture_bw(self, name, imagebytes, width, height):
        """ Overwrite the texture id with imagebytes """
        texture_id = self.textures[name]
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, imagebytes)

    def add_texture(self, name, filename, repeat_s=True, repeat_t=True):
        gl_texture_format = GL_RGBA
        gl_format = GL_RGBA
        gl_type = GL_UNSIGNED_BYTE

        im = Image.open(filename)
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

        if im.mode == 'F':
            print "mode is F"
            gl_texture_format = GL_RGB32F
            gl_format = GL_RED
            gl_type = GL_FLOAT


        ix, iy, imagebytes = im.size[0], im.size[1], im.tobytes("raw")
        print "opened", filename, ix, iy
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, gl_texture_format, ix, iy, 0, gl_format, gl_type, imagebytes)

        wrap_s = GL_REPEAT if repeat_s else GL_CLAMP
        wrap_t = GL_REPEAT if repeat_t else GL_CLAMP

        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self.textures[name] = texture_id

    def put_up_textures(self, shader):

        for i, texture_name in enumerate(self.textures):
            texture_id = self.textures[texture_name]
            if texture_name == 'colormap':
                texture_unit = 0
            elif texture_name == 'normalmap':
                texture_unit = 1
            elif texture_name == 'depthmap':
                texture_unit = 2
            glActiveTexture(GL_TEXTURE0 + texture_unit)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glUniform1i(shader.locations.get(texture_name, -1), texture_unit)


class AmbientMaterial(BaseMaterial):
    pass


class LambertianMaterial(BaseMaterial):
    pass


class BlinnPhongMaterial(BaseMaterial):
    def __init__(self, specularity=1.0, specular_color=np.array([1., 1., 1.])):
        super(BlinnPhongMaterial, self).__init__()

        self.uniforms['specularity'] = specularity
        self.uniforms['specular_color'] = specular_color


class BRDFMaterial(BaseMaterial):
    pass


class NormalMaterial(BaseMaterial):
    pass


class DepthMaterial(BaseMaterial):
    pass
