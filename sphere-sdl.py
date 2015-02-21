#!/usr/bin/env python
# Renders a simple sphere
# Author: Johannes

from OpenGL.GL import *
from sdl2 import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from OpenGL.GL.framebufferobjects import *
from PIL import Image
import numpy as np
import sys
import time
import math
import ctypes

import meshes

class Shader:
    def __init__(self, vertexshader, fragmentshader, attributes, uniforms):
        """ Initialize the Shader and get locations for the GLSL variables """
        with open(vertexshader) as f:
            vertex = shaders.compileShader(f.read(), GL_VERTEX_SHADER)
        with open(fragmentshader) as f:
            fragment = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)
        self.program = shaders.compileProgram(vertex, fragment)

        self.attribute_locations = {}
        for attribute in attributes:
            location = glGetAttribLocation(self.program, attribute)
            if location == None:
                raise Exception("Could not get location of attribute %s" % attribute)
            elif location == -1:
                print "Attribute %s is not used in the shader" % attribute
            self.attribute_locations[attribute] = location

        self.uniform_locations = {}
        for uniform in uniforms:
            location = glGetUniformLocation(self.program, uniform)
            if location == None:
                raise Exception("Could not get location of uniform %s" % uniform)
            elif location == -1:
                print "Uniform %s is not used in the shader." % uniform
            self.uniform_locations[uniform] = location

        self.locations = {}
        self.locations.update(self.attribute_locations)
        self.locations.update(self.uniform_locations)

class Camera:

    def __init__(self, position=None, rotation=None):
        """
        A simple camera. You can change (XYZ) position and (XYZ) rotation.
        """
        self.position = position if isinstance(position, np.ndarray) else np.array([0, 0, 0], 'float')

        # Initially, the camera looks in the -Z direction, and up in the Y direction (0, 0, 0)
        self.rotation = rotation if isinstance(rotation, np.ndarray) else np.array([0, 0, 0], 'float')

        self.init_frustum()

    def init_frustum(self):
        self.near, self.far = 1.0, 20
        self.left, self.right = -1.0, 1.0
        self.top, self.bottom = 1.0, -1.0

    @property
    def projectionmatrix(self):
        # See http://www.songho.ca/opengl/gl_projectionmatrix.html
        # Or 'Red Book' chapter 5
        return np.array([[self.near / self.right, 0, 0, 0],
                         [0, self.near/self.top, 0 , 0],
                         [0, 0, -(self.far+self.near)/(self.far-self.near), -2*self.far*self.near/(self.far - self.near)],
                         [0, 0, -1, 0]])

    @property
    def projectionmatrix2(self):
        # Got those from Christian
        # TODO double check them
        w = self.right - self.left
        h = self.top - self.bottom
        ox = 0.5
        oy = 0.5
        fx = 1
        fy = 1
        ortho = np.array([[2. / w, 0, 0, - 1],
                          [0, 2. / -h, 0, 1],
                          [0, 0, -2. / (self.far - self.near), (-self.far - self.near) / (self.far - self.near)],
                          [0, 0, 0, 1]])
        cam = np.array([[fx, 0, -ox - 0.5, 0],
                        [0, fy, -oy - 0.5 , 0],
                        [0, 0, self.near + self.far, self.near * self.far],
                        [0, 0, -1, 0]])
        return np.dot(ortho, cam)


    @property
    def translationmatrix(self):
        translation = np.eye(4)
        translation[0:3, 3] = - self.position
        return translation

    @property
    def rotationmatrix(self):
        x,y,z = -self.rotation
        zrotation = np.array([[np.cos(z), np.sin(z), 0, 0],
                              [-np.sin(z), np.cos(z), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        yrotation = np.array([[np.cos(y), 0, np.sin(y), 0],
                              [0, 1, 0, 0],
                              [-np.sin(y), 0, np.cos(y), 0],
                              [0, 0, 0, 1]])
        xrotation = np.array([[1, 0, 0 ,0],
                              [0, np.cos(x), np.sin(x), 0],
                              [0, -np.sin(x), np.cos(x), 0],
                              [0, 0, 0, 1]])
        xyz_rotation = np.dot(xrotation, np.dot(yrotation, zrotation))
        return xyz_rotation

    @property
    def viewmatrix(self):
        return np.dot(self.rotationmatrix, self.translationmatrix)

    def move_forward(self, length):
        self.position += np.dot(self.rotationmatrix.T, np.array([0, 0, -length, 1]))[0:3]

    def move_backward(self, length):
        self.move_forward(-length)

    def move_right(self, length):
        self.position += np.dot(self.rotationmatrix.T, np.array([length, 0, 0, 1]))[0:3]

    def move_left(self, length):
        self.move_right(-length)

class Context:
    def __init__(self, width, height):

        self.width = width
        self.height = height
        self.meshes = []
        self.camera = Camera(position=np.array([0., 0., 0.]))
        self.lighting_direction = 0
        self.current_draw_method = self.draw_color

        self.init_sdl()
        self.init_gl()
        self.init_shaders()
        self.init_framerate()

        self.put_up_texture()

    def addmesh(self, mesh):
        self.meshes.append(mesh)

    def popmesh(self):
        return self.meshes.pop()

    def init_framerate(self):
        self.tStart = self.t0 = time.time()
        self.frames = 0

    def init_gl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glCullFace(GL_BACK)

        # Initialize Framebuffer / Renderbuffer
        if not glBindRenderbuffer and not glBindFramebuffer:
            print 'Missing required extensions!'
            sys.exit()

        self.fbo = glGenFramebuffers(1)
        self.rbo = glGenRenderbuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, self.width, self.height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, self.rbo)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def init_sdl(self):
        SDL_Init(SDL_INIT_EVERYTHING)
        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8)
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8)
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8)
        SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8)
        SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32)
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16)
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1)

        self.window = SDL_CreateWindow("OpenGL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       self.width, self.height, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE)
        SDL_GL_CreateContext(self.window)
        self.event = SDL_Event()

    def init_shaders(self):
        self.color_shader = Shader('vertex.glsl', 'fragment-color.glsl',
                                   ['position', 'color', 'normal', 'texcoords'],
                                   ['PMatrix', 'MMatrix', 'VMatrix', 'light_direction', 'light_position', 'half_angle'])
        # For depth rendering
        self.depth_shader = Shader('vertex.glsl', 'fragment-depth.glsl',
                                   ['position', 'color', 'normal', 'texcoords'],
                                   ['PMatrix', 'MMatrix', 'VMatrix'])

        # For normal shading
        self.normal_shader = Shader('vertex.glsl', 'fragment-normal.glsl',
                                    ['position', 'color', 'normal', 'texcoords'],
                                    ['PMatrix', 'MMatrix', 'VMatrix'])

    def draw_to_renderbuffer(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.current_draw_method()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def draw_color(self):
        glClearColor(1.6, 1.6, 1.7, 1.0)
        self.framerate()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        shaders.glUseProgram(self.color_shader.program)

        # put up uniforms
        glUniformMatrix4fv(self.color_shader.locations['MMatrix'], 1, GL_TRUE, np.eye(4))
        glUniformMatrix4fv(self.color_shader.locations['VMatrix'], 1, GL_TRUE, self.camera.viewmatrix)
        glUniformMatrix4fv(self.color_shader.locations['PMatrix'], 1, GL_TRUE, self.camera.projectionmatrix)
        glUniform4f(self.color_shader.locations['light_direction'], math.sin(self.lighting_direction), 0, -1, 1)

        # Draw the meshes
        for mesh in self.meshes:
            mesh.draw(self.color_shader.locations)

        shaders.glUseProgram(0)
        SDL_GL_SwapWindow(self.window)

    def draw_depth(self):

        # Drawing the depth map
        glClearColor(1.0, 1.0, 1.0, 1.0)
        self.framerate()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        shaders.glUseProgram(self.depth_shader.program)

        # put up uniforms
        glUniformMatrix4fv(self.depth_shader.locations['MMatrix'], 1, GL_TRUE, np.eye(4))
        glUniformMatrix4fv(self.color_shader.locations['VMatrix'], 1, GL_TRUE, self.camera.viewmatrix)
        glUniformMatrix4fv(self.color_shader.locations['PMatrix'], 1, GL_TRUE, self.camera.projectionmatrix)

        # Draw the meshes
        for mesh in self.meshes:
            mesh.draw(self.depth_shader.locations)

        shaders.glUseProgram(0)
        SDL_GL_SwapWindow(self.window)

    def draw_normal(self):

        # Drawing the normal map
        glClearColor(0.0, 0.5, 0.7, 1.0)
        self.framerate()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        shaders.glUseProgram(self.normal_shader.program)

        # put up uniforms
        glUniformMatrix4fv(self.normal_shader.locations['MMatrix'], 1, GL_TRUE, np.eye(4))
        glUniformMatrix4fv(self.color_shader.locations['VMatrix'], 1, GL_TRUE, self.camera.viewmatrix)
        glUniformMatrix4fv(self.color_shader.locations['PMatrix'], 1, GL_TRUE, self.camera.projectionmatrix)

        # Draw the meshes
        for mesh in self.meshes:
            mesh.draw(self.normal_shader.locations)

        shaders.glUseProgram(0)
        SDL_GL_SwapWindow(self.window)

    def framerate(self, updaterate=1.0):
        t = time.time()
        self.frames += 1
        if t - self.t0 >= updaterate:
            seconds = t - self.t0
            fps = self.frames / seconds
            print "%.0f frames in %3.1f seconds = %6.3f FPS" % (self.frames, seconds, fps)
            self.t0 = t
            self.frames = 0

    def put_up_texture(self):
        # Put up the texture
        im = Image.open("texture.png")
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
        ix, iy, imagebytes = im.size[0], im.size[1], im.tobytes("raw")
        ID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, ID)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, ix, iy, 0, GL_RGB, GL_UNSIGNED_BYTE, imagebytes)

        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D, ID)

    def keyPressed(self, event):
        ''' Handles the event when a key is pressed. '''

        if event.key.keysym.sym == SDLK_ESCAPE:
            sys.exit()

        if event.key.keysym.sym == SDLK_UP or event.key.keysym.sym == SDLK_w:
            self.camera.move_forward(0.1)

        if event.key.keysym.sym == SDLK_DOWN or event.key.keysym.sym == SDLK_s:
            self.camera.move_backward(0.1)

        if  event.key.keysym.sym == SDLK_a:
            self.camera.move_left(0.1)

        if event.key.keysym.sym == SDLK_d:
            self.camera.move_right(0.1)

        if event.key.keysym.sym == SDLK_LEFT  or event.key.keysym.sym == SDLK_q:
            self.camera.rotation[1] += 0.05

        if event.key.keysym.sym == SDLK_RIGHT  or event.key.keysym.sym == SDLK_e:
            self.camera.rotation[1] -= 0.05

        if event.key.keysym.sym == SDLK_y:
            self.lighting_direction += 0.05

        # Change drawing methods
        if event.key.keysym.sym == SDLK_1 or event.key.keysym.sym == SDLK_r:
            self.current_draw_method = self.draw_color

        if event.key.keysym.sym == SDLK_2 or event.key.keysym.sym == SDLK_n:
            self.current_draw_method = self.draw_depth

        if event.key.keysym.sym == SDLK_3 or event.key.keysym.sym == SDLK_v:
            self.current_draw_method = self.draw_normal

        if event.key.keysym.sym == SDLK_t:
            print self.meshes[-1]
            self.meshes[-1].toggle_visibility()

        # Screenshots
        if event.key.keysym.sym == SDLK_f:
            self.draw_color()
            buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            color_image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=buffer)
            color_image = color_image.transpose(Image.FLIP_TOP_BOTTOM)
            with open("screen_color.png", 'w') as f:
                color_image.save(f)
            print "took color screenshot saved as 'screen_color.png'"
        if event.key.keysym.sym == SDLK_g:
            self.draw_depth()
            buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            depth_image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=buffer)
            depth_image = depth_image.transpose(Image.FLIP_TOP_BOTTOM)
            with open("screen_depth.png", 'w') as f:
                depth_image.save(f)
            print "took depth screenshot and saved as 'screen_depth.png'"
        if event.key.keysym.sym == SDLK_h:
            self.draw_normal()
            buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            normal_image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=buffer)
            normal_image = normal_image.transpose(Image.FLIP_TOP_BOTTOM)
            with open("screen_normal.png", 'w') as f:
                normal_image.save(f)
            print "took depth screenshot saved as 'screen_normal.png'"

        if event.key.keysym.sym == SDLK_p:
            print "Debug info (Key g)"
            print ""

    def resize_event(self, event):
        ''' Handles a resize event. Event: SDL-event data. '''
        self.reshape(event.window.data1, event.window.data2)

    def reshape(self, width, height):
        ''' Resize the frustum. Width and height in pixels. '''
        self.width, self.height = width, height

        glViewport(0, 0, width, height)
        aspect = float(height) / width

        self.camera.top, self.camera.bottom = aspect, -aspect
        self.camera.width, self.camera.height = width, height

        # Resize Renderbuffer object
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, self.width, self.height)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

    def controlloop(self):
        while SDL_PollEvent(ctypes.byref(self.event)) != 0:

            if self.event.type == SDL_QUIT:
                return False
            if self.event.type == events.SDL_KEYDOWN:
                c.keyPressed(self.event)
            if self.event.type == SDL_MOUSEMOTION:
                pass
            if self.event.type == SDL_MOUSEBUTTONDOWN:
                pass
            if self.event.type == SDL_WINDOWEVENT and self.event.window.event == SDL_WINDOWEVENT_RESIZED:
                c.resize_event(self.event)

        self.current_draw_method()
        return True

    @staticmethod
    def print_information():
        ''' Print out information about the OpenGL version. '''
        print "GL_RENDERER   = ", glGetString(GL_RENDERER)
        print "GL_VERSION    = ", glGetString(GL_VERSION)
        print "GL_VENDOR     = ", glGetString(GL_VENDOR)
        print "GL_EXTENSIONS = ", glGetString(GL_EXTENSIONS)


if __name__ == '__main__':

    # Create the context
    c = Context(width=300, height=300)

    # Add some simple meshes to the context
    c.addmesh(meshes.Icosphere(color=np.array([1.0, 1.0, 1.0]), position=np.array([0, 0, -2])))
    #c.addmesh(meshes.Icosphere(color=np.array([1.0, 0.6, 0]), position=np.array([-2, 0, -1])))
    #c.addmesh(meshes.Icosphere(color=np.array([1.0, 0.6, 0]), position=np.array([0, 4, -15])))
    #c.addmesh(meshes.Square(position=np.array([-2,0,-4])))
    #c.addmesh(meshes.Depthmap())
    #c.addmesh(meshes.StructureDepthmap(file="face-depth.png"))
    #c.addmesh(meshes.Pointcloud(vertices=np.array([[0, 0, -2], [1, 0, -5]])))
    # Print information about the Graphics card
    c.print_information()

    # Move camera to a convenient location
    #c.camera.position[2] += 0.8
    #c.camera.position[0] += 1.5
    #c.camera.rotation[1] += 0.75

    # The Loop
    running = True
    while running:
        running = c.controlloop()
