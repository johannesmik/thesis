#!/usr/bin/env python
# Renders a simple sphere
# Author: Johannes

import OpenGL
from OpenGL.GL import *
# from OpenGL.GLUT import *
from sdl2 import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from PIL import Image
import numpy as np
import sys
import time
import math
import ctypes

def normalized(array, axis=1):
    n = np.linalg.norm(array, axis=1)
    n[n==0] = 1
    return  array / np.expand_dims(n, 1)

class Mesh:
    """ Base class for all meshes. """

    def draw(self, locations):
        stride = 12  # 3 * 4 bits stride

        try:
            glEnableVertexAttribArray(locations['position'])
            glEnableVertexAttribArray(locations['color'])
            glEnableVertexAttribArray(locations['normal'])

            self.vertices.bind()
            glVertexAttribPointer(locations['position'], 3, GL_FLOAT, False, stride, self.vertices)

            self.colors.bind()
            glVertexAttribPointer(locations['color'], 3, GL_FLOAT, False, stride, self.colors)

            self.normals.bind()
            glVertexAttribPointer(locations['normal'], 3, GL_FLOAT, False, stride, self.normals)

            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_SHORT, self.indices)

        finally:
            self.vertices.unbind()
            self.colors.unbind()
            self.normals.unbind()
            glDisableVertexAttribArray(locations['position'])
            glDisableVertexAttribArray(locations['color'])
            glDisableVertexAttribArray(locations['normal'])

        return 0

class Triangle(Mesh):
    """ Create a small triangle """
    def __init__(self):
        self.vertices = vbo.VBO(np.array([[0., 1., 0], [-1., -1., 0], [1., -1., 0]], 'float32'),
                                usage=GL_STATIC_DRAW)

        self.colors = vbo.VBO(np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], 'float32'),
                              usage=GL_STATIC_DRAW)

        self.normals = vbo.VBO(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], 'float32'),
                               usage=GL_STATIC_DRAW)

        self.indices = [0, 1, 2]

class Icosphere(Mesh):
    """ Create a mesh for an Icosphere """

    def __init__(self, subdivisions=3, color=None):

        t = (1 + math.sqrt(5.0)) / 2.0

        self.color = color if color != None else np.array([1, 1, 1])

        vertices = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], 'float32')

        vertices = normalized(vertices)

        colors = np.array([self.color]*12, 'float32')

        normals = np.array([[-1, t, 0], [1., t, 0], [-1, -t, 0], [1, -t, 0],
                           [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                           [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], 'float32')

        indices = [0, 11, 5,   0, 5, 1,   0, 1, 7,   0, 7, 10,   0, 10, 11,   1, 5, 9,   5, 11, 4,   11, 10, 2,
                   10, 7, 6,   7, 1, 8,   3, 9, 4,   3, 4, 2,    3, 2, 6,   3, 6, 8,   3, 8, 9,   4, 9, 5,
                   2, 4, 11,   6, 2, 10,  8, 6, 7,   9, 8, 1]

        # Subdivide
        next_index = 12
        for i in range(subdivisions):
            new_indices = []
            for j in range(0, len(indices), 3):
                # Create three new vertices
                tmp1 = (vertices[indices[j],:] + vertices[indices[j+1],:])
                tmp2 = (vertices[indices[j+1],:] + vertices[indices[j+2],:])
                tmp3 = (vertices[indices[j+2],:] + vertices[indices[j],:])
                vertices = np.append(vertices, normalized(np.array([tmp1, tmp2, tmp3], 'float32')), axis=0)
                colors = np.append(colors, np.array([self.color]*3, 'float32'), axis=0)
                normals = np.append(normals, np.array([tmp1, tmp2, tmp3],  'float32'), axis=0)

                # Add 4 new faces
                new_indices.extend([indices[j], next_index, next_index+2])
                new_indices.extend([next_index, next_index + 1, next_index + 2])
                new_indices.extend([next_index, indices[j+1], next_index + 1])
                new_indices.extend([next_index + 1, indices[j+2], next_index + 2])
                next_index += 3
            indices = new_indices

        self.vertices = vbo.VBO(vertices, usage=GL_STATIC_DRAW)
        self.colors = vbo.VBO(colors, usage=GL_STATIC_DRAW)
        self.normals = vbo.VBO(normals, usage=GL_STATIC_DRAW)
        self.indices = indices


class Camera:

    def __init__(self, position=None, rotation=None):

        self.position = position if isinstance(position, np.ndarray) else np.array([0, 0, 0], 'float')

        # The XYZ rotation values
        # Initially, the camera looks in the -Z direction, and up in the Y direction (GL convention)
        self.rotation = rotation if isinstance(rotation, np.ndarray) else np.array([0, 0, 0], 'float')

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
        self.window = SDL_CreateWindow("OpenGL", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       width, height, SDL_WINDOW_OPENGL)
        SDL_GL_CreateContext(self.window)
        self.width = width
        self.height = height

        self.init_shader()
        self.init_framerate()
        self.init_frustum()

        self.meshes = []
        self.camera = Camera(position=np.array([0, 0, 10.0]))
        self.lighting_direction = 0

        glClearColor(0.6, 0.6, 0.7, 1.0)

    def addmesh(self, mesh):
        self.meshes.append(mesh)

    def init_framerate(self):
        self.tStart = self.t0 = time.time()
        self.frames = 0

    def init_shader(self):

        with open("vertex-shader.glsl") as f:
            vertex = shaders.compileShader(f.read(), GL_VERTEX_SHADER)
        with open("fragment-shader.glsl") as f:
            fragment = shaders.compileShader(f.read(), GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vertex, fragment)

        self.attributes = ['position', 'color', 'normal']
        self.attribute_locations = {}
        for attribute in self.attributes:
            location = glGetAttribLocation(self.shader, attribute)
            if location in (None, -1):
                raise Exception("Could not get location of attribute %s" % attribute)
            self.attribute_locations[attribute] = location

        self.uniforms = ['transform', 'light_direction']
        self.uniform_locations = {}
        for uniform in self.uniforms:
            location = glGetUniformLocation(self.shader, uniform)
            if location in (None, -1):
                raise Exception("Could not get location of uniform %s" % uniform)
            self.uniform_locations[uniform] = location

        self.locations = {}
        self.locations.update(self.attribute_locations)
        self.locations.update(self.uniform_locations)

    def init_frustum(self):
        self.near, self.far = 1.0, 20.0
        self.left, self.right = -1.0, 1.0
        self.top, self.bottom = 1.0, -1.0

    def draw(self):
        self.framerate()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        shaders.glUseProgram(self.shader)

        # put up uniforms
        transform = np.dot(self.get_ProjectionMatrix(), self.camera.viewmatrix)
        glUniformMatrix4fv(self.locations['transform'], 1, GL_TRUE, transform)
        glUniform3f(self.locations['light_direction'], math.sin(self.lighting_direction) *5, 0, -1)

        # Draw the meshes
        for mesh in self.meshes:
            mesh.draw(self.locations)

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

    def get_ProjectionMatrix(self):
        # See http://www.songho.ca/opengl/gl_projectionmatrix.html
        # Or 'Red Book' chapter 5
        return np.array([[self.near / self.right, 0, 0, 0],
                       [0, self.near/self.top, 0 , 0],
                       [0, 0, -(self.far+self.near)/(self.far-self.near), -2*self.far*self.near/(self.far - self.near)],
                       [0, 0, -1, 0]])

    def keyPressed(self, event):

        if event.key.keysym.sym == SDLK_ESCAPE:
            sys.exit()

        if event.key.keysym.sym == SDLK_UP:
            self.camera.move_forward(0.1)

        if event.key.keysym.sym == SDLK_DOWN:
            self.camera.move_backward(0.1)

        if event.key.keysym.sym == SDLK_LEFT:
            self.camera.rotation[1] += 0.05

        if event.key.keysym.sym == SDLK_RIGHT:
            self.camera.rotation[1] -= 0.05

        if event.key.keysym.sym == SDLK_y:
            self.lighting_direction += 0.05

        # Screenshot
        if event.key.keysym.sym == SDLK_s:
            buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=buffer)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            with open("test.png", 'w') as f:
                image.save(f)
            print "took screenshot"

        if event.key.keysym.sym == SDLK_d:
             print "Debug info"


    def reshape(self, width, height):
        glViewport(0, 0, width, height)
        aspect = float(height) / width

        self.top, self.bottom = aspect, -aspect
        self.width, self.height = width, height

    @staticmethod
    def print_information():
        print "GL_RENDERER   = ", glGetString(GL_RENDERER)
        print "GL_VERSION    = ", glGetString(GL_VERSION)
        print "GL_VENDOR     = ", glGetString(GL_VENDOR)
        print "GL_EXTENSIONS = ", glGetString(GL_EXTENSIONS)


if __name__ == '__main__':
    SDL_Init(SDL_INIT_EVERYTHING)
    SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8)
    SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8)
    SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8)
    SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8)
    SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32)
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16)
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1)

    # Create the context with a sphere in it
    c = Context(width=600, height=600)
    c.addmesh(Icosphere())
    c.print_information()

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glEnable(GL_MULTISAMPLE)
    glCullFace(GL_BACK)

    # The Loop
    running = True
    event = SDL_Event()
    while running:
        while SDL_PollEvent(ctypes.byref(event)) != 0:
            if event.type == SDL_QUIT:
                running = False
            if event.type == events.SDL_KEYDOWN:
                c.keyPressed(event)
            if (event.type == SDL_MOUSEMOTION):
                pass
            if (event.type == SDL_MOUSEBUTTONDOWN):
                pass
        c.draw()

