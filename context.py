#!/usr/bin/env python
# Author: Johannes

from OpenGL.GL import *
from sdl2 import *
from OpenGL.GL import shaders as glshaders
from OpenGL.arrays import vbo
from OpenGL.GL.framebufferobjects import *
from PIL import Image
import numpy as np
import sys
import time
import math
import ctypes

import shaders, meshes, cameras, materials, scenes

class Context:
    def __init__(self, width, height, show_framerate=True):

        self.width = width
        self.height = height
        self.show_framerate = show_framerate

        self.init_sdl()
        self.init_gl()
        self.init_framerate()

        self.shaderlib = shaders.ShaderLib()

    def render(self, scene, camera):
        self._render(scene, camera)
        SDL_GL_SwapWindow(self.window)
        return self.sdl_control(scene, camera)

    def _render(self, scene, camera):

        glClearColor(*scene.backgroundcolor)
        self.framerate()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)


        for mesh in scene.meshes:

            # Get the program of the mesh's material and bind to opengl
            shader = None
            if isinstance(mesh.material, materials.AmbientMaterial):
                shader = self.shaderlib.ambient
            elif isinstance(mesh.material, materials.DepthMaterial):
                shader = self.shaderlib.depth
            elif isinstance(mesh.material, materials.NormalMaterial):
                shader = self.shaderlib.normal
            elif isinstance(mesh.material, materials.LambertianMaterial):
                shader = self.shaderlib.lambertian

            glUseProgram(shader.program)

            # Get all lights and bind them up the program shader
            for number, light in enumerate(scene.lights):
                light.put_up_uniforms(shader, number)

            # Put up material uniforms and textures
            mesh.material.put_up_uniforms(shader)
            mesh.material.put_up_textures(shader)

            # Put up camera uniforms
            glUniformMatrix4fv(shader.locations['MMatrix'], 1, GL_TRUE, np.eye(4))
            glUniformMatrix4fv(shader.locations['VMatrix'], 1, GL_TRUE, camera.viewmatrix)
            glUniformMatrix4fv(shader.locations['PMatrix'], 1, GL_TRUE, camera.projectionmatrix)

            if mesh.visible:
                        try:
                            # IDEA: use a Buffer Object to switch uniforms faster!
                            if mesh.modelmatrix != None:
                                glUniformMatrix4fv(shader.locations['MMatrix'], 1, GL_TRUE, mesh.modelmatrix)
                            else:
                                print "modelmatrix not found"

                            if mesh.geometry.vertices and shader.locations['position'] != -1:
                                glEnableVertexAttribArray(shader.locations['position'])
                                mesh.geometry.vertices.bind()
                                glVertexAttribPointer(shader.locations['position'], 3, GL_FLOAT, False, 3*4, mesh.geometry.vertices)
                            if mesh.geometry.colors and shader.locations['color'] != -1:
                                glEnableVertexAttribArray(shader.locations['color'])
                                mesh.geometry.colors.bind()
                                glVertexAttribPointer(shader.locations['color'], 3, GL_FLOAT, False, 3*4, mesh.geometry.colors)
                            if mesh.geometry.texcoords and shader.locations['texcoords'] != -1:
                                glEnableVertexAttribArray(shader.locations['texcoords'])
                                mesh.geometry.texcoords.bind()
                                glVertexAttribPointer(shader.locations['texcoords'], 2, GL_FLOAT, False, 2*4, mesh.geometry.texcoords)
                            if mesh.geometry.normals and shader.locations['normal'] != -1:
                                glEnableVertexAttribArray(shader.locations['normal'])
                                mesh.geometry.normals.bind()
                                glVertexAttribPointer(shader.locations['normal'], 3, GL_FLOAT, False, 3*4, mesh.geometry.normals)

                            if mesh.geometry.indices:
                                glDrawElements(GL_TRIANGLES, len(mesh.geometry.indices), GL_UNSIGNED_SHORT, mesh.geometry.indices)
                            elif mesh.geometry.vertices:
                                numpoints = mesh.geometry.vertices.data.shape[0]
                                glDrawArrays(GL_POINTS, 0, numpoints)

                        finally:
                            glUniformMatrix4fv(shader.locations['MMatrix'], 1, GL_TRUE, np.eye(4))
                            if mesh.geometry.vertices and shader.locations['position'] != -1:
                                mesh.geometry.vertices.unbind()
                                glDisableVertexAttribArray(shader.locations['position'])
                            if mesh.geometry.colors and shader.locations['color'] != -1:
                                mesh.geometry.colors.unbind()
                                glDisableVertexAttribArray(shader.locations['color'])
                            if mesh.geometry.texcoords and shader.locations['texcoords'] != -1:
                                mesh.geometry.texcoords.unbind()
                                glDisableVertexAttribArray(shader.locations['texcoords'])
                            if mesh.geometry.normals and shader.locations['normal'] != -1:
                                mesh.geometry.normals.unbind()
                                glDisableVertexAttribArray(shader.locations['normal'])

    def init_framerate(self):
        self.tStart = self.t0 = time.time()
        self.frames = 0

    def init_gl(self):

        # GL Properties
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

    def current_buffer(self):
        ''' Returns the content of the current buffer as a numpy array '''

        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

        return np.array(image)

    def render_to_buffer(self, scene, camera):
        ''' Renders to framebuffer.  '''

        # TODO renderbuffer/framebuffer does not depth test
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self._render(scene, camera)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def framerate(self, updaterate=1.0):
        t = time.time()
        self.frames += 1
        if t - self.t0 >= updaterate:
            seconds = t - self.t0
            fps = self.frames / seconds
            self.t0 = t
            self.frames = 0
            SDL_GL_SwapWindow(self.window)

            if self.show_framerate:
                print "%.0f frames in %3.1f seconds = %6.3f FPS" % (self.frames, seconds, fps)

    def screenshot(self, scene, camera, filename):
        """
        Takes a screenshot of the current view.

        :param mode: 'color', 'depth', or 'normal'
        :type mode: String
        :param filename: Filename where screenshot is saved
        :type filename: String
        :return: None
        """

        # Draw to renderbuffer
        # TODO bind to framebuffer when renderbuffer does correct depth testing
        #glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self._render(scene, camera)

        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes(mode="RGB", size=(self.width, self.height), data=buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        with open(filename, 'w') as f:
            image.save(f)

        print "saved screenshot as '%s'" % (filename)

        # Clean up
        #glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def keyPressed(self, event, scene, camera):
        ''' Handles the event when a key is pressed. '''

        if event.key.keysym.sym == SDLK_ESCAPE:
            sys.exit()

        if event.key.keysym.sym == SDLK_UP or event.key.keysym.sym == SDLK_w:
            camera.move_forward(0.1)

        if event.key.keysym.sym == SDLK_DOWN or event.key.keysym.sym == SDLK_s:
            camera.move_backward(0.1)

        if  event.key.keysym.sym == SDLK_a:
            camera.move_left(0.1)

        if event.key.keysym.sym == SDLK_d:
            camera.move_right(0.1)

        if event.key.keysym.sym == SDLK_LEFT  or event.key.keysym.sym == SDLK_q:
            camera.rotation[1] += 0.05

        if event.key.keysym.sym == SDLK_RIGHT  or event.key.keysym.sym == SDLK_e:
            camera.rotation[1] -= 0.05

        if event.key.keysym.sym == SDLK_y:
            scene.lights[0].uniforms['direction'][0] = scene.lights[0].uniforms['direction'][0] + np.array([0.05, 0, 0])
            print scene.lights[0].uniforms['direction'][0]

        if event.key.keysym.sym == SDLK_t:
            print scene.meshes[-1]
            scene.meshes[-1].toggle_visibility()

        if event.key.keysym.sym == SDLK_r:
            for mesh in scene.meshes:
                mesh.material = materials.LambertianMaterial()
        if event.key.keysym.sym == SDLK_n:
            for mesh in scene.meshes:
                mesh.material = materials.DepthMaterial()
        if event.key.keysym.sym == SDLK_v:
            for mesh in scene.meshes:
                mesh.material = materials.NormalMaterial()
        # Screenshots
        if event.key.keysym.sym == SDLK_f:
            self.screenshot(scene, camera, 'images/screen_color.png')

        if event.key.keysym.sym == SDLK_p:
            print "Debug info (Key g)"
            print ""

    def resize_event(self, event, camera):
        ''' Handles a resize event. Event: SDL-event data. '''
        self.reshape(event.window.data1, event.window.data2)
        camera.reshape(event.window.data1, event.window.data2)

    def reshape(self, width, height):
        ''' Resize Width and height in pixels. '''
        self.width, self.height = width, height

        glViewport(0, 0, width, height)

        # Resize Renderbuffer object
        glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB, self.width, self.height)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)

    def sdl_control(self, scene, camera):
        ''' Check for window events '''
        while SDL_PollEvent(ctypes.byref(self.event)) != 0:

            if self.event.type == SDL_QUIT:
                return False
            if self.event.type == events.SDL_KEYDOWN:
                self.keyPressed(self.event, scene, camera)
            if self.event.type == SDL_MOUSEMOTION:
                pass
            if self.event.type == SDL_MOUSEBUTTONDOWN:
                pass
            if self.event.type == SDL_WINDOWEVENT and self.event.window.event == SDL_WINDOWEVENT_RESIZED:
                self.resize_event(self.event, camera)
        return True

    @staticmethod
    def print_opengl_info():
        ''' Print out information about the OpenGL version. '''
        print "GL_RENDERER   = ", glGetString(GL_RENDERER)
        print "GL_VERSION    = ", glGetString(GL_VERSION)
        print "GL_VENDOR     = ", glGetString(GL_VENDOR)
        print "GL_EXTENSIONS = ", glGetString(GL_EXTENSIONS)


if __name__ == '__main__':

    # Create the context
    c = Context(width=500, height=500)
    c.print_opengl_info()

    scene = scenes.exampleScene4()
    camera = cameras.PerspectiveCamera()

    # The Loop
    running = True
    while running:
        running = c.render(scene, camera)
