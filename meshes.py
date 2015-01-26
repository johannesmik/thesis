__author__ = 'johannes'

from OpenGL.GL import *
from OpenGL.arrays import vbo
from PIL import Image
import numpy as np
import math

def normalized(array, axis=1):
    n = np.linalg.norm(array, axis=1)
    n[n==0] = 1
    return  array / np.expand_dims(n, 1)

class Renderable:

    def __init__(self):
        self.visible = True

    def toggle_visibility(self):
        if self.visible:
            self.visible = False
        else:
            self.visible = True

class Mesh(Renderable):
    """ Base class for all meshes. """
    def __init__(self, position=None):
        if position == None:
            self.position = np.array([0, 0, 0])
        else:
            self.position = position

        Renderable.__init__(self)

    def draw(self, locations):
        if not self.visible:
            return None
        stride = 12  # 3 * 4 bits stride
        try:
            # TODO use a Buffer Object to switch uniforms faster!
            glUniformMatrix4fv(locations['MMatrix'], 1, GL_TRUE, self.modelmatrix)
            if locations['position'] != -1:
                glEnableVertexAttribArray(locations['position'])
                self.vertices.bind()
                glVertexAttribPointer(locations['position'], 3, GL_FLOAT, False, stride, self.vertices)
            if locations['color'] != -1:
                glEnableVertexAttribArray(locations['color'])
                self.colors.bind()
                glVertexAttribPointer(locations['color'], 3, GL_FLOAT, False, stride, self.colors)
            if locations['texcoords'] != -1:
                glEnableVertexAttribArray(locations['texcoords'])
                self.texcoords.bind()
                glVertexAttribPointer(locations['texcoords'], 2, GL_FLOAT, False, 2*4, self.texcoords)
            if locations['normal'] != -1:
                glEnableVertexAttribArray(locations['normal'])
                self.normals.bind()
                glVertexAttribPointer(locations['normal'], 3, GL_FLOAT, False, stride, self.normals)

            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_SHORT, self.indices)

        finally:
            glUniformMatrix4fv(locations['MMatrix'], 1, GL_TRUE, np.eye(4))
            if locations['position'] != -1:
                self.vertices.unbind()
                glDisableVertexAttribArray(locations['position'])
            if locations['color'] != -1:
                self.colors.unbind()
                glDisableVertexAttribArray(locations['color'])
            if locations['texcoords'] != -1:
                self.texcoords.unbind()
                glDisableVertexAttribArray(locations['texcoords'])
            if locations['normal'] != -1:
                self.normals.unbind()
                glDisableVertexAttribArray(locations['normal'])

        return 0

    @property
    def modelmatrix(self):
        tmp = np.eye(4)
        tmp[0:3,3] = self.position
        return tmp


class Triangle(Mesh):
    """ Create a small triangle """
    def __init__(self, position=None):
        Mesh.__init__(self, position)
        self.vertices = vbo.VBO(np.array([[0., 1., 0], [-1., -1., 0], [1., -1., 0]], 'float32'),
                                usage=GL_STATIC_DRAW)

        self.colors = vbo.VBO(np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], 'float32'),
                              usage=GL_STATIC_DRAW)

        self.texcoords = vbo.VBO(np.array([[0, 0]]*3, 'float32'),
                                 usage=GL_STATIC_DRAW)

        self.normals = vbo.VBO(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], 'float32'),
                               usage=GL_STATIC_DRAW)

        self.indices = [0, 1, 2]

class Square(Mesh):
    """ A simple textured square """
    def __init__(self, position=None):
        Mesh.__init__(self, position)
        self.vertices = vbo.VBO(np.array([[1., 1., 0], [-1., 1., 0], [-1., -1., 0], [1., -1., 0.]], 'float32'),
                                usage=GL_STATIC_DRAW)

        self.colors = vbo.VBO(np.array([[0, 0, 0]]*4, 'float32'),
                              usage=GL_STATIC_DRAW)

        self.texcoords = vbo.VBO(np.array([[1, 1], [0, 1], [0, 0], [1, 0]], 'float32'),
                              usage=GL_STATIC_DRAW)

        self.normals = vbo.VBO(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], 'float32'),
                               usage=GL_STATIC_DRAW)

        self.indices = [0, 1, 2, 0, 2, 3]

class Pointcloud(Renderable):
    """ Base class for Pointcloud like meshes. """
    def __init__(self, position=None):
        if position == None:
            self.position = np.array([0, 0, 0])
        else:
            self.position = position

        Renderable.__init__(self)

    def draw(self, locations):
        if not self.visible:
            return None
        stride = 12  # 3 * 4 bits stride
        try:
            # TODO use a Buffer Object to switch uniforms faster!
            glUniformMatrix4fv(locations['MMatrix'], 1, GL_TRUE, self.modelmatrix)
            if locations['position'] != -1:
                glEnableVertexAttribArray(locations['position'])
                self.vertices.bind()
                glVertexAttribPointer(locations['position'], 3, GL_FLOAT, False, stride, self.vertices)
            if locations['color'] != -1:
                glEnableVertexAttribArray(locations['color'])
                self.colors.bind()
                glVertexAttribPointer(locations['color'], 3, GL_FLOAT, False, stride, self.colors)
            if locations['normal'] != -1:
                glEnableVertexAttribArray(locations['normal'])
                self.normals.bind()
                glVertexAttribPointer(locations['normal'], 3, GL_FLOAT, False, stride, self.normals)


            glDrawArrays(GL_POINTS, 0, self.numpoints)

        finally:
            glUniformMatrix4fv(locations['MMatrix'], 1, GL_TRUE, np.eye(4))
            if locations['position'] != -1:
                self.vertices.unbind()
                glDisableVertexAttribArray(locations['position'])
            if locations['color'] != -1:
                self.colors.unbind()
                glDisableVertexAttribArray(locations['color'])
            if locations['normal'] != -1:
                self.normals.unbind()
                glDisableVertexAttribArray(locations['normal'])

        return 0

    @property
    def modelmatrix(self):
        tmp = np.eye(4)
        tmp[0:3,3] = self.position
        return tmp

class Depthmap(Pointcloud):
    def __init__(self, position=None):
        Pointcloud.__init__(self, position)
        # Read in Depth image
        depth_img = Image.open("screen_depth.png")
        depth_array = np.asarray(depth_img, dtype=np.float32)
        height, width, num_colors = depth_array.shape
        self.numpoints = height * width

        n, f = 1.0, 20
        l, r = -1.0, 1.0
        t, b = 1.0, -1.0

        normals = np.zeros((height, width, num_colors), dtype='float32')

        for i in range(height):
            for j in range(width):
                c = depth_array[i,j]
                # Project each pixel to ndc
                ndc = [ 2*j/float(width) - 1, 2*i/float(height) - 1, 2* (1-  depth_array[i,j,0]/255.) -1]
                # Project ndc to eye space
                zeye = (2*f*n) / (ndc[2]*(f-n)-(f+n))
                xeye = -zeye*(ndc[0]*(r-l)+(r+l))/(2.0*n)
                yeye = -zeye*(ndc[1]*(t-b)+(t+b))/(2.0*n)
                # Project into World space (Quick hack for known camera position)
                xworld = xeye
                yworld= yeye
                zworld = zeye + 2.00
                depth_array[i, j] = [xworld, yworld, zworld]

        for i in range(height):
            for j in range(width):
                if 1 <= i < height-1 and 1 <= j < width-1:
                    # Horizonalt-Vertical and Diagonal vectors
                    v1 = depth_array[i+1, j] - depth_array[i-1, j]
                    v2 = depth_array[i, j-1] - depth_array[i, j+1]
                    v3 = depth_array[i+1, j-1] - depth_array[i-1, j+1]
                    v4 = depth_array[i-1, j-1] - depth_array[i+1, j+1]

                    # Normalize those vectors
                    v1 = v1/np.linalg.norm(v1)
                    v2 = v2/np.linalg.norm(v2)
                    v3 = v3/np.linalg.norm(v3)
                    v4 = v4/np.linalg.norm(v4)

                    # Calculate norm
                    normals[i, j] = np.cross(v1, v2) + np.cross(v3, v4)

        self.vertices = vbo.VBO(np.reshape(depth_array, (width*height, 3)), usage=GL_STATIC_DRAW)

        self.colors = vbo.VBO(np.array([[0,1,0]]*self.numpoints, 'float32'), usage=GL_STATIC_DRAW)

        self.normals = vbo.VBO(np.reshape(normals, (width*height, 3)), usage=GL_STATIC_DRAW)

class Icosphere(Mesh):
    """ Create a mesh for an Icosphere """
    def __init__(self, subdivisions=3, color=None, position=None):
        Mesh.__init__(self, position)
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

        texcoords = np.array([[0, 0]]*vertices.shape[0], 'float32')

        self.vertices = vbo.VBO(vertices, usage=GL_STATIC_DRAW)
        self.colors = vbo.VBO(colors, usage=GL_STATIC_DRAW)
        self.texcoords = vbo.VBO(texcoords, usage=GL_STATIC_DRAW)
        self.normals = vbo.VBO(normals, usage=GL_STATIC_DRAW)
        self.indices = indices