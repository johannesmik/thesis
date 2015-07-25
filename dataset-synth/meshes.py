__author__ = 'johannes'

from OpenGL.GL import *
from OpenGL.arrays import vbo
from PIL import Image
import pcl
import numpy as np
import math
import core

# TODO clean up this file


def normalized(array):
    n = np.linalg.norm(array, axis=1)
    n[n == 0] = 1
    return array / np.expand_dims(n, 1)


class Mesh(core.Object3D):
    # TODO cache matrices
    def __init__(self, geometry=None, material=None, name=None, position=None, rotation=None):
        super(Mesh, self).__init__(name=name, position=position, rotation=rotation)

        # To be initialized in the subclasses
        self.geometry = geometry
        self.material = material

    def draw(self):
        # Todo Implement this
        pass


class PointCloud(core.Object3D):
    def __init__(self, geometry=None, material=None, name=None, position=None, rotation=None):
        super(PointCloud, self).__init__(name=name, position=position, rotation=rotation)

        self.geometry = geometry
        self.material = material

    def draw(self):
        # Todo implement this
        pass


class Geometry(object):
    def __init__(self):
        # To be initialized in the subclasses

        # Vertices, normals texcoords are all VBOs
        # Indices is a list
        self.vertices = None
        self.normals = None
        self.texcoords = None
        self.indices = None
        self.material = None


class ObjectFile(Geometry):
    def __init__(self, filename):
        """
         Loads an obj file. Currently works with files exported from blender (check 'export normal' setting as well)
         Vertices are not 'reused' here, that means 1 vertex belongs to exactly 1 face. Normally, vertices are reused,
         often one vertex belongs to one or more faces. We don't do this here because the obj file generally doesn't
         describe one normal per vertice, but instead one normal per vector per face, So the normal on the vertex can
         be different, depending on which face it belongs to.

         Overwrites vertices, texcoords, normals, indices

         Adapted from http://www.nandnor.net/?p=86
        """
        super(ObjectFile, self).__init__()

        verts, texcoords, norms = [], [], []
        verts_out, texcoords_out, norms_out = [], [], []
        indices = []

        for line in open(filename, "r"):
            vals = line.split()
            if vals[0] == "v":
                verts.append([float(v) for v in vals[1:4]])
            if vals[0] == "vn":
                norms.append([float(v) for v in vals[1:4]])
            if vals[0] == "vt":
                texcoords.append([float(v) for v in vals[1:3]])
            if vals[0] == "f":
                current_index = len(verts_out)
                # Triangles
                if len(vals) == 4:
                    for f in vals[1:]:
                        w = f.split("/")
                        if len(w) > 0 and w[0] is not '':
                            verts_out.append(list(verts[int(w[0]) - 1]))
                        if len(w) > 1 and w[1] is not '':
                            texcoords_out.append(list(texcoords[int(w[1]) - 1]))
                        if len(w) > 2 and w[2] is not '':
                            norms_out.append(list(norms[int(w[2]) - 1]))
                    indices.extend([current_index + i for i in [0, 1, 2]])
                # Quads
                elif len(vals) == 5:
                    for f in vals[1:]:
                        w = f.split("/")
                        if len(w) > 0 and w[0] is not '':
                            verts_out.append(list(verts[int(w[0]) - 1]))
                        if len(w) > 1 and w[1] is not '':
                            texcoords_out.append(list(texcoords[int(w[1]) - 1]))
                        if len(w) > 2 and w[2] is not '':
                            norms_out.append(list(norms[int(w[2]) - 1]))
                    indices.extend([current_index + i for i in [0, 1, 2, 0, 2, 3]])  # Appends the quad as two

        self.vertices = vbo.VBO(np.array(verts_out, 'float32'), usage=GL_STATIC_DRAW)
        self.texcoords = vbo.VBO(np.array(texcoords_out, 'float32'), usage=GL_STATIC_DRAW)
        self.normals = vbo.VBO(np.array(norms_out, 'float32'), usage=GL_STATIC_DRAW)
        self.indices = indices


class TriangleGeometry(Geometry):
    """ Create a small triangle """

    def __init__(self):
        super(TriangleGeometry, self).__init__()

        self.vertices = vbo.VBO(np.array([[0., 1., 0], [-1., -1., 0], [1., -1., 0]], 'float32'),
                                usage=GL_STATIC_DRAW)

        self.texcoords = vbo.VBO(np.array([[0, 0], [0.5, 1], [1, 0]], 'float32'),
                                 usage=GL_STATIC_DRAW)

        self.normals = vbo.VBO(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], 'float32'),
                               usage=GL_STATIC_DRAW)

        self.indices = [0, 1, 2]


class BoxGeometry(Geometry):
    """ Create a Box mesh """

    def __init__(self):
        super(BoxGeometry, self).__init__()

        self.vertices = vbo.VBO(np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1],
                                          [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, -1]] * 6,
                                          'float32'), usage=GL_STATIC_DRAW)

        self.texcoords = vbo.VBO(np.array([[0, 0], [0.5, 1], [1, 0]], 'float32'),
                                 usage=GL_STATIC_DRAW)

        normal_directions = [[0, 0, 1], [1, 0, 0], [0, 0, -1], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        normals = [[normal_direction] * 8 for normal_direction in normal_directions]

        self.normals = vbo.VBO(np.array(normals, 'float32'),
                               usage=GL_STATIC_DRAW)

        indices = [0, 1, 2, 0, 2, 3, 4, 0, 3, 4, 3, 7, 5, 4, 7, 5, 7, 6, 1, 5, 6, 1, 6, 2, 4, 5, 1, 4, 1, 0, 6, 7, 3, 6, 3, 2]
        self.indices = [offset + index for offset in range(6) for index in indices]


class IcosphereGeometry(Geometry):
    """ Icosphere """

    def __init__(self, subdivisions=2):
        super(IcosphereGeometry, self).__init__()
        t = (1 + math.sqrt(5.0)) / 2.0

        vertices = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
                             [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                             [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], 'float32')

        vertices = normalized(vertices)

        normals = np.array([[-1, t, 0], [1., t, 0], [-1, -t, 0], [1, -t, 0],
                            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], 'float32')

        indices = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2,
                   10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5,
                   2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]

        # Subdivide
        next_index = 12
        for i in range(subdivisions):
            new_indices = []
            for j in range(0, len(indices), 3):
                # Create three new vertices
                tmp1 = (vertices[indices[j], :] + vertices[indices[j + 1], :])
                tmp2 = (vertices[indices[j + 1], :] + vertices[indices[j + 2], :])
                tmp3 = (vertices[indices[j + 2], :] + vertices[indices[j], :])
                vertices = np.append(vertices, normalized(np.array([tmp1, tmp2, tmp3], 'float32')), axis=0)
                normals = np.append(normals, np.array([tmp1, tmp2, tmp3], 'float32'), axis=0)

                # Add 4 new faces
                new_indices.extend([indices[j], next_index, next_index + 2])
                new_indices.extend([next_index, next_index + 1, next_index + 2])
                new_indices.extend([next_index, indices[j + 1], next_index + 1])
                new_indices.extend([next_index + 1, indices[j + 2], next_index + 2])
                next_index += 3
            indices = new_indices

        texcoords = np.array([[0, 0]] * vertices.shape[0], 'float32')

        self.vertices = vbo.VBO(vertices, usage=GL_STATIC_DRAW)
        self.texcoords = vbo.VBO(texcoords, usage=GL_STATIC_DRAW)
        self.normals = vbo.VBO(normals, usage=GL_STATIC_DRAW)
        self.indices = indices


class SquareGeometry(Geometry):
    """ A square, with texcoords """

    def __init__(self):
        Geometry.__init__(self)
        self.vertices = vbo.VBO(np.array([[1., 1., 0], [-1., 1., 0], [-1., -1., 0], [1., -1., 0.]], 'float32'),
                                usage=GL_STATIC_DRAW)

        self.texcoords = vbo.VBO(np.array([[1, 1], [0, 1], [0, 0], [1, 0]], 'float32'),
                                 usage=GL_STATIC_DRAW)

        self.normals = vbo.VBO(np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], 'float32'),
                               usage=GL_STATIC_DRAW)

        self.indices = [0, 1, 2, 0, 2, 3]


class WaveGeometry(Geometry):
    """ A wavy plane """

    def __init__(self, periodicity=(2*np.pi), hscale=1, xvertices=100, yvertices=100):
        Geometry.__init__(self)
        x = np.linspace(-1, 1, xvertices)
        y = np.linspace(-1, 1, yvertices)
        s = (2.0 * np.pi) / periodicity

        vertices = [[i, j, hscale * math.cos(s * math.sqrt(i**2 + j**2))] for j in y for i in x]
        normals = [[hscale * s * math.sin(s * math.sqrt(i**2 + j**2)) * np.cos(np.arctan2(j, i)),
                    hscale * s * math.sin(s * math.sqrt(i**2 + j**2)) * np.sin(np.arctan2(j, i)),
                    1.0] for j in y for i in x]

        indices = []
        for i in range(xvertices - 1):
            for j in range(yvertices - 1):
                offset = j * xvertices
                indices.extend([offset + i, offset + xvertices + i, offset + i + 1][::-1])
                indices.extend([offset + i + 1, offset + xvertices + i, offset + xvertices + i + 1][::-1])

        self.vertices = vbo.VBO(np.array(vertices, 'float32'), usage=GL_STATIC_DRAW)
        self.texcoords = vbo.VBO(np.array([[0, 0]] * xvertices * yvertices, 'float32'), usage=GL_STATIC_DRAW)
        self.normals = vbo.VBO(np.array(normals, 'float32'), usage=GL_STATIC_DRAW)
        self.indices = indices


class Pointcloud(Geometry):
    """ Base class for Pointcloud like meshes. """

    def __init__(self, vertices=None, position=None):

        Geometry.__init__(self)
        if vertices is None:
            self.vertices = None
        else:
            self.vertices = vbo.VBO(vertices.astype('float32'), usage=GL_STATIC_DRAW)

        if position is None:
            self.position = np.array([0, 0, 0])
        else:
            self.position = position

    @property
    def modelmatrix(self):
        tmp = np.eye(4)
        tmp[0:3, 3] = self.position
        return tmp


class DepthmapGeometry(Pointcloud):
    # Recover the geometry from a Depthmap
    # TODO: universal camera. At the moment we assume certain intrinsic parameters here
    def __init__(self, filename, position=None):
        self.file = filename
        Pointcloud.__init__(self, position)

        # Read in Depth image
        depth_img = Image.open(self.file)
        depth_array = np.asarray(depth_img, dtype=np.float32)
        print "shape of depth array", depth_array.shape
        height, width, num_colors = depth_array.shape
        self.height, self.width = height, width
        self.numpoints = height * width

        # Intrinsic parameters
        # TODO have to be determined for every depthmap
        n, f = 1.0, 2
        l, r = -1.0, 1.0
        t, b = 1.0, -1.0

        normals = np.zeros((height, width, num_colors), dtype='float32')

        for i in range(height):
            for j in range(width):
                c = depth_array[i, j]
                # Project each pixel to ndc
                xndc = j / float(width - 1) * 2 - 1
                yndc = -i / float(height - 1) * 2 + 1
                zndc = 2 * (depth_array[i, j, 0] / 255.) - 1
                # Project ndc to eye space
                zeye = 2 * f * n / (zndc * (f - n) - (f + n))
                xeye = -zeye * (xndc * (r - l) + (r + l)) / (2.0 * n)
                yeye = -zeye * (yndc * (t - b) + (t + b)) / (2.0 * n)
                depth_array[i, j] = [xeye, yeye, zeye]

        # Another reprojection
        # xndc, yndc = np.meshgrid(np.linspace(-1, 1, width), np.linspace(1, -1, height))
        # zndc = 2 * (depth_array[:,:, 0] / 255.) - 1
        # zeye = 2 * f * n / (zndc*(f-n)-(f+n))
        # print "zeye shape", zeye.shape
        # print "xndc shape", xndc.shape
        # print "yndc shape", yndc.shape
        # xeye = -zeye*(xndc*(r-l)+(r+l))/(2.0*n)
        # yeye = -zeye*(yndc*(t-b)+(t+b))/(2.0*n)
        # a = np.dstack((xeye, yeye, zeye))
        # a = np.reshape(a, (width*height, 3))
        # a = a.astype(np.float32)

        # Project depth into world coordinates (Quick hack for known camera position (0,0,0))
        depth_array += np.array([0.0, 0.0, 0.0])

        self.vertices = vbo.VBO(np.reshape(depth_array, (width * height, 3)), usage=GL_STATIC_DRAW)
        # self.vertices = vbo.VBO(a, usage=GL_STATIC_DRAW)

        self.colors = vbo.VBO(np.array([[0, 1, 0]] * self.numpoints, 'float32'), usage=GL_STATIC_DRAW)

        self.calculate_normals_pcl()
        # self.calculate_normals_1(depth_array, width, height)

    def calculate_normals_pcl(self):
        """ Use PCL to calculate normals, use a standard environment of 100 neighbours """

        pointcloud = pcl.PointCloud()
        pointcloud.from_array(self.vertices.data)
        normals = pointcloud.calc_normals(ksearch=30)
        self.normals = vbo.VBO(normals, usage=GL_STATIC_DRAW)

    def calculate_normals_1(self, depth_array, width, height):
        """ One (slow) way to calculate normals """
        normals = np.zeros((height, width, 3), dtype='float32')
        for i in range(height):
            for j in range(width):
                if 1 <= i < height - 1 and 1 <= j < width - 1:
                    # Horizontal-Vertical and Diagonal vectors
                    v1 = depth_array[i + 1, j] - depth_array[i - 1, j]
                    v2 = depth_array[i, j - 1] - depth_array[i, j + 1]
                    v3 = depth_array[i + 1, j - 1] - depth_array[i - 1, j + 1]
                    v4 = depth_array[i - 1, j - 1] - depth_array[i + 1, j + 1]

                    # Normalize those vectors
                    v1 = v1 / np.linalg.norm(v1)
                    v2 = v2 / np.linalg.norm(v2)
                    v3 = v3 / np.linalg.norm(v3)
                    v4 = v4 / np.linalg.norm(v4)

                    # Calculate norm
                    normals[i, j] = - np.cross(v1, v2) - np.cross(v3, v4)
        self.normals = vbo.VBO(np.reshape(normals, (width * height, 3)), usage=GL_STATIC_DRAW)


class StructureDepthmapGeometry(DepthmapGeometry):
    """ Like Depthmap geometry, but now points get connected.
    Idea: Remove this and put the indices intoo the DepthmapGeometry class everytime.
    """

    def __init__(self, filename, position=None):
        DepthmapGeometry.__init__(self, filename, position)

        height, width = self.height, self.width

        # Create indices
        self.indices = [i + shift for i in range(0, width * height - width) for shift in (0, width + 1, 1) if
                        (i + 1) % width != 0]
        self.indices.extend(
            [i + shift for i in range(width, width * height) for shift in (0, 1, -width) if (i + 1) % width != 0])


class ColoredDepthmapGeometry(DepthmapGeometry):
    """
    Vertex colors are determined by a color image
    Idea unify this class into the Depthmap Geometry class
    # TODO unify
    """

    def __init__(self, filename, position=None):
        DepthmapGeometry.__init__(self, filename, position)

        # Overwrite color VBO
        color_img = Image.open("screen_color.png")
        color_array = np.asarray(color_img, dtype=np.float32)
        height, width, num_colors = color_array.shape

        color_array = np.reshape(color_array, (height * width, 3))
        print "max", np.min(color_array)

        color_array /= 255.

        print color_array[0:20]
        self.colors = vbo.VBO(color_array, usage=GL_STATIC_DRAW)
