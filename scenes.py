__author__ = 'johannes'

import numpy as np
import meshes, materials, lights

class Scene(object):
    def __init__(self, backgroundcolor=None):
        self.meshes = []
        self.lights = []
        self.cameras = []
        self.backgroundcolor = backgroundcolor

    def addMesh(self, mesh):
        self.meshes.append(mesh)

    def addLight(self, light):
        self.lights.append(light)

    def addCamera(self, camera):
        self.cameras.append(camera)


class exampleScene1(Scene):
    " One Sphere "

    def __init__(self):
        super(exampleScene1, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)

        #light1 = lights.AmbientLight(color=np.array([0.2, 0.2, 0.2, 1]))
        light2 = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

        self.addMesh(sphere)
        self.addLight(light2)
        #self.addLight(light1)

class exampleScene2(Scene):
    " Three spheres "

    def __init__(self):
        super(exampleScene2, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([2, 2, -4]), geometry=sphere_geometry, material=sphere_material)
        sphere.size = 2
        self.addMesh(sphere)

        light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

        self.addLight(light)

class exampleScene3(Scene):
    ' One square with a depthmap on it '
    def __init__(self):

        super(exampleScene3, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        square_material = materials.LambertianMaterial()
        square_material.add_normalmap('texture_normal.png')
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -1.5]), geometry=meshes.SquareGeometry(), material=square_material)
        square.size = 1.5

        light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

        self.addMesh(square)
        self.addLight(light)

class exampleScene4(Scene):
    ' A sphere with a Rectangle behind it'

    def __init__(self):
        super(exampleScene4, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        square_material = materials.LambertianMaterial()
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -3]), geometry=meshes.SquareGeometry(), material=square_material)
        square.size = 3
        self.addMesh(square)

        light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))
        self.addLight(light)

class exampleScene5(Scene):
    ''' Four spheres with two Point Light (And light fall-off)

        Example for Pointlights
    '''

    def __init__(self):
        super(exampleScene5, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([-2, 0, 0]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 3', position=np.array([0, 2, 0]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 4', position=np.array([0, 0, -5]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.1)
        self.addLight(light)

        light = lights.PointLight(position=np.array([1.5, 0, -2]), color=np.array([1, 1, 1, 1]), falloff=1)
        self.addLight(light)

class exampleScene6(Scene):
    ''' Two spheres and a Rectangle with one Spot Light (And light fall-off)

        Example for Spot Lights
    '''


    def __init__(self):
        super(exampleScene6, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([0, 0, -5]), geometry=sphere_geometry, material=sphere_material)
        self.addMesh(sphere)

        square_geometry = meshes.SquareGeometry()
        square_material = materials.LambertianMaterial()
        square = meshes.Mesh(name='Square', position=np.array([0, 0, -3.5]), geometry=square_geometry, material=square_material)
        square.size = 2
        self.addMesh(square)

        light = lights.SpotLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0,
                                 cone_angle=np.pi/8., direction=np.array([0, 0, 2]))
        self.addLight(light)
