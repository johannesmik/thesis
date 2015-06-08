__author__ = 'johannes'

import re
import numpy as np
import meshes, materials, lights, cameras

class Scene(object):
    def __init__(self, backgroundcolor=None):
        self.objects = {}
        self.meshes = []
        self.lights = []
        self.cameras = []
        self.backgroundcolor = backgroundcolor
        print "initialized a Scene"

    def _add_mesh(self, mesh):
        self.add(mesh)

    def _add_light(self, light):
        self.add(light)

    def _add_camera(self, camera):
        self.add(camera)

    def add(self, obj):
        return self.add_object(obj)

    def add_object(self, obj):

        if obj.name in self.objects:
            match = re.search(r'(.*?)(\d+)$', obj.name)
            if match:
                new_index = int(match.group(2)) + 1
                obj.name = "%s%03d" % (match.group(1), new_index)
            else:
                obj.name = obj.name + "001"
            return self.add_object(obj)

        if obj.name not in self.objects:
            self.objects[obj.name] = obj

        if isinstance(obj, cameras.Camera):
            self.meshes.append(obj)
        if isinstance(obj, lights.Light):
            self.lights.append(obj)
        if isinstance(obj, meshes.Mesh):
            self.meshes.append(obj)

    def get_object(self, name):
        return self.objects.get(name, None)

class SimpleSphere(Scene):
    " One Sphere with an Ambient and Directional Light "

    def __init__(self):
        super(SimpleSphere, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)

        light1 = lights.AmbientLight(color=np.array([0.2, 0.2, 0.2, 1]))
        light2 = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

        self.add(sphere)
        self.add(light1)
        self.add(light2)

class TwoSpheres(Scene):
    " Two spheres with a Directional Light "

    def __init__(self):
        super(TwoSpheres, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([2, 2, -4]), geometry=sphere_geometry, material=sphere_material)
        sphere.size = 2
        self.add(sphere)

        light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

        self.add(light)

class ThreeSpheres(Scene):
    " Three spheres with a Directional Light "

    def __init__(self):
        super(ThreeSpheres, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([2, 2, -4]), geometry=sphere_geometry, material=sphere_material)
        sphere.size = 2
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 3', position=np.array([-2, -2, -4]), geometry=sphere_geometry, material=sphere_material)
        sphere.size = 2
        self.add(sphere)

        light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

        self.add(light)

class FourSpheres(Scene):
    ''' Four spheres with two Point Light (And light fall-off)

        Example for Pointlights
    '''

    def __init__(self):
        super(FourSpheres, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([-2, 0, 0]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 3', position=np.array([0, 2, 0]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 4', position=np.array([0, 0, -5]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.1)
        self.add(light)

        light = lights.PointLight(position=np.array([1.5, 0, -2]), color=np.array([1, 1, 1, 1]), falloff=1)
        self.add(light)

        light = lights.AmbientLight(color=np.array([0.3, 0.3, 0.3, 1]))
        self.add(light)

class DepthTexture(Scene):
    ' One square with a depthmap on it '
    def __init__(self, lighttype="directional", material="lambertian"):

        super(DepthTexture, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        if material == "lambertian":
            square_material = materials.LambertianMaterial()
        elif material == "normal":
            square_material = materials.NormalMaterial()

        square_material.add_depthmap('images/sphere_depth_small.png')
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -1.5]), geometry=meshes.SquareGeometry(), material=square_material)
        square.size = 1.5

        if lighttype == "directional":
            light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))
        elif lighttype == "point":
            light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)

        self.add(square)
        self.add(light)

class NormalTexture(Scene):
    ' One square with a depthmap on it '
    def __init__(self, lighttype="directional", material="lambertian"):

        super(NormalTexture, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        if material == "lambertian":
            square_material = materials.LambertianMaterial()
        elif material == "normal":
            square_material = materials.NormalMaterial()

        square_material.add_normalmap('images/texture_normal.png')
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -1.5]), geometry=meshes.SquareGeometry(), material=square_material)
        square.size = 1.5

        if lighttype == "directional":
            light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))
        elif lighttype == "point":
            light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)

        self.add(square)
        self.add(light)

class SphereRectangle(Scene):
    ' A sphere with a Rectangle behind it'

    def __init__(self, lighttype):
        super(SphereRectangle, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        square_material = materials.LambertianMaterial()
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -3]), geometry=meshes.SquareGeometry(), material=square_material)
        square.size = 3
        self.add(square)

        if lighttype == "directional":
            light = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))
        elif lighttype == "point":
            light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)
        self.add(light)

class SpotLightExample(Scene):
    ''' Two spheres and a Rectangle with one Spot Light (And light fall-off)

        Example for Spot Lights
    '''


    def __init__(self):
        super(SpotLightExample, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere', position=np.array([0, 0, -5]), geometry=sphere_geometry, material=sphere_material)
        self.add(sphere)

        square_geometry = meshes.SquareGeometry()
        square_material = materials.LambertianMaterial()
        square = meshes.Mesh(name='Square', position=np.array([0, 0, -3.5]), geometry=square_geometry, material=square_material)
        square.size = 2
        self.add(square)

        light = lights.SpotLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.1,
                                 cone_angle=np.pi/8., direction=np.array([0, 0, 2]))
        self.add(light)

class Monkey(Scene):

    def __init__(self):
        super(Monkey, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]))

        monkey_geometry = meshes.ObjectFile('./assets/suzanne.obj')
        monkey_material = materials.BlinnPhongMaterial()
        monkey_material.add_colormap("./assets/suzanne-texture.png")
        monkey = meshes.Mesh(name='Suzanne', position=np.array([0, 0, -3]), geometry=monkey_geometry, material=monkey_material)
        self.add(monkey)

        light = lights.PointLight(position=np.array([0, 2, 0]), color=np.array([1, 1, 1, 1]), falloff=0.5)
        self.add(light)

        light = lights.PointLight(position=np.array([1.5, 4, -2]), color=np.array([1, 1, 1, 1]), falloff=0.5)
        self.add(light)

        light = lights.AmbientLight(color=np.array([0.3, 0.3, 0.3, 1]))
        self.add(light)

