__author__ = 'johannes'

import re

import numpy as np

import meshes
import materials
import lights
import cameras

light_ambient = lights.AmbientLight(color=np.array([0.2, 0.2, 0.2, 1]))
light_pointlight1 = lights.PointLight(name='Point Light 1',
                                     position=np.array([0., 0., 0.]),
                                     color=np.array(np.array([0.2, 0.2, 0.2, 1])),
                                     falloff=0.2)
light_pointlight2 = lights.PointLight(name='Point Light 2',
                                     position=np.array([0., 0., 0.]),
                                     color=np.array(np.array([1., 1., 1., 1.])),
                                     falloff=1.0)
light_directional = lights.DirectionalLight(name='Directional Light', color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

light_spotlight = lights.SpotLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.1,
                                 cone_angle=np.pi / 8., direction=np.array([0, 0, 2]))


class Scene(object):
    def __init__(self, backgroundcolor=None, name="Untitled Scene"):
        self.objects = {}
        self.meshes = []
        self.lights = []
        self.cameras = []
        self.backgroundcolor = backgroundcolor
        self.name = name

    def _add_mesh(self, mesh):
        self.meshes.append(mesh)

    def _add_light(self, light):
        self.lights.append(light)

    def _add_camera(self, camera):
        self.cameras.append(camera)

    def _set_lights(self):
        pass

    def activate(self):
        self._set_lights()

    def add(self, obj):
        return self.add_object(obj)

    def add_object(self, obj):

        if obj.name in self.objects and obj == self.objects[obj.name]:
            return

        if obj.name in self.objects:
            match = re.search(r'(.*?)(\d+)$', obj.name)
            if match:
                new_index = int(match.group(2)) + 1
                obj.name = "%s%03d" % (match.group(1), new_index)
            else:
                obj.name += "001"
            return self.add_object(obj)

        if obj.name not in self.objects:
            self.objects[obj.name] = obj

        if isinstance(obj, cameras.Camera):
            self._add_mesh(obj)
        if isinstance(obj, lights.Light):
            self._add_light(obj)
        if isinstance(obj, meshes.Mesh):
            self._add_mesh(obj)

    def remove_object(self, obj):
        if obj.name in self.objects:
            # Object found
            self.objects.pop(obj.name)
            self.cameras = [y for y in self.cameras if y != obj]
            self.lights = [y for y in self.lights if y != obj]
            self.meshes = [y for y in self.meshes if y != obj]
        else:
            print "object ", obj.name, " not found in scene (tried to remove object)."

    def remove_object_by_name(self, name):
        if name in self.objects:
            # Object found
            self.objects.pop(name)
            self.cameras = [y for y in self.cameras if y.name != name]
            self.lights = [y for y in self.lights if y.name != name]
            self.meshes = [y for y in self.meshes if y.name != name]
        else:
            print "object ", name, " not found in scene (tried to remove object)."

    def remove_lights(self):
        """ Removes all lights from the scene. """
        for light in self.lights:
            self.objects.pop(light.name)
        self.lights = []

    def get_object(self, name):
        return self.objects.get(name, None)


class SimpleSphere(Scene):
    """ One Sphere with an Ambient and Directional Light """

    def __init__(self):
        super(SimpleSphere, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Simple Sphere')

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                             material=sphere_material)

        self.add(sphere)

        self.add(light_ambient)
        self.add(light_directional)

    def _set_lights(self):
        light_ambient.set_parameters(color=np.array([0.2, 0.2, 0.2, 1]))
        light_directional.set_parameters(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

class SimpleWave(Scene):
    """ Simple Wave with an Ambient and Directional Light """

    def __init__(self):
        super(SimpleWave, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Simple Wave')

        wave_geometry = meshes.WaveGeometry(periodicity=0.5, hscale=0.1)
        wave_material = materials.LambertianMaterial()
        wave = meshes.Mesh(name='Wave 1', position=np.array([0, 0, -2]), geometry=wave_geometry,
                             material=wave_material)

        wave_material.set_basecolor(np.array([1.0, 0.6, 0]))

        self.add(wave)
        self.add(light_pointlight1)

    def _set_lights(self):
        light_pointlight1.set_parameters(color=np.array([1, 1, 1, 1]), position=np.array([0, 0, 0]), falloff=0.0)

class TwoSpheres(Scene):
    """ Two spheres with a Directional Light """

    def __init__(self):
        super(TwoSpheres, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Two Spheres')

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([2, 2, -4]), geometry=sphere_geometry,
                             material=sphere_material)
        sphere.set_size(2)
        self.add(sphere)

        self.add(light_directional)

    def _set_lights(self):
        light_directional.set_parameters(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))


class FourSpheres(Scene):
    """ Four spheres with two Point Light (And light fall-off)

        Example for Pointlights
    """

    def __init__(self):
        super(FourSpheres, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Four Spheres')

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([-2, 0, 0]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 3', position=np.array([0, 2, 0]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 4', position=np.array([0, 0, -5]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

    def _set_lights(self):
        light_pointlight1.set_parameters(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.1)
        light_pointlight2.set_parameters(position=np.array([1.5, 0, -2]), color=np.array([1, 1, 1, 1]), falloff=1)
        light_ambient.set_parameters(color=np.array([0.3, 0.3, 0.3, 1]))


class DepthTexture(Scene):
    """ One square with a depthmap on it """

    def __init__(self, lighttype="directional", material="lambertian"):

        super(DepthTexture, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Depth Texture')

        self.lighttype = lighttype

        if material == "lambertian":
            square_material = materials.LambertianMaterial()
        elif material == "normal":
            square_material = materials.NormalMaterial()

        square_material.add_depthmap('../assets/sphere_depth.png')
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -1.5]), geometry=meshes.SquareGeometry(),
                             material=square_material)
        square.size = 1.5
        self.add(square)

        if lighttype == "directional":
            self.add(light_directional)
        elif lighttype == "point":
            self.add(light_pointlight1)

    def _set_lights(self):
        if self.lighttype == "directional":
            light_directional.set_parameters(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))
        elif self.lighttype == "point":
            light_pointlight1.set_parameters(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)


class NormalTexture(Scene):
    """ One square with a depthmap on it """

    def __init__(self, lighttype="directional", material="lambertian"):

        super(NormalTexture, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Normal Texture')
        self.lighttype = lighttype

        if material == "lambertian":
            square_material = materials.LambertianMaterial()
        elif material == "normal":
            square_material = materials.NormalMaterial()

        square_material.add_normalmap('assets/texture_sphere_normal.png')
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -1.5]), geometry=meshes.SquareGeometry(),
                             material=square_material)
        square.size = 1.5

        self.add(square)

        if lighttype == "directional":
            self.add(light_directional)
        elif lighttype == "point":
            self.add(light_pointlight1)

    def _set_lights(self):
        if self.lighttype == "directional":
            light_directional.set_parameters(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))
        elif self.lighttype == "point":
            light_pointlight1.set_parameters(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)


class SphereRectangle(Scene):
    """ A sphere with a Rectangle behind it"""

    def __init__(self, lighttype):
        super(SphereRectangle, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Sphere and Rectangle')

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        square_material = materials.LambertianMaterial()
        square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -3]), geometry=meshes.SquareGeometry(),
                             material=square_material)
        square.size = 3
        self.add(square)

        if lighttype == "directional":
            self.add(light_directional)
        elif lighttype == "point":
            self.add(light_pointlight1)

    def _set_lights(self):
        if self.lighttype == "directional":
            light_directional.set_parameters(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))
        elif self.lighttype == "point":
            light_pointlight1.set_parameters(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)




class SpotLightExample(Scene):
    """ Two spheres and a Rectangle with one Spot Light (And light fall-off)

        Example for Spot Lights
    """

    def __init__(self):
        super(SpotLightExample, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Spot Light Example')

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry()
        sphere_material = materials.LambertianMaterial()
        sphere = meshes.Mesh(name='Sphere', position=np.array([0, 0, -5]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        square_geometry = meshes.SquareGeometry()
        square_material = materials.LambertianMaterial()
        square = meshes.Mesh(name='Square', position=np.array([0, 0, -3.5]), geometry=square_geometry,
                             material=square_material)
        square.size = 2
        self.add(square)

        self.add(light_spotlight)

    def _set_lights(self):
        light_spotlight.set_parameters(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.1,
                                       cone_angle=np.pi / 8., direction=np.array([0, 0, 2]))


class Monkey(Scene):
    def __init__(self):
        super(Monkey, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Monkey')

        monkey_geometry = meshes.ObjectFile('../assets/suzanne.obj')
        monkey_material = materials.BlinnPhongMaterial()
        monkey_material.add_colormap("../assets/suzanne-texture.png")
        monkey = meshes.Mesh(name='Suzanne', position=np.array([0, 0, -2]), geometry=monkey_geometry,
                             material=monkey_material)
        self.add(monkey)

        self.add(light_pointlight1)
        self.add(light_pointlight2)
        self.add(light_ambient)

    def _set_lights(self):
        light_pointlight1.set_parameters(position=np.array([0, 2, 0]), color=np.array([1, 1, 1, 1]), falloff=0.5)
        light_pointlight2.set_parameters(position=np.array([1.5, 4, -2]), color=np.array([1, 1, 1, 1]), falloff=0.5)
        light_ambient.set_parameters(color=np.array([0.3, 0.3, 0.3, 1]))

class Face(Scene):
    def __init__(self):
        super(Face, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='Face')

        # Model from 'TurboSquid'
        face_geometry = meshes.ObjectFile('../assets/humanface.obj')
        face_material = materials.BlinnPhongMaterial()
        face_material.add_colormap("../assets/humanface-texture.png")
        face = meshes.Mesh(name='Face', position=np.array([0, 0, -2]), geometry=face_geometry, material=face_material)
        self.add(face)

        self.add(light_pointlight1)
        self.add(light_pointlight2)
        self.add(light_ambient)

    def _set_lights(self):
        light_pointlight1.set_parameters(position=np.array([0, 2, 0]), color=np.array([1, 1, 1, 1]), falloff=0.5)
        light_pointlight2.set_parameters(position=np.array([1.5, 2, -2]), color=np.array([1, 1, 1, 1]), falloff=0.5)
        light_ambient.set_parameters(color=np.array([0.3, 0.3, 0.3, 1]))

###################################################
#### Scenes used for the dataset in the thesis ####
###################################################

class DiffuseSphere(Scene):

        def __init__(self):
            super(DiffuseSphere, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='1 Diffuse Sphere')

            sphere_geometry = meshes.IcosphereGeometry(subdivisions=3)
            sphere_material = materials.LambertianMaterial()
            sphere_material.set_basecolor(np.array([1, 1, 1]))
            sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                                 material=sphere_material)

            self.add(sphere)

            self.add(light_ambient)
            self.add(light_pointlight1)

        def _set_lights(self):
            light_ambient.set_parameters(color=np.array([0.2, 0.2, 0.2, 1]))
            light_pointlight1.set_parameters(color=np.array([1, 1, 1, 1]),
                                             position=np.array([1, 1, 0]),
                                             falloff=0.4)


class SpecularSphere(Scene):
        """ A sphere with a pointlight. """

        def __init__(self):
            super(SpecularSphere, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='2 Specular Sphere')

            sphere_geometry = meshes.IcosphereGeometry(subdivisions=3)
            sphere_material = materials.BlinnPhongMaterial(specularity=50)
            sphere_material.set_basecolor(np.array([1, 1, 1]))
            sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                                 material=sphere_material)

            self.add(sphere)

            self.add(light_ambient)
            self.add(light_pointlight1)

        def _set_lights(self):
            light_ambient.set_parameters(color=np.array([0.2, 0.2, 0.2, 1]))
            light_pointlight1.set_parameters(color=np.array([1, 1, 1, 1]),
                                             position=np.array([1, 1, 0]),
                                             falloff=0.4)


class SpecularSphereBacklight(Scene):
        """ A sphere with a pointlight behind the camera """

        def __init__(self):
            super(SpecularSphereBacklight, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='3 Specular Sphere Backlight')

            sphere_geometry = meshes.IcosphereGeometry(subdivisions=3)
            sphere_material = materials.BlinnPhongMaterial(specularity=50)
            sphere_material.set_basecolor(np.array([1, 1, 1]))
            sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                                 material=sphere_material)

            self.add(sphere)

            self.add(light_ambient)
            self.add(light_pointlight1)

        def _set_lights(self):
            light_ambient.set_parameters(color=np.array([0.2, 0.2, 0.2, 1]))
            light_pointlight1.set_parameters(color=np.array([1, 1, 1, 1]),
                                             position=np.array([0, 0, 0.5]),
                                             falloff=0.4)


class ThreeSpheres(Scene):
    """ Three spheres with a Point Light """

    def __init__(self):
        super(ThreeSpheres, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='4 Three Spheres')

        sphere_geometry = meshes.IcosphereGeometry(subdivisions=3)
        sphere_material = materials.BlinnPhongMaterial(specularity=50.0, specular_color=np.array([0.62, 0.62, 0.62, 0.62]))
        sphere_material.set_basecolor(np.array([0, 1, 0]))
        sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -3.5]), geometry=sphere_geometry,
                             material=sphere_material)
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry(subdivisions=3)
        sphere_material = materials.BlinnPhongMaterial(specularity=50.0)
        sphere_material.set_basecolor(np.array([0, 0, 1]))
        sphere = meshes.Mesh(name='Sphere 2', position=np.array([2, 0, -4]), geometry=sphere_geometry,
                             material=sphere_material)
        sphere.size = 1
        self.add(sphere)

        sphere_geometry = meshes.IcosphereGeometry(subdivisions=3)
        sphere_material = materials.LambertianMaterial()
        sphere_material.set_basecolor(np.array([1, 0, 0]))
        sphere = meshes.Mesh(name='Sphere 3', position=np.array([-2, 0, -4]), geometry=sphere_geometry,
                             material=sphere_material)
        sphere.size = 1
        self.add(sphere)

        self.add(light_ambient)
        self.add(light_pointlight1)

    def _set_lights(self):
        light_ambient.set_parameters(color=np.array([0.5, 0.5, 0.5, 1]))
        light_pointlight1.set_parameters(position=np.array([5, 5, 5]),
                                         color=np.array([1, 1, 1, 1]),
                                         falloff=0.0)


class Wave(Scene):
    """ Wave with a PointLight """

    def __init__(self):
        super(Wave, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='5 Wave')

        wave_geometry = meshes.WaveGeometry(periodicity=0.7, hscale=0.1)
        wave_material = materials.LambertianMaterial()
        wave = meshes.Mesh(name='Wave 1', position=np.array([0, 0, -2]), geometry=wave_geometry,
                             material=wave_material)

        wave_material.set_basecolor(np.array([1.0, 0.6, 0]))

        self.add(wave)
        self.add(light_ambient)
        self.add(light_pointlight1)

    def _set_lights(self):
        light_ambient.set_parameters(color=np.array([0.3, 0.3, 0.3, 1]))
        light_pointlight1.set_parameters(position=np.array([5, 5, 5]),
                                         color=np.array([1, 1, 1, 1]),
                                         falloff=0.0)


class WaveBacklight(Scene):
    # TODO add different specularities
    """ Wave with a PointLight behind the camera """

    def __init__(self):
        super(WaveBacklight, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='6 Wave Backlight')

        wave_geometry = meshes.WaveGeometry(periodicity=0.7, hscale=0.1)
        wave_material = materials.LambertianMaterial()
        wave = meshes.Mesh(name='Wave 1', position=np.array([0, 0, -2]), geometry=wave_geometry,
                             material=wave_material)

        wave_material.set_basecolor(np.array([1.0, 0.6, 0]))

        self.add(wave)
        self.add(light_ambient)
        self.add(light_pointlight1)

    def _set_lights(self):
        light_ambient.set_parameters(color=np.array([0.2, 0.2, 0.2, 1]))
        light_pointlight1.set_parameters(position=np.array([0, 0, 1]),
                                         color=np.array([1, 1, 1, 1]),
                                         falloff=0.05)

class ColoredBoxes(Scene):
    """ Three colored/textured boxes """
    # TODO color them

    def __init__(self):
        super(ColoredBoxes, self).__init__(backgroundcolor=np.array([1, 1, 1, 1]), name='7 Colored Boxes')

        box_geometry = meshes.BoxGeometry()
        box_material = materials.LambertianMaterial()
        box = meshes.Mesh(name='Box 1', position=np.array([0, 0, -2]), geometry=box_geometry,
                             material=box_material)

        box.set_size(0.75, 2, 0.5)

        self.add(box)
        self.add(light_ambient)
        self.add(light_pointlight1)

    def _set_lights(self):
        light_ambient.set_parameters(color=np.array([0.2, 0.2, 0.2, 1]))
        light_pointlight1.set_parameters(position=np.array([1, 1, 1]),
                                         color=np.array([1, 1, 1, 1]),
                                         falloff=0.2)
