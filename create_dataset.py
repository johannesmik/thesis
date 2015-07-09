from __future__ import print_function

import numpy as np

import cameras
import context
import lights
import materials
import meshes
import scenes

if __name__ == '__main__':

    c = context.Context(width=512, height=424, show_framerate=True)
    c.print_opengl_info()
    camera = cameras.PerspectiveCamera2()
    # Setup camera movement
    r = 2.0
    w = 0.0
    for t in range(270):
        if t <= 90:
            w -= 0.5 * np.pi / 90.
        else:
            w += 0.5 * np.pi / 90.
        camera.set_frame(t, position=np.array([r * np.sin(w), 0, r * np.cos(w)]), rotation=np.array([0, w, 0]))

    pointlight = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)
    sphere_geometry = meshes.IcosphereGeometry(subdivisions=4)

    ## Setup Sphere Scene
    scene_sphere = scenes.Scene(backgroundcolor=np.array([0., 0, 0, 1]), name="sphere")
    sphere = meshes.Mesh(name='Square 1', position=np.array([0, 0, -3.5]), geometry=sphere_geometry,
                         material=materials.LambertianMaterial())
    scene_sphere.add(sphere)
    scene_sphere.add(pointlight)

    ## Setup Head Scene

    scene_head = scenes.Scene(backgroundcolor=np.array([0, 0, 0, 1]), name="head")
    head_geometry = meshes.ObjectFile('./assets/humanface.obj')
    head_material = materials.BlinnPhongMaterial()
    head_material.add_colormap("./assets/humanface-texture.png")
    head = meshes.Mesh(name='Face', position=np.array([0, 0, -1]), geometry=head_geometry, material=head_material)
    scene_head.add(head)
    # scene_head.add(pointlight)
    light = lights.PointLight(position=np.array([0, 2, 0]), color=np.array([1, 1, 1, 1]), falloff=0.5)
    scene_head.add(light)
    light = lights.PointLight(position=np.array([1.5, 2, -2]), color=np.array([1, 1, 1, 1]), falloff=0.5)
    scene_head.add(light)
    light = lights.AmbientLight(color=np.array([0.3, 0.3, 0.3, 1]))
    scene_head.add(light)

    scenes = [scene_sphere, scene_head]
    path = "images/dataset"

    for scene in scenes:

        stripped_name = scene.name.replace(' ', '')

        # Color
        for t in range(270):
            camera.set_current_frame(t)
            c.render(scene, camera)
            c.screenshot(scene, camera, "%s/%s_color.tiff" % (path, stripped_name))

        # Normal
        for mesh in scene.meshes:
            mesh.material = materials.NormalMaterial()

        c.screenshot(scene, camera, "%s/%s_normal.tiff" % (path, stripped_name))

        # Depth
        for mesh in scene.meshes:
            mesh.material = materials.DepthMaterial()

        c.screenshot_bw(scene, camera, "%s/%s_depth.tiff" % (path, stripped_name))