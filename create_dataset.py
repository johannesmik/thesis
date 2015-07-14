from __future__ import print_function

import numpy as np
import yaml
import os

import cameras
import context
import lights
import materials
import meshes
import scenes


def save_yaml(filename, scene, camera, time, verbose=False):
    with open(filename, 'w') as f:
        yaml.dump({'scene name' : scene.name,
                   'time' : time,
                   'camera': {'rotation' : camera.rotation.tolist(),
                              'position' : camera.position.tolist()}}, f)

    if verbose:
        print("Saved camera params as %s" % filename)

if __name__ == '__main__':

    c = context.Context(width=512, height=424, show_framerate=False)

    frames = 5
    verbosity = False

    ## Setup camera movement: Move around (0, 0, -2) in a ccw rotation
    camera = cameras.PerspectiveCamera2()
    r = 2.0
    w = 0.0
    for t in range(frames):
        w = 2 * t * np.pi / (frames - 1)
        print (w / np.pi)
        camera.set_frame(t, position=np.array([r * np.sin(w), 0, r * np.cos(w) -2]), rotation=np.array([0, w, 0]))


    ## Setup Lights
    ambientlight = lights.AmbientLight(color=np.array([0.2, 0.2, 0.2, 1]))
    pointlight_origin = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.5)
    pointlight_right = lights.PointLight(position=np.array([1.5, 2, -2]), color=np.array([1, 1, 1, 1]), falloff=0.5)
    ir_light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0.0, name="IR Light")

    ## Setup Sphere Scene
    sphere_geometry = meshes.IcosphereGeometry(subdivisions=4)
    scene_sphere = scenes.Scene(backgroundcolor=np.array([0., 0, 0, 1]), name="sphere")
    sphere = meshes.Mesh(name='Square 1', position=np.array([0, 0, -2]), geometry=sphere_geometry,
                         material=materials.LambertianMaterial())
    scene_sphere.add(sphere)
    scene_sphere.add(pointlight_origin)

    ## Setup Head Scene
    scene_head = scenes.Scene(backgroundcolor=np.array([0, 0, 0, 1]), name="head")
    head_geometry = meshes.ObjectFile('./assets/humanface.obj')
    head_material = materials.BlinnPhongMaterial()
    head_material.add_colormap("./assets/humanface-texture.png")
    head = meshes.Mesh(name='Face', position=np.array([0, 0, -2]), geometry=head_geometry, material=head_material)
    scene_head.add(head)
    scene_head.add(ambientlight)
    scene_head.add(pointlight_origin)
    # scene_head.add(pointlight_right)

    scenes = [scene_sphere, scene_head]
    path = "~/dataset"
    path = os.path.expanduser(path)

    # Wait for confirmation
    user_reply = raw_input("Data will be created in %s. Continue? [y/n]" % (path))
    if user_reply != 'y' and user_reply != 'Y':
        exit()

    if not os.path.exists(path):
        os.makedirs(path)

    for scene in scenes:

        stripped_name = scene.name.replace(' ', '')

        # Color
        for t in range(frames):
            camera.set_current_frame(t)
            c.render(scene, camera)
            save_yaml("%s/%s_%04d_camera.yaml" % (path, stripped_name, t), scene, camera, t, verbose=verbosity)
            c.screenshot(scene, camera, "%s/%s_%04d_color.tiff" % (path, stripped_name, t), verbose=verbosity)

        # Normal
        for mesh in scene.meshes:
            mesh.material = materials.NormalMaterial()
        for t in range(frames):
            camera.set_current_frame(t)
            c.render(scene, camera)
            c.screenshot(scene, camera, "%s/%s_%04d_normal.tiff" % (path, stripped_name, t), verbose=verbosity)

        # Depth
        for mesh in scene.meshes:
            mesh.material = materials.DepthMaterial()
        for t in range(frames):
            camera.set_current_frame(t)
            c.render(scene, camera)
            c.screenshot_bw(scene, camera, "%s/%s_%04d_depth.tiff" % (path, stripped_name, t), verbose=verbosity)

        # IR
        scene.remove_lights()
        scene.add(ir_light)
        for mesh in scene.meshes:
            mesh.material = materials.LambertianMaterial()
        for t in range(frames):
            camera.set_current_frame(t)
            ir_light.set_position(camera.position)
            c.render(scene, camera)
            c.screenshot_bw(scene, camera, "%s/%s_%04d_ir.tiff" % (path, stripped_name, t), verbose=verbosity)