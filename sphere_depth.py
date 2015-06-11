__author__ = 'johannes'

"""
 Show a simple sphere in depthmap.
"""

import numpy as np
import context, scenes, cameras, meshes, materials, lights

if __name__ == '__main__':

    # Setup context and camera
    c = context.Context(width=512, height=424, show_framerate=False)
    camera = cameras.PerspectiveCamera2()

    # Setup the scene
    scene = scenes.Scene(backgroundcolor=np.array([1., 1., 1., 1.]))
    sphere_geometry = meshes.IcosphereGeometry(subdivisions=4)
    sphere_material = materials.DepthMaterial()
    sphere = meshes.Mesh(name='Sphere 1', position=np.array([0, 0, -2]), geometry=sphere_geometry, material=sphere_material)

    square_material = materials.DepthMaterial()
    square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -3]), geometry=meshes.SquareGeometry(), material=square_material)
    square.size = 3

    light1 = lights.AmbientLight(color=np.array([0.2, 0.2, 0.2, 1]))
    light2 = lights.DirectionalLight(color=np.array([1, 1, 1, 1]), direction=np.array([0, 0, -1]))

    scene.add(sphere)
    scene.add(square)
    scene.add(light1)
    scene.add(light2)

    # The Loop
    running = True
    while running:
        running = c.render(scene, camera)