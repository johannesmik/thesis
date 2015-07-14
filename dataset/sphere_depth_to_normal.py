__author__ = 'johannes'
"""
Takes a depthmap from a sphere and calculates the normal
"""

import numpy as np

import context
import scenes
import cameras
import meshes
import materials
import lights


if __name__ == '__main__':

    # Create the context and camera
    c = context.Context(width=512, height=424, show_framerate=False)
    camera = cameras.PerspectiveCamera2()

    # Setup the scene
    scene = scenes.Scene(backgroundcolor=np.array([1., .6, 0, 1]))
    square_material = materials.NormalMaterial()
    square_material.add_depthmap('../assets/sphere_depth.tiff')
    square = meshes.Mesh(name='Square 1', position=np.array([0, 0, -1.5]), geometry=meshes.SquareGeometry(),
                         material=square_material)
    square.size = 1.0

    light = lights.PointLight(position=np.array([0, 0, 0]), color=np.array([1, 1, 1, 1]), falloff=0)

    scene.add(square)
    scene.add(light)

    # The Loop
    running = True
    while running:
        running = c.render(scene, camera)