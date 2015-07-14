# Dataset creation

## create_dataset.py

Creates a dataset (Color, Normal, Depth, IR images) of currently two scenes.

The camera rotates around the object in a counter-clockwise motion.

Number of frames and path can be given as command parameters, for example:

```bash
$ python create_dataset.py 200 ~/dataset2
```

## context.py

This file shows a scene, and you can navigate through it using the keyboard.

### Keyboard shortcuts

  - W, A, S, D, Arrow Keys: Move around
  - Q, E: Rotate up and down
  - R, N, V: Switch between color, depth and normal materials
  - F: make a color screenshot, and save file to `images/screen-color.png`
  - G: make a bw screenshot, and save file to `images/screen_bw.tiff'

# Prerequisites

Please make sure that you have [Python2](www.python.org) and [PyOpenGL](http://pyopengl.sourceforge.net/) and
[Numpy](http://www.numpy.org/) and [Pillow](http://python-pillow.github.io/)(a PIL fork) and SDL2 installed.

On Ubuntu:

```bash
$ sudo apt-get install python python-numpy python-pip python-pil libsdl2-dev
$ sudo pip install PyOpenGL PyOpenGL_accelerate
$ sudo pip install pysdl2
```
