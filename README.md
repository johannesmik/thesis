Source code for my master's thesis (2015).

In this repository there are six directories:

  - assets: contains calibration data, images, and 3D objects
  - blender: contains Blender source files.
  - clustering: source code for the Section Material Estimation
  - dataset-synth: source code to create artificial RGB-D-IR images
  - optimization: source code (Python/Cuda) for the Section Optimization
  - scripts: miscellaneous scripts for testing or image purposes

# Installation and requirements

We describe the installation process for an Ubuntu system (tested under Ubuntu 14.04 LTS).

The dataset creation code requires an OpenGL 3.0 capable graphics card and supporting drivers.

The optimization code requires a CUDA capable graphics card. (We tested with a GeForce GTX 460, which has a CUDA compute capability of 2.1)

## Dataset Creation

For the dataset creation, you need to have [Python2](www.python.org), [PyOpenGL](http://pyopengl.sourceforge.net/),
[Numpy](http://www.numpy.org/), [Pillow](http://python-pillow.github.io/)(a PIL fork), and PySDL2 installed.

On Ubuntu:

```bash
$ sudo apt-get install libsdl2-dev
$ sudo apt-get install python python-numpy python-pil python-pip
$ sudo pip install PyOpenGL PyOpenGL_accelerate
$ pip install --user -U pysdl2
```

Line 1 installs the necessary SDL2 libraries.
Line 2 installs Python, Numpy, Pillow, and pip.
Line 3 installs the necessary PyOpenGL packages.
Line 4 downloads the latest PySDL2 source code, builds it, and installs them in the users home directory `~HOME/.local`.

## Clustering

For clustering we need [Python2](www.python.org), [Numpy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/),
[Scikit Learn](http://scikit-learn.org/) installed:

```bash
$ sudo apt-get install build-essential python-dev python-setuptools python-pip
$ sudo apt-get install python python-numpy python-matplotlib python-scipy
$ suda apt-get install libatlas-dev libatlas3gf-base
$ pip install --user -U scikit-learn
```

Line 1 installs necessary packages for building scikit-learn in line 4.
Line 2 installs Python, Numpy, Matplotlib, Scipy (required for scikit-learn).
Line 3 installs the ATLAS implementation of BLAS (Basic Linear Algebra Subprograms).
Line 4 downloads the most recent scikit-learn sources, builds them, and installs them in the users home directory `~HOME/.local`.

## Optimization

First, install PyCUDA as instructed on the [PyCUDA installation guide for Ubuntu](http://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu).

Next, we again need the [Python2](www.python.org), [Numpy](http://www.numpy.org/), [Matplotlib](http://matplotlib.org/).
They should already be installed from the previous steps.

```bash
$ sudo apt-get install python python-numpy python-matplotlib python-pil
```

# Dataset creation


## create_thesis_dataset.py

Creates a dataset (Color, Normal, Depth, IR images) of the scenes used in the thesis.

## context.py

This file shows a scene, and you can navigate through it using the keyboard.

```bash
$ python context.py
```

### Keyboard shortcuts

  - W, A, S, D, Arrow Keys: Move around
  - Q, E: Rotate up and down
  - R, N, V: Switch between color, depth and normal materials
  - P: switch between color view and infrared view.
  - F: make a color screenshot, and save file to `~/screenshots/screen_color.png`
  - G: make a bw screenshot, and save file to `~/screenshots/screen_bw.tiff'

# Clustering

## thesis_images_1.py

Clusters the scene 'Specular Sphere' used in the thesis for the explanation of the algorithm.

```bash
$ python thesis_images_1.py
```

## thesis_results_1.py and thesis_results_2.py

As the name suggests, those are the results used in my thesis. Opens a lot of image plots.

# Optimization

## thesis_optimization_1.py

Optimizes the monkey with 'perfect' prior knowledge of the material.

```bash
$ python thesis_optimization_1.py
```
