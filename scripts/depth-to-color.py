__author__ = 'johannes'

import numpy as np
import matplotlib.pyplot as plt
import yaml
import cv2
from PIL import Image

import utils



with open("../assets/kinect-camera-calibration/501375743042/calib_ir.yaml", 'r') as stream:
    calib_ir = yaml.load(stream)

with open("../assets/kinect-camera-calibration/501375743042/calib_color.yaml", 'r') as stream:
    calib_color = yaml.load(stream)

with open("../assets/kinect-camera-calibration/196605135147/calib_pose.yaml", 'r') as stream:
    calib_pose = yaml.load(stream)

depth_camera_matrix = np.array(calib_ir['cameraMatrix']['data']).reshape(3, 3)
depth_camera_dist = np.array(calib_ir['distortionCoefficients']['data'])

color_camera_matrix = np.array(calib_color['cameraMatrix']['data']).reshape(3, 3)
color_camera_dist = np.array(calib_color['distortionCoefficients']['data'])

camera_rotation = np.array(calib_pose['rotation']['data']).reshape((3, 3))
camera_translation = np.array(calib_pose['translation']['data']).reshape(3,)
camera_transformation = np.zeros((4, 4))
camera_transformation[:3, :3] = camera_rotation
camera_transformation[:3, 3] = camera_translation
camera_transformation[3, 3] = 1

print 'camera transformation', camera_transformation

print depth_camera_matrix

print depth_camera_dist

# Read the images
color_image = utils.read_blob('/home/johannes/dataset-experiment2/color_0_daylight.blob', 'color')
color_image = color_image * 255
#color_image = np.asarray(color_image, dtype=np.float32)
# Swaps BGRA to RGBA
#color_image[:, :, [0, 1, 2, 3]] = color_image[:, :, [2, 1, 0, 3]]
color_image = color_image / 255.
depth_image = Image.open('/home/johannes/dataset-experiment2/depth_0_daylight.png')
depth_image = np.asarray(depth_image, dtype=np.float32)
depth_image = depth_image / 255. * 4500
depth_image = utils.read_blob('/home/johannes/dataset-experiment2/depth_0_daylight.blob', 'depth')
depth_image = depth_image * 4500
color_h, color_w = color_image.shape[:2]
depth_h, depth_w = depth_image.shape[:2]

# Undistort
# Skipped

z = depth_image
x_s_ir, y_s_ir = np.meshgrid(np.linspace(1, depth_w, depth_w), np.linspace(1, depth_h, depth_h))

X_ir = z / depth_camera_matrix[0][0] * (x_s_ir - depth_camera_matrix[0][2])
Y_ir = z / depth_camera_matrix[1][1] * (y_s_ir - depth_camera_matrix[1][2])
C_ir = np.dstack((X_ir, Y_ir, z, np.ones((depth_h, depth_w))))

print 'depth', C_ir

# Transpose by camera_transformation
i = np.eye(3)
#C_color =  np.einsum('ij, klj -> kli', np.linalg.inv(camera_transformation), C_ir)
C_color =  np.einsum('ij, klj -> kli', camera_transformation, C_ir)
X_color = C_color[:,:,0]
Y_color = C_color[:,:,1]
Z_color = C_color[:,:,2]

#Z_color[np.where(Z_color == 0)] += 0.00001

# Project into the color screen coordinates
x_s_color = color_camera_matrix[0][0] * X_color / Z_color + color_camera_matrix[0][2]
y_s_color = color_camera_matrix[1][1] * Y_color / Z_color + color_camera_matrix[1][2]

# round coordinates
x_s_color = np.round(x_s_color).astype(np.int)
y_s_color = np.round(y_s_color).astype(np.int)


print "550?" , color_camera_matrix[1][2]
print 'max x s color',  np.max(x_s_color)
print 'max y s color', np.max(y_s_color)

print 'color_h color_w', color_h, color_w

# Negative (non-sense values originating from z = 0) to 1
x_s_color = np.clip(x_s_color, 0, color_w - 1)
y_s_color = np.clip(y_s_color, 0, color_h - 1)

print x_s_color
transformed_color = color_image[y_s_color, x_s_color]

plt.figure()
cm = plt.get_cmap("gray")
plt.imshow(color_image[:,:,:3], cmap=cm)
plt.title('color')

plt.figure()
plt.imshow(transformed_color[:,:,:3], cmap=cm)
plt.title('transformed color')

plt.figure()
plt.imshow(depth_image, cmap=cm)
plt.title('depth image')
#plt.colorbar()

print 'depth min max', np.min(depth_image), np.max(depth_image)
plt.show()



