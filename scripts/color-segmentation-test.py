"""
    Test the color segmentation using k-means
"""

import numpy as np
import cv2
import time

img = cv2.imread('../assets/green_ball_small.jpg')
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K = 3
t0 = time.time()
ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_PP_CENTERS)
t_kmeans = time.time() - t0

print "found K", K
print ret
print label
print center
print "time needed", t_kmeans

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)

cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save file
# cv2.imwrite('./green_ball_segmented_%d_colors.jpg' % K, res2)
