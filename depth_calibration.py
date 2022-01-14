import numpy as np
import cv2
import os
from glob import glob

"""
old camera parameters
fx=654.75924295
fy=842.74853381
cx=325.50911362
cy=619.35980012
"""

isRGB = True
images = np.load("images.npz")
pattern_size = (6,9)

if (isRGB):
    images = images["color_images"]
    CAMERA_PATH = "color_params.npz"
else:
    images = images["depth_images"]
    CAMERA_PATH = "depth_params.npz"

#create object points
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= 17/9

obj_points = []
img_points = []
h, w = 0, 0
for i in range(10):
    img = images[i]
    #img = (img/256).astype('uint8')
    cv2.imshow("img", img)
    if img is None:
        print("Failed to load", fn)
        continue
    cv2.waitKey()
    h, w = img.shape[:2]
    found, corners = cv2.findChessboardCorners(img, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
    if found:
        pass
        #cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    if not found:
        print('chessboard not found')
        continue

    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

    #save img_points for future stereo calibration
    #img_file = shelve.open(os.path.splitext(fn)[0],'n')
    #img_file['img_points'] = corners.reshape(-1, 2)
    #img_file.close()

    print('ok')

rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                                   img_points,
                                                                   (w, h),
                                                                   None,
                                                                   None,
                                                                   criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 0.001),
                                                                   flags = 0)
#save calibration results
#camera_file = shelve.open(CAMERA_PATH, 'n')
#camera_file['camera_matrix'] = camera_matrix
#camera_file['dist_coefs'] = dist_coefs
#camera_file.close()

print("RMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())