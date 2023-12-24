#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import glob
import csv
import os

# Parameters
chessboard_square_size_mm = 19
termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Object points
object_points_chessboard = np.zeros((6*8, 3), np.float32)
object_points_chessboard[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
object_points_chessboard = object_points_chessboard * chessboard_square_size_mm

# Arrays to store object points and image points
object_points_list = []  # 3D points in real-world space
image_points_list = []  # 2D points in image plane

# Find chessboard corners in images
image_files = sorted(glob.glob('*.jpg'))

print(image_files)

for image_file in image_files:
    img = cv2.imread(image_file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray_image, (8, 6), None)

    print(f"{image_file[:-4]}:{ret}")
    if ret:
        object_points_list.append(object_points_chessboard)
        corners_refined = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), termination_criteria)
        image_points_list.append(corners_refined)
        
        cv2.drawChessboardCorners(img, (8, 6), corners_refined, ret)
        #cv2.imshow('img', img)
        #cv2.imwrite(os.path.join("corners_image", f'{image_file[:-4]}_corners.png'), img) 
        #cv2.waitKey(500)

#cv2.destroyAllWindows()

# Get intrinsic parameters
height_image, width_image = img.shape[:2]
ret_calibration, camera_matrix, distortion_coefficients, r_vectors, t_vectors = cv2.calibrateCamera(
    object_points_list, image_points_list, gray_image.shape[::-1], None, None
)

print("Camera matrix: ")
print(camera_matrix)
print()

print("Distortion coefficients: ")
print(distortion_coefficients)
print()

print("Rotation Vectors: ")
print(r_vectors)
print()

print("Translation Vectors: ")
print(t_vectors)
print()

# Write intrinsic parameters to CSV
'''
with open('PartA_output.csv', 'w') as file_csv:
    file_csv.write("Camera matrix\n")
    for row in camera_matrix:
        file_csv.write(','.join(map(str, row)))
        file_csv.write("\n")
    file_csv.write("Distortion coefficients\n")
    file_csv.write(','.join(map(str, distortion_coefficients[0])))
    file_csv.write("\n")
'''

def calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True):

    if eye_to_hand:
        # change coordinates from gripper2base to base2gripper
        R_base2gripper, t_base2gripper = [], []
        for R, t in zip(R_gripper2base, t_gripper2base):
            R_b2g = R.T
            t_b2g = -R_b2g @ t
            R_base2gripper.append(R_b2g)
            t_base2gripper.append(t_b2g)
        
        # change parameters values
        R_gripper2base = R_base2gripper
        t_gripper2base = t_base2gripper

    # calibrate
    R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )

    return R, t

target = [[78.00, 219, 306, 179.00, 34.0, 146.00],
            [381.00, 293, 348, -148.00, 5.0, 118.00],
            [429.00, -33, 357, -140.00, 7.0, 9.00],
            [463.00, 169, 296, -143.00, 0.0, 89.00],
            [200.00, 80, 500, -180.00, 0.0, 135.00],
            #"200.00, 140, 500, -180.00, 0.0, 135.00",
            [240.00, 80, 500, -180.00, 0.0, 135.00],
            [240.00, 140, 500, -180.00, 0.0, 135.00],
            [220.00, 210, 500, -180.00, 0.0, 135.00]]
print(len(target))



def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

t_gripper2base = []
R_gripper2base = []
for i in range(len(target)):
        
        t_gripper2base.append(np.array(((target[i]))[0:3]))
        R_gripper2base.append(eul2rot(((target[i]))[3:6]))
    # else:
    #     t_gripper2base = np.append(t_gripper2base, np.array(((target[i]))[0:3]))
    #     R_gripper2base = np.append(R_gripper2base, eul2rot(((target[i]))[3:6]))
print(t_gripper2base)
print(R_gripper2base)

#R_gripper2base = 
#t_gripper2base = 
R_target2cam = r_vectors
t_target2cam = t_vectors
#R, t = calibrate_eye_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True)
R, t = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
    )
print(R)
print(t)
