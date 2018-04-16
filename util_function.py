import pickle
import os
import cv2
import numpy as np
import matplotlib.image as mpimg

cal_dir = "./camera_cal/"
cal_file = "./calibration_pickle.p"

def camera_calibration(board_size=(9, 6)):
    if os.path.isfile(cal_file):
        with open(cal_file, mode='rb') as f:
            dist_pickle = pickle.load(f)
    else:
        cal_images = os.listdir(cal_dir)

        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        for path in cal_images:
            cal_image = mpimg.imread(cal_dir + path)
            gray = cv2.cvtColor(cal_image, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, board_size, None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Get Camera matrix and distortion coefficients
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Save the result
        dist_pickle = {}
        dist_pickle["mtx"] = mtx
        dist_pickle["dist"] = dist
        pickle.dump(dist_pickle, open(cal_file, "wb"))

    return dist_pickle


def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    elif orient == 'y':
        abs_sobel = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    scale_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    grad_binary = np.zeros_like(scale_sobel)
    grad_binary[(scale_sobel >= thresh[0]) & (scale_sobel <= thresh[1])] = 1
    return grad_binary


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    absx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    absy = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    grad_dir = np.arctan2(absy, absx)

    dir_binary = np.zeros_like(grad_dir)
    dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1
    return dir_binary


def hls_threshold(image, ch='S', thresh=(0, 255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    if ch == 'H':
        value = hls[:, :, 0]
    elif ch == 'L':
        value = hls[:, :, 1]
    elif ch == 'S':
        value = hls[:, :, 2]

    hls_binary = np.zeros_like(value)
    hls_binary[(value > thresh[0]) & (value <= thresh[1])] = 1
    return hls_binary


