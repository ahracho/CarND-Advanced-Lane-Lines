import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle

from LineTracker import LineTracker
from Line import Line
from util_function import *

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# Global Variables
nwindows = 72
line = Line()
tracker = LineTracker(nwindows=nwindows, margin=50)


# (1) Camera Calibration => Do only once per video
# : process_image() function executed on every frame
dist_pickle = camera_calibration()
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def process_image(image):
    # (2) Undistort the frame image
    image = cv2.undistort(image, mtx, dist, None, mtx)

    # (3) Set several thresholds to detect lane lines
    # (3-1) Gradient : Use Sobel X operation
    grad_thresh = (10, 100)
    abs_sobelx = abs_sobel_thresh(image, orient='x', thresh=grad_thresh)

    # (3-2) Color Threshold : Use H and S channel
    h_thresh = (10, 100)
    s_thresh = (10, 255)
    h_binary = hls_threshold(image, ch="H", thresh=h_thresh)
    s_binary = hls_threshold(image, ch="S", thresh=s_thresh)

    # (3-3) Threshold Combination
    combined = np.zeros_like(s_binary)
    combined[((abs_sobelx == 1) & ((s_binary == 1) | (h_binary == 1)))] = 1

    # (4) Perspective Transform : To find lane lines from top-down view
    top_limit = 0.67
    from_mid = 100
    from_edge = 0.17
    bottom_limit = 0.98

    m_w = image.shape[1]
    m_h = image.shape[0]

    src = np.float32([[m_w/2 - from_mid, m_h*top_limit],
                      [m_w/2 + from_mid, m_h*top_limit],
                      [m_w*(1-from_edge), m_h*bottom_limit],
                      [m_w*from_edge+100, m_h*bottom_limit]])
    dst = np.float32([[m_w/5, 20],
                      [m_w*4/5, 20],
                      [m_w*4/5, m_h],
                      [m_w/5, m_h]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(
        combined, M, (m_w, m_h), flags=cv2.INTER_LINEAR)

    # (5) Find window centroids and calculate polynomial coefficients
    tracker.find_window_centroid(warped, line)

    # (5-1) Calculate points to draw polynomial lines
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = line.left_fit[0]*ploty**2 + \
        line.left_fit[1]*ploty + line.left_fit[2]
    right_fitx = line.right_fit[0]*ploty**2 + \
        line.right_fit[1]*ploty + line.right_fit[2]

    # (5-2) Calculate curvature and vehicle position
    left_curve, right_curve = line.calculate_curvature()
    curve_str = "Radius of Curvature : " + \
        str(int((left_curve+right_curve)//2)) + '(m)'

    diff = line.distance_from_center(m_w)
    if diff < 0:
        dist_str = "The vehicle is " + \
            "{0:.2f}".format(-diff) + "m left of center"
    else:
        dist_str = "The vehicle is " + \
            "{0:.2f}".format(diff) + "m right of center"

    # (6) Draw lines back on the original image
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    cv2.putText(result, curve_str, (50, 100),
                cv2.FONT_HERSHEY_DUPLEX, 1, (50, 50, 50), thickness=2)
    cv2.putText(result, dist_str, (50, 150),
                cv2.FONT_HERSHEY_DUPLEX, 1, (50, 50, 50), thickness=2)

    return result


white_output = 'test_video.mp4'

clip1 = VideoFileClip("./harder_challenge_video.mp4").subclip(0,5)
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
