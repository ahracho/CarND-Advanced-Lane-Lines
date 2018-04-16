## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This repository is for the project of Udacity Nanodegree - Self-driving Car Engineer : Advanced Finding Lane Lines Proejct. It is forked from (https://github.com/udacity/CarND-Advanced-Lane-Lines).

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

It needs us to apply everything we learned from the lecture including image pre-processing - sobel and direction gradient, color threshold, perspective transform etc. After calculating gradients, need to find lane lines applying sliding window method. With window centroid points found, calculate polynomial coeffiecients to draw along the lane lines, and project back onto the original image. During the process, I can calculate the estimate curvature and vehicle position also.

Details on the pipeline is described in `writeup.md`.


This project has 3 types of outputs:

1. final_video.mp4 : final video with lanes marked
2. pipeline.py : script used to produce final_video.mp4
~~~sh
python pipeline.py
~~~
3. writeup.md : writeup file that specify details on how I completed this project.
---
4. Line.py / LineTracker.py / util_function.py : dependent files for executing pipeline.py

