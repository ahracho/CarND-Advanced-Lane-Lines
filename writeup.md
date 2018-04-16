# **Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistorted_board.png "Undistorted"
[image2]: ./writeup_images/process.png "Pipeline"
[image3]: ./writeup_images/undistorted_screen.png "Test Image"
[image4]: ./writeup_images/gradient_sample.jpg "Gradient"
[image5]: ./writeup_images/warped.jpg "Warped"
[image6]: ./writeup_images/line_plot.png "Line"
[image7]: ./writeup_images/final_result.jpg "Line"
[image8]: ./writeup_images/wrong_detect.jpg "Wrong Image"
[video1]: ./final_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Files Submitted & Code Quality

My project includes the following files:
* pipeline.py           : execution script for finding lane lines
* final_video.mp4   : final output video (based on project_video.mp4)
* writeup.md          : summarizing the results


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in _camera\_calibration()_ function _util\_function.py_.

I used all of 20 images given in this project to calibrate camera. I start by preparing "object points(`objpoints `)", which will be the (x, y, z) coordinates of the chessboard corners in the world. I set the chessboard is fixed on the (x, y) plane at z=0, so that the object points are the same for each calibration image.  I used return value of ` findChessboardCorners() ` function as "image points(`imgpoints`)", which indicates (x, y) position of each corner on the images.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

I save the result on `calibration_pickle.p` file so that I calculate camera matrix and distortion coefficients just once and reuse them.  

![undistortion][image1]

### Pipeline (single images)

Image pre-process pipeline basically follows the process provided in lectures as below. 
![pipeline process][image2]

#### 1. Provide an example of a distortion-corrected image.

I used camera matrix and distortion coefficients I get from previous step. On each frame of the video, pipeline starts from distortion-correction. The code for this step is in line 31 in `pipeline.py`.  

Sample images for undistorted image is as below.   
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The first thing I have problem with was finding good threshold combination so that I can detect lane lines regardless of color, shade and obstacles etc. I used test images in `test_images` folder to verity how my filter works. Normally, it worked fine with `straight_linesX.jpg`, I need to tune the thresholds to make it work for images with shadow.

1. Sobel Gradient (Line 36-37 in `pipeline.py` and abs_sobel_thresh() in `util_function.py`)  
First, I tried sobel gradient both x and y, but it seemed sobel-y gradient made no big difference, that's why I used only sobel-x gradient. I set threshold as (10, 100). With a wide range of threshold, noise data can be added but I wanted most of lane lines to be detected here.

2. Color Gradient (Line 40-43 in `pipeline.py` and hls_threshold() in `util_function.py`)  
I changed image into HLS channel and adjusted threshold to H and S channel. By change color channel into HLS, I could get lane line information even when there was shadow on lanes and they have darker color pixel. I set threshold for H channel as (10, 100), and for S channel as (10, 255). 

3. Combination (Line 46-47 in `pipeline.py`)
Last thing I did for gradient was combining three binary image into one. Since my sobel binary image contains clear lane lines but with many noises, I used & operation between sobel and color gradient (Line 47).

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspecitve transform is in Line 48-70 in `pipeline.py`. At first, I set source points and destination points. In this project, image size is never change, but for reuseable code, I set points based on ratio not pixel position (except `from_mid` variable which indicates margin from the center of the image). 

~~~python
top_limit = 0.67 # 67% of  the image height is generally not related to lanes (they are mostly background) 
from_mid = 100 # From the center of the image, +/-100 pixels (left and right lanes never meets on the image)
from_edge = 0.17 # Cropping both on left and right side of the image
bottom_limit = 0.98 # Croppint bottom of the image which mostly is the vehicle hood.

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
~~~

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 540, 482      | 256, 20        | 
| 740, 482      | 1024, 20      |
| 1062, 705     | 1024, 720      |
| 217, 705      | 256, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Line tracking process is defined in find_window_centroid() method in `LineTracker.py`. I defined LineTracker class which contains variables for sliding window and line tracking method. I used 'sliding window method' to find centroids, and I used cumulated centroids data to find the most reasonable centroid points.  

Before coming up with ideas to use cumulated data, I calculated centroids from the scratch for every frame which results in plotting lines jumping around (totally unstable). Since position of the lanes are continuous throughout the video, I decide to use the information from the previous frames.

As I don't have any information for the very first frame of the video, following sliding window method, I summed up the white points for each column and supposed those were lane points (Line 29-33 in `LineTracker.py`).

~~~ python
if len(self.recent_center) == 0:
    histogram = np.sum(warped[m_h//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_current = np.argmax(histogram[: midpoint-50])
    rightx_current = np.argmax(histogram[midpoint:]) + midpoint
~~~

Everytime sliding windows, counting white points and if the number of points is bigger than thershold(`minpix = 100`), it is regarded as valid lane points and take average x-axis position as new centroids (Line 74-77).  

~~~python
if len(good_left_inds) > minpix:
    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
if len(good_right_inds) > minpix:
    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
~~~

From second frame of the video, I used centroids information cumulated in LineTracker.recent_center variable.  I I supposed new centroid for the new frame won't be far away from the previous one. Placing window at the centroid of the previous frame, and repeating the calculation I mentioned above.

After I get new centroid, I added one more logic to tune the value. I adjust smooth factor and alpha to combine new centroid value and the average of the previous data (Line 85-91). Sometimes I could pick wrong points as lane lines, but by adjusting alpha value I can figure out how much I should count on new information. If alpha equals 1, then I totally rely on data from the previous frames, and if alpha equals 0, I only use newly calculated centroids information.  

`new_centroid = alpha * (average value of previous (self.smooth_factor) frames) + (1-alpha) * new_centroid `  
~~~python
alpha = 0.2
if len(self.recent_center) > 0:
    center = np.array(self.recent_center, dtype=np.uint32)
    leftx_current = int(
        alpha * np.mean(center[-self.smooth_factor:], axis=0)[window][0] + (1 - alpha) * leftx_current)
    rightx_current = int(
        alpha * np.mean(center[-self.smooth_factor:], axis=0)[window][1] + (1 - alpha) * rightx_current
~~~ 

After calculating centroids for each frame, I calculate ploynomial coefficient. For this, I just use centroid points. I have used all of the white points in each window to get polynomial coefficients, but it causes polynomial lines jumping around the frames. Because centroid positions are good representations for the lanes, I thought it should be used to draw poly lines (Line 97-102).

And then I updated ployfit coefficients to calculate curvature of the lanes (Line 105-106).

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

1. Calculate Curvature  
This code is defined in calculate_curvature() in Line class (`Line.py`). 

Right after calculating centroids in tracker instance, I get polynomial coefficient both for left and right lanes. And because they are calculated based on pixel, they need to be changed into meter-based number.  

Meter per pixel information was given in the lecture, so I used this value to calculate curvature. First, need to get polynomial coefficents again based on meter information(Line 16-17, `Line.py`). And then under the formula, calculate curvature for both left and right lanes, I used the average value to print out.

Curvature value normally stays between 1km ~ 3km but sometimes it overs 5km especially on straight lanes. As you can see in `final_video.mp4` I can detect lanes most of the time, but curvature value jumps around (**I'm not sure how to tune this vaule. NEED IMPROVEMENT**).  


2. Position of the vehicle
Calculate vehicle position was much easier than calculating curvature. The code for this is in distance_from_center() in `Line.py`. For the bottom of the image, I have left and right lane centroids so that I can calculate mid-point of the lane in the image. Since mid-point of the image (640) is regarded as the center of the vehicle, all I have to do is to see the difference of the mid-point of the lane and image(640) (Line 35). 

If the value is positive, it means vehicle is right of the center (Vehicle center is on right side of the lane center). And difference need to be changed into meter by multiplying meter/pixel for x-axis.  



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Last thing in pipeline is to plot my result back on the original image. This code is described in Line 96-116 in `pipeline.py`.  

I got inverse matrix from `Minv = cv2.getPerspectiveTransform(dst, src)`, and use it to draw polynomial plots back on the original image.  

~~~python 
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
~~~

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video.

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. Find good threshold for the video
I need to find good threshold combination to accurately detect lane lines from the image. For the project_video.mp4, which mostly shows clear lanes, it was not that hard. But for challenge_video.mp4 and harder_challenge_video.mp4, which contain shadow and darker lanes, my filter still doesn't work well (especially for harder_challenge_video.mp4). I guess my filter does not detect dark yellow lines. I need to think more how to improve filters.

2. Polynomial Lines are jumping around
When I try to find window centroids from the scratch for each frame, I encounted polynomial lines jumped around. Instead of calculating centroids over and over again, I used information from the last frame as described in `Pipeline Question #4`. This is based on the idea that lanes are continuous, so the difference of the x-position of previous frame and the next one. After applying this logic, polynomial lines stay stable.

But there is possibility that mis-detected lines keep having effect on calculating new centroids. This is what happens in harder_challenge_video.mp4. Once wrong position has been detected, it takes long to recover. I need to improve logics how to find better solution for recovery.

