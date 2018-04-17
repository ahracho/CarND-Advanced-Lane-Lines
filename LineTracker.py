import numpy as np
import cv2


class LineTracker():
    def __init__(self, nwindows, margin, smooth_factor=10):
        # centroids found by find_window_centroid() function
        self.recent_center = []
        # number of windows for sliding window method
        self.nwindows = nwindows
        # 1/2 of the window width
        self.margin = margin
        # number of frames used for smoothing lanes
        self.smooth_factor = smooth_factor

    def find_window_centroid(self, warped, line):
        # I used 'sliding windown method' to find centroids
        # Basically, I used cumulated centroids data to find the most reasonable centroid points

        m_h = warped.shape[0]
        m_w = warped.shape[1]

        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # On the very first frame of the video,
        # start to find centroid points based on density of white dots
        if len(self.recent_center) == 0:
            histogram = np.sum(warped[m_h//2:, :], axis=0)
            midpoint = np.int(histogram.shape[0]//2)
            leftx_current = np.argmax(histogram[: midpoint-50])
            rightx_current = np.argmax(histogram[midpoint:]) + midpoint

        # Set minimum number of pixels found to recenter window
        minpix = 100
        
        # Create empty lists to receive left / right lane pixel indices and centroids
        left_lane_inds = []
        right_lane_inds = []
        centroids = []

        window_height = int(m_h//self.nwindows)

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Set window centroid based on the last one
            # since lane lines will be away that much from the last frame
            if len(self.recent_center) > 0:
                leftx_current = self.recent_center[-1][window][0]
                rightx_current = self.recent_center[-1][window][1]
            
            # Identify window boundaries in x and y (and right and left)
            win_y_low = m_h - (window+1)*window_height
            win_y_high = m_h - window*window_height
            win_xleft_low = max(leftx_current - self.margin, 0)
            win_xleft_high = min(leftx_current + self.margin, m_w)
            win_xright_low = max(rightx_current - self.margin, 0)
            win_xright_high = min(rightx_current + self.margin, m_w)

            # Find white points in window boundaries
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                        (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If number of points is bigger than minpix, it is regarded as feasible window
            # and take average for the new centroid
            # If not, it will use value same as the last frame
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # I added one more logic to find new centroid.
            # To prevent new centroids jump around, I weight previous value on new centroid
            # new_centroid = \
            # a * (average value of previous (self.smooth_factor) frames) + (1-a) * new_centroid
            # alpha(a) is the percentage how much I will count on previous data history 
            
            alpha = 0.2
            if len(self.recent_center) > 0:
                center = np.array(self.recent_center, dtype=np.uint32)
                leftx_current = int(
                    alpha * np.mean(center[-self.smooth_factor:], axis=0)[window][0] + (1 - alpha) * leftx_current)
                rightx_current = int(
                    alpha * np.mean(center[-self.smooth_factor:], axis=0)[window][1] + (1 - alpha) * rightx_current)
            centroids.append([leftx_current, rightx_current])

        self.recent_center.append(centroids)
        self.recent_center = self.recent_center[-self.smooth_factor:]

        # Find polynomial points based on centroids (not on all white dots)
        line.leftx = np.array(centroids, dtype=np.uint32)[:, 0]
        line.lefty = np.array(
            [m_h - i*window_height for i in range(self.nwindows)])
        line.rightx = np.array(centroids, dtype=np.uint32)[:, 1]
        line.righty = np.array(
            [m_h - i*window_height for i in range(self.nwindows)])
        
        # Fit a second order polynomial to each
        line.left_fit = np.polyfit(line.lefty, line.leftx, 2)
        line.right_fit = np.polyfit(line.righty, line.rightx, 2)
