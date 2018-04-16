import numpy as np


class Line():
    def __init__(self, xm=3.7/700, ym=30/720):
        self.left_fit = []
        self.right_fit = []
        self.xm_per_pix = xm  # meters per pixel in horizontal axis
        self.ym_per_pix = ym  # meters per pixel in vertical axis
        self.leftx = []
        self.lefty = []
        self.rightx = []
        self.righty = []

    def calculate_curvature(self, y=720):
        left_fit_meter = np.polyfit(self.lefty*self.ym_per_pix, self.leftx*self.xm_per_pix, 2)
        right_fit_meter = np.polyfit(self.righty*self.ym_per_pix,
                                     self.rightx*self.xm_per_pix, 2)

        left_curve = (
            (1 + (2*left_fit_meter[0]*y*self.ym_per_pix + left_fit_meter[1])**2) ** 1.5) / np.absolute(2*left_fit_meter[0])

        right_curve = (
            (1 + (2*right_fit_meter[0]*y*self.ym_per_pix + right_fit_meter[1])**2) ** 1.5) / np.absolute(2*right_fit_meter[0])

        return left_curve, right_curve

    def distance_from_center(self, width=1280):
        left_pos = self.leftx[0]
        right_pos = self.rightx[0]

        center = int((right_pos + left_pos) // 2)
        # Positive value means vehicle is right of center
        # Negative value means vehicle is left of center
        return (width//2 - center) * self.xm_per_pix
