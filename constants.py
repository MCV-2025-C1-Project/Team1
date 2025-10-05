from typing import Literal

import cv2

COLOR_SPACES = Literal['gray_scale','rgb', 'hsv', 'lab, ycbcr']

CV2_CVT_COLORS = {
    'gray_scale': cv2.COLOR_BGR2GRAY,
    'rgb': cv2.COLOR_BGR2RGB,
    'hsv': cv2.COLOR_BGR2HSV,
    'lab': cv2.COLOR_BGR2Lab,
    'ycbcr': cv2.COLOR_BGR2YCrCb
}
