from typing import Literal

import cv2

COLOR_SPACES = Literal['rgb', 'hsv', 'gray_scale']

CV2_CVT_COLORS = {
    'gray_scale': cv2.COLOR_BGR2GRAY,
    'rgb': cv2.COLOR_BGR2RGB,
    'hsv': cv2.COLOR_BGR2HSV,
    'lab': cv2.COLOR_BGR2Lab,
    'lab_processed': cv2.COLOR_BGR2Lab
}