import cv2
import numpy as np


def clahe_preprocessing(image, color_space, clip=2.0, grid_size=(8,8)):
    """
    
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid_size)

    if color_space == 'gray_scale':
        preprocessed_image = clahe.apply(image)

    elif color_space in ('lab', 'ycbcr'):
        luminance, c1, c2 = cv2.split(image)
        luminance_pr = clahe.apply(luminance)
        preprocessed_image = cv2.merge([luminance_pr, c1, c2])

    elif color_space in ('rgb', 'hsv'):
        if color_space == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        c1, c2, luminance = cv2.split(image)
        luminance_pr = clahe.apply(luminance)
        preprocessed_image = cv2.merge([c1, c2, luminance_pr])
        if color_space == 'rgb':
            preprocessed_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    
    return preprocessed_image

def hist_eq(image, color_space):
    """
    
    """
    if color_space == 'gray_scale':
        preprocessed_image = cv2.equalizeHist(image)

    elif color_space in ('lab', 'ycbcr'):
        luminance, c1, c2 = cv2.split(image)
        luminance_pr = cv2.equalizeHist(luminance)
        preprocessed_image = cv2.merge([luminance_pr, c1, c2])

    elif color_space in ('rgb', 'hsv'):
        if color_space == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        c1, c2, luminance = cv2.split(image)
        luminance_pr = cv2.equalizeHist(luminance)
        preprocessed_image = cv2.merge([c1, c2, luminance_pr])
        if color_space == 'rgb':
            preprocessed_image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

    return preprocessed_image

def gamma(image, gamma=1.2):
    """
    
    """
    inv = 1.0 / max(gamma, 1e-6)
    lut_table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)]).astype(np.uint8)
    preprocessed_image = cv2.LUT(image, lut_table)
    return preprocessed_image

def contrast(image, alpha=1.2, beta=0.0):
    """
    
    """
    preprocessed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return preprocessed_image

def gaussian_blur(image, kernel_size=3, sigma=0):
    """
    
    """
    kernel_size = kernel_size + (1 - kernel_size % 2)
    preprocessed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return preprocessed_image

def median_blur(image, kernel_size=3):
    """
    
    """
    kernel_size = kernel_size + (1 - kernel_size % 2)
    preprocessed_image = cv2.medianBlur(image, kernel_size)
    return preprocessed_image

def bilateral(image, pixel_diameter=7, sigma_color=50, sigma_space=50):
    """
    
    """
    preprocessed_image = cv2.bilateralFilter(image, pixel_diameter, sigma_color, sigma_space)
    return preprocessed_image

def unsharp(image, kernel_size=3, sigma=0, amount=1.0, gamma=0):
    """
    
    """
    kernel_size = kernel_size + (1 - kernel_size % 2)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    preprocessed_image = cv2.addWeighted(image, 1 + amount, blurred_image, -amount, gamma)
    return preprocessed_image