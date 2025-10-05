import cv2
import numpy as np


def clahe_preprocessing(image, color_space, clip=2.0, grid_size=(8,8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) on the luminance channel.

    This function enhances local contrast. The luminance (or value) channel is isolated
    depending on the specified color space, processed with CLAHE, and merged back with
    the untouched chroma channels.

    Parameters
    ----------
    image : numpy.ndarray
        Input image already converted to the target color space specified by `color_space`.
        Expected dtype uint8.
    color_space : {"gray_scale", "lab", "ycbcr", "rgb", "hsv"}
        Color space interpretation used to decide which channel is treated as luminance:
        - ``"gray_scale"``: CLAHE is applied directly.
        - ``"lab``" or "ycbcr": First channel is treated as luminance.
        - ``"rgb"``: Temporarily converted to HSV; V channel enhanced; converted back.
        - ``"hsv"``: Third channel (value) enhanced directly.
    clip : float, optional
        CLAHE clip limit (higher values give more contrast amplification). Default ``2.0``.
    grid_size : tuple[int, int], optional
        Size (in tiles) of the CLAHE grid. Default ``(8, 8)``.

    Returns
    -------
    numpy.ndarray
        Image with enhanced luminance in the same color space as input.

    Notes
    -----
    For ``"rgb"`` the function internally converts to HSV, enhances V, then converts back.
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
    Apply global histogram equalization on the luminance channel (or full grayscale image).

    Parameters
    ----------
    image : numpy.ndarray
        Input image in the specified color space (uint8).
    color_space : {"gray_scale", "lab", "ycbcr", "rgb", "hsv"}
        Determines how luminance is isolated for equalization.

    Returns
    -------
    numpy.ndarray
        Equalized image in the same color space.
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
    Apply gamma correction via lookup table.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or color), uint8.
    gamma : float, optional
        Gamma exponent (>1 darkens mid-tones, <1 brightens). Default ``1.2``.

    Returns
    -------
    numpy.ndarray
        Gamma-corrected image.
    """
    inv = 1.0 / max(gamma, 1e-6)
    lut_table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)]).astype(np.uint8)
    preprocessed_image = cv2.LUT(image, lut_table)
    return preprocessed_image

def contrast(image, alpha=1.2, beta=0.0):
    """
    Apply linear contrast/brightness adjustment.

    Performs: ``out = alpha * image + beta`` (then clipped to uint8).

    Parameters
    ----------
    image : numpy.ndarray
        Input uint8 image.
    alpha : float, optional
        Contrast gain (>1 increases contrast). Default ``1.2``.
    beta : float, optional
        Brightness shift added after scaling. Default ``0.0``.

    Returns
    -------
    numpy.ndarray
        Adjusted image.
    """
    preprocessed_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return preprocessed_image

def gaussian_blur(image, kernel_size=3, sigma=0):
    """
    Apply Gaussian blur.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or color).
    kernel_size : int, optional
        Base kernel size (will be forced to nearest odd). Default ``3``.
    sigma : float, optional
        Gaussian standard deviation; 0 lets OpenCV compute it. Default ``0``.

    Returns
    -------
    numpy.ndarray
        Blurred image.

    Notes
    -----
    Kernel size is adjusted to be odd to satisfy OpenCV requirements.
    """
    kernel_size = kernel_size + (1 - kernel_size % 2)
    preprocessed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return preprocessed_image

def median_blur(image, kernel_size=3):
    """
    Apply median filtering (edge-preserving noise reduction).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (uint8).
    kernel_size : int, optional
        Base kernel size (forced to odd). Default ``3``.

    Returns
    -------
    numpy.ndarray
        Denoised image.
    """
    kernel_size = kernel_size + (1 - kernel_size % 2)
    preprocessed_image = cv2.medianBlur(image, kernel_size)
    return preprocessed_image

def bilateral(image, pixel_diameter=7, sigma_color=50, sigma_space=50):
    """
    Apply bilateral filtering (edge-preserving smoothing).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or color).
    pixel_diameter : int, optional
        Diameter of each pixel neighborhood. Default ``7``.
    sigma_color : float, optional
        Filter sigma in the color space (larger => more colors mixed). Default ``50``.
    sigma_space : float, optional
        Filter sigma in coordinate space (larger => farther pixels influence each other). Default ``50``.

    Returns
    -------
    numpy.ndarray
        Smoothed image with edges retained.
    """
    preprocessed_image = cv2.bilateralFilter(image, pixel_diameter, sigma_color, sigma_space)
    return preprocessed_image

def unsharp(image, kernel_size=3, sigma=0, amount=1.0, gamma=0):
    """
    Apply unsharp masking (sharpening via blurred subtraction).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (uint8).
    kernel_size : int, optional
        Base Gaussian kernel size (forced to odd). Default ``3``.
    sigma : float, optional
        Gaussian sigma (0 => auto). Default ``0``.
    amount : float, optional
        Sharpening strength; higher values amplify edges more. Default ``1.0``.
    gamma : float, optional
        Scalar added after weighted sum (bias). Default ``0``.

    Returns
    -------
    numpy.ndarray
        Sharpened image.
    """
    kernel_size = kernel_size + (1 - kernel_size % 2)
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    preprocessed_image = cv2.addWeighted(image, 1 + amount, blurred_image, -amount, gamma)
    return preprocessed_image