import cv2
import numpy as np
import pywt

METHODS = ['median', 'gaussian', 'bilateral', 'nlm', 'wavelets']

def denoise_image(
    image: np.ndarray, mode: str,
    kernel_size: int = 3, sigma: float = 1,
    nlm_h: float = 10, nlm_hcolor: float = 10, nlm_template_window_size: int = 7, nlm_search_window_size: int = 21, 
    wavelet: str = 'bior1.3', wavelet_threshold: float = 0.5, wavelet_mode: 'str' = 'soft'
) -> np.ndarray:
    """
    Removes noise from an image.

    Parameters
    ----------
    image: np.ndarray
        Input noisy image. For all modes, be sure to pass the image in ```np.float64```
        data type. In case of using "nlm" be sure to use the BGR color space
        as input, as it transforms to CieLAB under the hood.
    mode: {"median", "gaussian", "bilateral", "nlm", "wavelets"}
        Type of denoising algorithm to use.
    kernel_size: int
        Size of the kernel to apply for modes ```median```, ```gaussian```, and ```bilateral```.
        Defaults to ```3```.
    sigma: float
        Standard deviation for modes ```gaussian``` and ```bilateral```.
        Defaults to ```1```.
    nlm_h: float
        Value defining the filter strength on luminant component for mode ```nlm```.
        Higher values removes noise better but also removes details.
        Defaults to ```10```.
    nlm_hcolor: float
        Same as "nlm_h" but for color components for mode ```nlm```. Defaults to 10.
    nlm_template_window_size: int
        Size in pixels of the template patch that is used to compute weights for mode ```nlm```.
        Should be odd. Defaults to ```7```.
    nlm_search_window_size: int
        Size in pixels of the window that is used to compute weighted average
        for given pixel for mode ```nlm```. Should be odd. Affect performance linearly:
        greater searchWindowsSize - greater denoising time. Defaults to 21.
    wavelet: str
        Wavelet to apply to the image for mode ```wavelets```. Use one of those
        defined by the PyWavelets library. Defaults to ```bior1.3```.
    wavelet_threshold: float
        Threshold to apply in the wavelet frequency domain to get rid of high
        frequencies (noise) for mode ```wavelets```. Defaults to ```0.5```.
    wavelet_mode: {"soft", "hard", "garrote", "greater", "less"}
        Mode to threshold the wavelets for mode ```wavelets```. For more info on
        each mode, go to the PyWavelets documentation. Defaults to "soft".
    
    Returns
    -------
    np.ndarray
        Denoised image.
    """
    
    if mode == 'median':
        denoised_image = cv2.medianBlur(image, kernel_size)
    elif mode == 'gaussian':
        denoised_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    elif mode == 'bilateral':
        denoised_image = cv2.bilateralFilter(image, kernel_size, sigma, sigma)
    elif mode == 'nlm':
        denoised_image = cv2.fastNlMeansDenoisingColored(image, dst=None, h=nlm_h, hColor=nlm_hcolor, templateWindowSize=nlm_template_window_size, searchWindowSize=nlm_search_window_size)
    elif mode == 'wavelets':
        coefs = pywt.dwt2(image, wavelet)
        coefs = pywt.threshold(coefs, wavelet_threshold, mode=wavelet_mode)
        denoised_image = pywt.idwt2(coefs, wavelet,)
    else:
        raise ValueError(f"Mode not implemented, choose one of {METHODS}")
    return denoised_image

if __name__ == '__main__':
    image = cv2.imread('datasets/qsd1_w3/00023.jpg')
    cv2.imshow('Original Image', image)
    cv2.waitKey(0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    denoised_image = denoise_image(image, 5, 'gaussian', sigma=2)


    denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_Lab2BGR)
    cv2.imshow('Denoised Image', denoised_image)
    cv2.waitKey(0)
