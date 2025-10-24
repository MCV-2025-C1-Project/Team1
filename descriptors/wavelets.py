import cv2
import numpy as np
import histograms

import pywt

def wavelets_descriptor(image: np.ndarray, wavelet: str, bins: int, num_windows: int, num_dimensions: int = 1) -> np.ndarray:
    if image.dtype != np.float64:
        image = image.astype(np.float64)
        image = image / 255

    # Extract wavelets
    cA, (cH, cV, cD) = pywt.dwt2(image, wavelet)

    wavelets_decomposition = np.stack([cH, cV, cD], axis=-1)
    wavelets_decomposition = (wavelets_decomposition * 255).astype(np.uint8)

    # Generate histogram
    hist = histograms.gen_hist(wavelets_decomposition, bins, num_windows, num_dimensions)
    return hist

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Wavelets must work side by side with histograms
    # Main idea: use wavelets to decompose image into frequency representation (horizontal, vertical, diagonal)
    # then extract histogram from these representations. I can use block histograms as the results from last week
    # suggest they are the best for matching local features.
    import pywt.data

    # Load image
    gray_image = pywt.data.camera()
    image = cv2.imread('datasets/qsd1_w3/00001.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_image = gray_image / 255

    w_desc = wavelets_descriptor(gray_image, 'bior1.3', 256)

    print('Yay!')