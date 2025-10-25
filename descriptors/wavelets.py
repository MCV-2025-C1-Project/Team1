"""
import cv2
import numpy as np
from . import histograms
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

"""

import numpy as np
import pywt
from . import histograms

# def wavelets_descriptor(image: np.ndarray, wavelet: str, bins: int, num_windows: int, num_dimensions: int = 1) -> np.ndarray:
#     """
#     Single-level 2D DWT per channel. For each channel:
#       - compute cH, cV, cD
#       - take magnitudes and stack into a 3-channel detail tensor
#       - rescale to 0..255 uint8
#       - feed to histograms.gen_hist (positional args)
#     Concatenate the per-channel histograms into one 1D descriptor.
#     """
#     # ensure float64 in [0,1]
#     img = image.astype(np.float64, copy=False)
#     if img.max() > 1.0:
#         img /= 255.0

#     # force 3D (H, W, C)
#     if img.ndim == 2:
#         img = img[..., None]
#     H, W, C = img.shape

#     parts = []
#     for c in range(C):
#         ch = img[..., c]
#         # 2D DWT (single level) expects 2D input
#         cA, (cH, cV, cD) = pywt.dwt2(ch, wavelet)

#         # detail magnitudes; stack => (h', w', 3)
#         det = np.stack([np.abs(cH), np.abs(cV), np.abs(cD)], axis=-1)

#         # normalize each band to 0..255 → uint8
#         dmin = det.min(axis=(0, 1), keepdims=True)
#         dmax = det.max(axis=(0, 1), keepdims=True)
#         rng  = np.maximum(dmax - dmin, 1e-12)
#         det01 = (det - dmin) / rng
#         b8 = (det01 * 255.0).astype(np.uint8)

#         # Safety: ensure 3D for gen_hist
#         if b8.ndim == 2:
#             b8 = b8[..., None]

#         # gen_hist(image, bins, blocks, hist_dims) — positional only
#         h = histograms.gen_hist(b8, bins, num_windows, num_dimensions)
#         parts.append(h)

#     # concatenate per-channel histograms into one vector
#     return np.concatenate(parts, axis=0)

import numpy as np
import pywt
from . import histograms

def wavelets_descriptor(image: np.ndarray,
                        wavelet: str,
                        bins: int,
                        num_windows: int,
                        num_dimensions: int = 1) -> np.ndarray:
    """
    Replicates the old behavior (multiply by 255 and cast to uint8), but robust:
      - If image is 3D, apply DWT per channel (pywt.dwt2 needs 2D).
      - For each channel: stack (cH, cV, cD) -> shape (h', w', 3).
      - Convert to uint8 exactly as before: (detail * 255).astype(np.uint8).
      - Feed that 3-channel array to histograms.gen_hist (positional args).
      - Concatenate the per-channel histograms.

    NOTE: No abs() and no min-max normalization are used, on purpose, to keep
    the same numerical behavior as your original code.
    """

    # ensure float64; if input is 0..255, bring to 0..1 like you had
    img = image.astype(np.float64, copy=False)
    if img.max() > 1.0:
        img /= 255.0

    # force 3D shape logic
    if img.ndim == 2:
        # single channel
        channels = [img]
    elif img.ndim == 3:
        # split per channel
        channels = [img[..., c] for c in range(img.shape[2])]
    else:
        raise ValueError(f"Unsupported image ndim={img.ndim}")

    parts = []
    for ch in channels:
        # DWT per channel (2D input)
        cA, (cH, cV, cD) = pywt.dwt2(ch, wavelet)

        # stack detail subbands exactly like before (no abs, no normalization)
        det = np.stack([cH, cV, cD], axis=-1)  # (h', w', 3)

        # old behavior: scale by 255 and cast to uint8 (values can be <0 or >1)
        det_u8 = (det * 255.0).astype(np.uint8)

        # histograms.gen_hist(image, bins, blocks, hist_dims) with positional args
        h = histograms.gen_hist(det_u8, bins, num_windows, num_dimensions)
        parts.append(h)

    # concatenate per-channel histograms into one vector
    return np.concatenate(parts, axis=0)
