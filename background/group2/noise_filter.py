import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import structural_similarity as ssim

def fourier_noise_score(img, radius_ratio=0.25, normalize=True, scale=100):
    """
    Computes a noise score based on the amount of high-frequency energy
    in the image's Fourier magnitude spectrum.

    Parameters:
    ----------
    img : np.ndarray
        Input image (BGR or grayscale).
    radius_ratio : float, optional
        Fraction of the image size defining the radius of the low-frequency
        region. Smaller values → more aggressive high-frequency detection.
        Default = 0.25
    normalize : bool, optional
        Whether to normalize the magnitude spectrum before computing ratios.
        Default = True
    scale : float, optional
        Scaling factor for the final score (for readability). Default = 100

    Returns:
    -------
    noise_score : float
        The ratio of high-frequency energy to total spectral energy.
        Higher values → more noise.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Compute Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    # Normalize magnitude if needed
    if normalize:
        magnitude /= magnitude.max() + 1e-8

    # Create high-frequency mask
    h, w = gray.shape
    crow, ccol = h // 2, w // 2
    radius = int(min(crow, ccol) * radius_ratio)
    y, x = np.ogrid[:h, :w]
    mask_high = ((x - ccol)**2 + (y - crow)**2) >= radius**2

    # Compute ratio of high-frequency energy to total energy
    high_energy = np.sum(magnitude[mask_high])
    total_energy = np.sum(magnitude)
    hf_ratio = high_energy / (total_energy + 1e-8)

    noise_score = hf_ratio * scale
    return noise_score

def analyze_noise(noisy, clean):
    # Convert to grayscale for simplicity
    gray_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    gray_clean = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    
    # Compute noise residual
    noise = gray_noisy.astype(np.float32) - gray_clean.astype(np.float32)
    
    mean = np.mean(noise)
    std = np.std(noise)
    min_val, max_val = np.min(noise), np.max(noise)
    
    plt.hist(noise.ravel())
    plt.title("Noise distribution")
    plt.show()
    
    return mean, std, min_val, max_val

# Non-local Means
def remove_noise_nlmeans(img, h=10, templateWindowSize=7, searchWindowSize=21):
    """
    Removes Gaussian-like noise preserving textures and edges.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)

# Bilateral filter
def remove_noise_bilateral(img, d=9, sigmaColor=75, sigmaSpace=75):
    """
    Reduces Gaussian noise while keeping edges sharp.
    """
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

# Gaussian blur
def remove_noise_gaussian(img, ksize=5, sigma=1.0):
    """
    Reduces mild Gaussian noise with a Gaussian kernel.
    """
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

def evaluate_denoising(denoised, gt):
    """
    Computes MSE and SSIM between denoised image and its ground truth.
    """
    denoised = denoised.astype(np.uint8)
    gt = gt.astype(np.uint8)

    m = mse(gt, denoised)
    s = ssim(gt, denoised, channel_axis=2)
    return m, s

def mse_to_psnr(mse_value, max_pixel_value=255):
    """
    Converts Mean Squared Error (MSE) to Peak Signal-to-Noise Ratio (PSNR).

    Parameters
    ----------
    mse_value : float
        Mean Squared Error between two images.
    max_pixel_value : int, optional
        Maximum possible pixel value (default 255 for 8-bit images).

    Returns
    -------
    psnr_value : float
        PSNR value in decibels (dB).
    """
    if mse_value == 0:
        return float('inf')  # perfect match → infinite PSNR
    
    psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse_value)
    return psnr_value

def remove_noise_median(img, ksize=3):
    return cv2.medianBlur(img, ksize)