import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def compute_metrics(prediction: np.ndarray, groundtruth: np.ndarray) -> tuple[float]:
    """
    Computes MSE, PSNR and SSIM between two images. These images
    should be np.float64.

    Parameters
    ----------
    prediction: np.ndarray
        Predicted image.
    groundtruth: np.ndarray
        Groundtruth image.
    
    Returns
    -------
    tuple[float]
        A tuple containing the values for the MSE, PSNR and SSIM in that order.
    """
    if prediction.dtype == np.uint8:
        prediction = prediction.astype(np.float64) / 255
    if groundtruth.dtype == np.uint8:
        groundtruth = groundtruth.astype(np.float64) / 255

    mse = mean_squared_error(groundtruth, prediction)
    psnr = peak_signal_noise_ratio(groundtruth, prediction)
    ssim = structural_similarity(groundtruth, prediction, data_range=1, channel_axis=2)

    return mse, psnr, ssim
