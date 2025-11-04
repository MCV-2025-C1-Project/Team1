import cv2
import numpy as np
from typing import List, Tuple, Optional

def sift_descriptor(
    img: np.ndarray,
    nfeatures: int = 0,
    nOctaveLayers: int = 3,
    contrastThreshold: float = 0.04,
    edgeThreshold: int = 10,
    sigma: float = 1.6,
    mask: Optional[np.ndarray] = None,
):
    """
    Compute SIFT keypoints and 128-d descriptors for an image.

    Returns:
        keypoints: list of cv2.KeyPoint
        descriptors: (N, 128) float32 array or None if no keypoints
    """
    # Ensure grayscale for stable behavior
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )
    keypoints, descriptors = sift.detectAndCompute(gray, mask)
    return keypoints, descriptors

if __name__ == "__main__":
    path = "../qsd1_w4/00001.jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    kps, des = sift_descriptor(img)
    print(f"Keypoints: {len(kps)}")
    print(f"Descriptors shape: {des.shape}")

    # Draw keypoints with scale and orientation
    out = cv2.drawKeypoints(
        img, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("sift_keypoints.jpg", out)
