
import cv2
import numpy as np
from typing import List, Tuple, Optional

def akaze_descriptor(
    img: np.ndarray,
    # AKAZE params (tweak as needed)
    descriptor_type: int = cv2.AKAZE_DESCRIPTOR_MLDB,  # binary
    descriptor_size: int = 0,        # 0 = full size for MLDB
    descriptor_channels: int = 3,    # typically 1, 2, or 3 for MLDB
    threshold: float = 0.001,        # detection threshold
    n_octaves: int = 4,
    n_octave_layers: int = 4,
    diffusivity: int = cv2.KAZE_DIFF_PM_G2,
    mask: Optional[np.ndarray] = None,
):
    """
    Compute AKAZE keypoints and descriptors for an image.

    Returns:
        keypoints: list of cv2.KeyPoint
        descriptors: (N, D) np.ndarray or None if no keypoints
                     (binary uint8 for MLDB; float32 for non-MLDB types)
    """
    # Ensure grayscale for stable behavior
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    akaze = cv2.AKAZE_create(
        descriptor_type=descriptor_type,
        descriptor_size=descriptor_size,
        descriptor_channels=descriptor_channels,
        threshold=threshold,
        nOctaves=n_octaves,
        nOctaveLayers=n_octave_layers,
        diffusivity=diffusivity,
    )

    keypoints, descriptors = akaze.detectAndCompute(gray, mask)
    return keypoints, descriptors

if __name__ == "__main__":
    path = "../../qsd1_w4/00001.jpg"
    img = cv2.imread(path)
    kps, des = akaze_descriptor(img)
    print(f"Keypoints: {len(kps)}")
    print(f"Descriptors shape: {None if des is None else des.shape}, dtype: {None if des is None else des.dtype}")

    # Draw keypoints with scale and orientation
    out = cv2.drawKeypoints(
        img, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("akaze_keypoints.jpg", out)


