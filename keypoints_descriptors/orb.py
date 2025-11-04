import cv2
import numpy as np
from typing import List, Tuple, Optional

def orb_descriptor(
    img: np.ndarray,
    n_features: int = 1000,       # cap on number of keypoints to keep
    scale_factor: float = 1.2,    # image pyramid scaling between levels
    n_levels: int = 8,            # number of pyramid levels
    edge_threshold: int = 31,     # size of border where features are not detected
    first_level: int = 0,
    WTA_K: int = 2,              # number of points that produce each BRIEF comparison (2,3,4)
    score_type: int = cv2.ORB_HARRIS_SCORE,  # or cv2.ORB_FAST_SCORE
    patch_size: int = 31,         # size of the patch used by the BRIEF descriptor
    fast_threshold: int = 20,     # FAST corner detector threshold
    mask: Optional[np.ndarray] = None,
):
    """
    Compute ORB keypoints and binary descriptors for an image.

    Returns:
        keypoints: list of cv2.KeyPoint
        descriptors: (N, 32) uint8 array of binary descriptors, or None if no keypoints
    """
    # Ensure grayscale (ORB works on single-channel)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold,
        firstLevel=first_level,
        WTA_K=WTA_K,
        scoreType=score_type,
        patchSize=patch_size,
        fastThreshold=fast_threshold,
    )

    keypoints, descriptors = orb.detectAndCompute(gray, mask)
    return keypoints, descriptors

if __name__ == "__main__":
    path = "../qsd1_w4/00001.jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    kps, des = orb_descriptor(img)
    print(f"Keypoints: {len(kps)}")
    print("Descriptors shape:", None if des is None else des.shape)  # typically (N, 32)

    # Draw keypoints with orientation (scale isn’t drawn as circles for ORB like SIFT’s "rich" mode, but this works)
    out = cv2.drawKeypoints(
        img, kps, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    cv2.imwrite("orb_keypoints.jpg", out)
