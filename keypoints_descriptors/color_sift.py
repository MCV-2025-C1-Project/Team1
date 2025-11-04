import cv2
import numpy as np
from typing import List, Tuple, Optional

def _opponent_channels(img_bgr: np.ndarray):
    """Return opponent color space channels (float32, same size as image, range approx [-255, 255])."""
    B, G, R = cv2.split(img_bgr.astype(np.float32))
    O1 = (R - G) / np.sqrt(2.0)
    O2 = (R + G - 2.0 * B) / np.sqrt(6.0)
    O3 = (R + G + B) / np.sqrt(3.0)  # intensity-like
    return O1, O2, O3

def _rootsift_transform(desc: Optional[np.ndarray]):
    """Apply RootSIFT (L1-normalize then element-wise sqrt). Keeps dtype float32."""
    if desc is None:
        return None
    eps = 1e-12
    desc = desc.astype(np.float32)
    desc /= (np.sum(desc, axis=1, keepdims=True) + eps)
    return np.sqrt(desc, out=desc)

def color_sift_descriptor(
    img: np.ndarray,
    mode: str = "opponent",  # "opponent" or "rgb"
    nfeatures: int = 0,
    nOctaveLayers: int = 3,
    contrastThreshold: float = 0.04,
    edgeThreshold: int = 10,
    sigma: float = 1.6,
    mask: Optional[np.ndarray] = None,
    use_rootsift: bool = False,
):
    """
    Compute Color-SIFT descriptors (384-D) by concatenating 3 per-channel SIFTs.
      - mode="opponent": O1,O2,O3 channels (recommended for paintings)
      - mode="rgb": R,G,B channels

    Returns:
        keypoints: list[cv2.KeyPoint] detected on intensity (O3 or gray)
        descriptors: (N, 384) float32 or None if no keypoints
    """
    if img.ndim != 3:
        raise ValueError("Color-SIFT expects a color image (BGR).")

    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        nOctaveLayers=nOctaveLayers,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
        sigma=sigma
    )

    if mode.lower() == "opponent":
        O1, O2, O3 = _opponent_channels(img)
        # Detect keypoints on intensity-like channel (O3)
        kps = sift.detect(O3 if mask is None else O3.astype(np.uint8), mask)
        # Compute per-channel descriptors at the same keypoints
        kps, d1 = sift.compute(O1, kps)
        _,  d2 = sift.compute(O2, kps)
        _,  d3 = sift.compute(O3, kps)
    elif mode.lower() == "rgb":
        B, G, R = cv2.split(img)  # still BGR order; we'll pass R,G,B for clarity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect on intensity (gray)
        kps = sift.detect(gray if mask is None else gray, mask)
        kps, dR = sift.compute(R.astype(np.uint8), kps)
        _,  dG = sift.compute(G.astype(np.uint8), kps)
        _,  dB = sift.compute(B.astype(np.uint8), kps)
        d1, d2, d3 = dR, dG, dB
    else:
        raise ValueError("mode must be 'opponent' or 'rgb'")

    if d1 is None or d2 is None or d3 is None:
        return kps, None

    # Concatenate → 384-D
    desc = np.hstack([d1, d2, d3]).astype(np.float32)

    if use_rootsift:
        # Apply per-channel RootSIFT before concatenation (slightly better),
        # or apply once after concatenation (simpler). We'll do after concat for simplicity.
        desc = _rootsift_transform(desc)

    return kps, desc



def match_value_from_descriptors(
    desA,
    desB,
    ratio: float = 0.75
):
    """
    Compute a single 'match value' between two descriptor sets using
    k-NN (k=2) + Lowe ratio filtering.

    Args:
        desA: (NA, D) float32 descriptors for image A
        desB: (NB, D) float32 descriptors for image B
        ratio: Lowe ratio threshold (default 0.75)

    Returns:
        score: float in [0, 1]  -> good_matches / min(NA, NB)
        good_matches: int       -> count of matches passing the ratio test
        total_attempts: int     -> number of kNN queries performed (= NA)
    """
    # Guard cases
    if desA is None or desB is None or len(desA) == 0 or len(desB) == 0:
        return 0.0, 0, 0

    # FLANN params for float descriptors (SIFT/Color-SIFT)
    D = desA.shape[1]
    index_params = dict(algorithm=1, trees=8 if D > 128 else 5)  # KD-tree
    search_params = dict(checks=64 if D > 128 else 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # k-NN with k=2 for Lowe ratio
    knn = flann.knnMatch(desA, desB, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    good_matches = len(good)
    total_attempts = len(knn)  # typically == len(desA)

    # Normalize by the smaller set size to keep score comparable across images
    denom = max(1, min(len(desA), len(desB)))
    score = good_matches / denom

    return float(score), int(good_matches), int(total_attempts)


# Example usage (mirrors your SIFT script)
if __name__ == "__main__":
    path = "../qsd1_w4/00001.jpg"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    # Opponent-SIFT (recommended)
    kps, des = color_sift_descriptor(img, mode="opponent", use_rootsift=False)
    print(f"Keypoints: {len(kps)}")
    print("Color-SIFT descriptors shape:", None if des is None else des.shape)  # (N, 384)

    # Visualize keypoints (same as before)
    out = cv2.drawKeypoints(img, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("color_sift_keypoints.jpg", out)
