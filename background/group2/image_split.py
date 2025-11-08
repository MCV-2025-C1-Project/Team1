import cv2
import numpy as np

def split_images(img, debug=False, min_frac=0.25):
    """
    Split an image into two artworks if they are separated horizontally (left/right)
    or vertically (top/bottom). Otherwise return the original.

    Args:
        img: Input image (BGR)
        debug: If True, print details and show intermediate masks
        min_frac: minimum fraction (0..1) of the original size each side must keep
                  along the split dimension (default 25%)

    Returns:
        (True, (imgA, imgB)) if split found  [A,B is left->right or top->bottom]
        (False, img) otherwise
    """
    H, W = img.shape[:2]

    # 1) Gradient on HSV S and V to find textured regions (artworks)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # input is BGR
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    kernel_grad = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad_s = cv2.morphologyEx(s_channel, cv2.MORPH_GRADIENT, kernel_grad)
    grad_v = cv2.morphologyEx(v_channel, cv2.MORPH_GRADIENT, kernel_grad)

    gradient_magnitude = cv2.add(grad_s, grad_v)

    # 2) Threshold -> binary mask of "interesting" regions
    max_value = float(gradient_magnitude.max())
    thr = 0.1 * max_value
    binary = (gradient_magnitude >= thr).astype(np.uint8)

    # 3) Morphology to clean (slightly anisotropic helps fill painting interiors)
    #    We apply a close then open to fill holes and remove tiny noise.
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    opened = cv2.morphologyEx(opened, cv2.MORPH_OPEN,  k_open)

    # 4) Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    if num_labels < 3:
        if debug: print("Only one foreground component -> no split")
        return False, img

    # sort components by area (skip background=0)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda x: x[1], reverse=True)
    if len(areas) < 2:
        if debug: print("Less than 2 components -> no split")
        return False, img

    # keep two largest
    label1, _ = areas[0]
    label2, _ = areas[1]

    x1, y1, w1, h1 = stats[label1, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
    x2, y2, w2, h2 = stats[label2, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]

    # helper to compute separation gap and split index for one axis
    def check_axis(a1, a2, size, min_keep_px):
        """
        a1=(start1, length1), a2=(start2, length2) along a 1D axis of given 'size'.
        Returns (is_separated, gap, split_index, dims_ok)
        """
        s1, l1 = a1; e1 = s1 + l1
        s2, l2 = a2; e2 = s2 + l2

        # sort so a is left/top, b is right/bottom
        if s1 <= s2:
            left_s, left_e = s1, e1
            right_s, right_e = s2, e2
        else:
            left_s, left_e = s2, e2
            right_s, right_e = s1, e1

        separated = left_e < right_s  # strict gap
        gap = (right_s - left_e) if separated else 0

        # split at midpoint of the gap
        split_idx = (left_e + right_s) // 2 if separated else None

        # size check: each side after split must be >= min_keep_px
        dims_ok = False
        if separated and split_idx is not None:
            left_size  = split_idx
            right_size = size - split_idx
            dims_ok = (left_size >= min_keep_px) and (right_size >= min_keep_px)

        return separated, gap, split_idx, dims_ok

    min_w = int(min_frac * W)
    min_h = int(min_frac * H)

    # horizontal (left/right) check along X
    sep_h, gap_h, split_x, ok_h = check_axis((x1, w1), (x2, w2), W, min_w)
    # vertical (top/bottom) check along Y
    sep_v, gap_v, split_y, ok_v = check_axis((y1, h1), (y2, h2), H, min_h)

    # decide best orientation: prefer valid split with the larger normalized gap
    choice = None
    if sep_h and ok_h:
        choice = ("horizontal", gap_h / max(1, W), int(split_x))
    if sep_v and ok_v:
        cand = ("vertical", gap_v / max(1, H), int(split_y))
        if choice is None or cand[1] > choice[1]:
            choice = cand

    if choice is None:
        if debug:
            print("No valid horizontal/vertical separation that passes size checks")
            print(f"Horizontal: separated={sep_h}, ok={ok_h}, gap={gap_h}")
            print(f"Vertical:   separated={sep_v}, ok={ok_v}, gap={gap_v}")
        return False, img

    orientation, _, split_idx = choice

    if debug:
        print(f"SPLIT: {orientation} at {split_idx} "
              f"(min_frac={min_frac:.2f}, W={W}, H={H})")

    if orientation == "horizontal":
        left_img  = img[:, :split_idx].copy()
        right_img = img[:, split_idx:].copy()
        return True, (left_img, right_img)
    else:
        top_img    = img[:split_idx, :].copy()
        bottom_img = img[split_idx:, :].copy()
        return True, (top_img, bottom_img)
