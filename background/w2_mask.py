import cv2
import numpy as np


def fill_short_zero_runs(arr, max_zeros=600):
    """
    Fill short interior runs of zeros with linear interpolation.

    Problem: Because edge detection is not perfect, sometimes the paper edge is cut
    Solution: Fill those gaps with a gradient

    This function scans a 1-D numeric array and identifies contiguous runs
    of zeros. If a run is strictly inside the array (i.e., not touching
    the first or last element), its length is at most max_zeros, and it
    is bounded on both sides by non-zero values, then the run is replaced
    with a linear gradient interpolating between its boundary values.

    Leading and trailing zero runs are left unchanged.

    Parameters
    ----------
    arr : array-like of float or int
        Input 1-D sequence.
    max_zeros : int, optional
        Maximum zero-run length eligible for interpolation.
        Default is 500.

    Returns
    -------
    ndarray of float
        Copy of the input array with eligible zero runs replaced
        by interpolated values.

    """

    x = np.asarray(arr, dtype=float).copy()
    n = x.size
    i = 0
    while i < n:
        if x[i] != 0:
            i += 1
            continue

        # Found start of a zero-run
        start = i
        while i < n and x[i] == 0:
            i += 1
        end = i  # first non-zero after the run, or n if trailing zeros

        L = end - start  # run length

        # Fill only if run is interior and short enough
        if start > 0 and end < n and x[start-1] != 0 and x[end] != 0:
            left, right = x[start-1], x[end]
            # linear interpolation across the L interior points
            x[start:end] = np.linspace(left, right, L + 2)[1:-1]
    return x

def rect_sum(integral_image, first_col, first_row, last_col, last_row):
    """sum of ones in [y0:y1, x0:x1] inclusive-exclusive"""
    return integral_image[last_row, last_col] - integral_image[first_row, last_col] - integral_image[last_row, first_col] + integral_image[first_row, first_col]

def find_rectangle(mask, black_weight: float = 1.0, coarse_step: int = 16, min_side: int = 40, fine_step: int = 4, pad: int = 2):
    H, W = mask.shape

    binary_mask = (mask > 0).astype(np.uint8)

    image_integral = cv2.integral(binary_mask).astype(np.int32)
    
    best = (-1e18, 0, 0, 0, 0)
    for y0 in range(0, H - min_side + 1, coarse_step):
        for x0 in range(0, W - min_side + 1, coarse_step):
            for y1 in range(y0 + min_side, H + 1, coarse_step):
                for x1 in range(x0 + min_side, W + 1, coarse_step):
                    white_in = rect_sum(image_integral, x0, y0, x1, y1)
                    area = (y1 - y0) * (x1 - x0)
                    black_in = area - white_in
                    score = white_in - black_weight * black_in
                    if score > best[0]:
                        best = (score, x0, y0, x1, y1)

    score, x0, y0, x1, y1 = best

    x0r = max(x0 - 2*coarse_step, 0)
    y0r = max(y0 - 2*coarse_step, 0)
    x1r = min(x1 + 2*coarse_step, W)
    y1r = min(y1 + 2*coarse_step, H)

    best = (-1e18, x0, y0, x1, y1)
    for Y0 in range(y0r, min(y0r + 4*coarse_step, H - min_side + 1), fine_step):
        for X0 in range(x0r, min(x0r + 4*coarse_step, W - min_side + 1), fine_step):
            for Y1 in range(max(Y0 + min_side, y1r - 4*coarse_step), y1r + 1, fine_step):
                for X1 in range(max(X0 + min_side, x1r - 4*coarse_step), x1r + 1, fine_step):
                    white_in = rect_sum(image_integral, X0, Y0, X1, Y1)
                    area = (Y1 - Y0) * (X1 - X0)
                    black_in = area - white_in
                    score = white_in - black_weight * black_in
                    if score > best[0]:
                        best = (score, X0, Y0, X1, Y1)

    score, x0, y0, x1, y1 = best

    x0p = max(x0 - pad, 0); y0p = max(y0 - pad, 0)
    x1p = min(x1 + pad, W); y1p = min(y1 + pad, H)

    center_of_img = (W//2, H//2)

    if not (x0 <= center_of_img[0] <= x1 and y0 <= center_of_img[1] <= y1):
        print("Ajustant rectangle perque contingui el centre de la imatge")
        
        cx, cy = center_of_img

        if cx < x0:
            x0 = cx
        elif cx >= x1:
            x1 = min(cx + 1, W) 

        if cy < y0:
            y0 = cy
        elif cy >= y1:
            y1 = min(cy + 1, H) 

        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)

        final_mask = np.zeros_like(mask)
        final_mask[y0:y1, x0:x1] = 255

    else:
        final_mask = np.zeros_like(mask)
        final_mask[y0p:y1p, x0p:x1p] = 255

    return final_mask

def get_mask(image, color_space: str, num_std_dev: float = 1.5):
    H, W, C = image.shape

    # Get image border's colors for thresholding
    first_col = image[:, 0]
    last_col  = image[:, W - 1]
    first_row = image[0, :]
    last_row  = image[H - 1, :]

    borders = np.concatenate([first_col, last_col, first_row, last_row], axis=0)

    mean_color = np.mean(borders, axis=0)
    std_dev_color = np.std(borders, axis=0) * num_std_dev

    # threshold: shape (C, 2) -> [:,0]=lower (float), [:,1]=upper (float)
    threshold = np.stack([mean_color - std_dev_color, mean_color + std_dev_color], axis=1).astype(np.float32)

    if color_space == 'lab':
        # Drop L channel, keep a,b
        threshold = threshold[1:]
        # OpenCV Lab uses 0..255 for L,a,b in uint8 representation
        mins = np.array([0, 0], dtype=np.float32)
        maxs = np.array([255, 255], dtype=np.float32)

        lower_threshold = np.clip(np.round(threshold[:, 0]), mins, maxs).astype(np.uint8)
        upper_threshold = np.clip(np.round(threshold[:, 1]), mins, maxs).astype(np.uint8)

        # Use a,b only
        ab = image[..., 1:]
        mask_background = np.all(
            (lower_threshold.reshape(1, 1, 2) <= ab) & (ab <= upper_threshold.reshape(1, 1, 2)),
            axis=2
        ).astype(np.uint8) * 255

    elif color_space == 'hsv':

        mins = np.array([0,   0,   0  ], dtype=np.float32)
        maxs = np.array([179, 255, 255], dtype=np.float32)

        lower = np.clip(np.round(threshold[:, 0]), mins, maxs).astype(np.int16)
        upper = np.clip(np.round(threshold[:, 1]), mins, maxs).astype(np.int16)

        Hc = image[..., 0].astype(np.int16)
        Sc = image[..., 1].astype(np.int16)
        Vc = image[..., 2].astype(np.int16)

        if lower[0] <= upper[0]:
            mask_h = (Hc >= lower[0]) & (Hc <= upper[0])
        else:
            mask_h = (Hc >= lower[0]) | (Hc <= upper[0])

        mask_s = (Sc >= lower[1]) & (Sc <= upper[1])
        mask_v = (Vc >= lower[2]) & (Vc <= upper[2])

        mask_background = (mask_h & mask_s & mask_v).astype(np.uint8) * 255
    else:
        mins = np.array([0, 0, 0], dtype=np.float32)
        maxs = np.array([255, 255, 255], dtype=np.float32)

        lower_threshold = np.clip(np.round(threshold[:, 0]), mins, maxs).astype(np.uint8)
        upper_threshold = np.clip(np.round(threshold[:, 1]), mins, maxs).astype(np.uint8)

        mask_background = np.all(
            (lower_threshold.reshape(1, 1, 3) <= image) & (image <= upper_threshold.reshape(1, 1, 3)),
            axis=2
        ).astype(np.uint8) * 255

    mask_frame = cv2.bitwise_not(mask_background)

    # Keep only largest connected component (the frame)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_frame, connectivity=8)
    if num > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        largest_label = 0

    mask_frame = np.zeros_like(mask_frame)
    mask_frame[labels == largest_label] = 255

    # Fill short zero runs horizontally and vertically
    mask_frame = np.apply_along_axis(fill_short_zero_runs, 0, mask_frame)
    mask_frame = np.apply_along_axis(fill_short_zero_runs, 1, mask_frame).astype(np.uint8)

    # Optionally: find best rectangle
    # mask_frame = find_rectangle(mask_frame)

    return mask_frame

def largest_axis_aligned_rectangle(mask: np.ndarray):
    """
    mask: binary image (uint8) with 1/255 for foreground (white), 0 for background (black)
    returns (top, left, bottom, right) inclusive coordinates of the maximal rectangle
    """
    # ensure boolean
    M = (mask > 0).astype(np.uint8)
    h, w = M.shape

    heights = np.zeros(w, dtype=int)
    best_area, best = 0, (0, 0, 0, 0)  # (top, left, bottom, right)

    for r in range(h):
        # Build histogram of consecutive 1s ending at this row
        heights = (heights + 1) * M[r]  # reset to 0 where M[r] == 0

        # Largest rectangle in histogram via monotonic stack
        stack = []  # pairs: (col_index, start_col)
        c = 0
        while c <= w:
            cur_h = heights[c] if c < w else 0
            start = c
            while stack and stack[-1][0] > cur_h:
                height, start = stack.pop()
                area = height * (c - start)
                if area > best_area:
                    top = r - height + 1
                    left = start
                    bottom = r
                    right = c - 1
                    best_area, best = area, (top, left, bottom, right)
            stack.append((cur_h, start))
            c += 1

    return best
