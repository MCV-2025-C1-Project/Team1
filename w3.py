import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from eval_grads import longest_run_from_runs, all_runs_above, longest_run
import sys
import glob
import os
PATH = "qsd2_w3/"

gt = [1,1,2,1,1,2,2,2,2,1,1,2,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,2]

def center_line_lab(img_lab, axis=1):
    """
    img_lab: HxWx3 array in Lab (L in [0,100], a/b typically [-128,127] or similar)
    axis: 1 -> center row; 0 -> center column
    returns: Nx3 center line (float)
    """
    try:
        H, W, C = img_lab.shape
        if axis == 1:  # row
            r = H // 2
            return img_lab[r, :, :].astype(float)
        elif axis == 0:  # column
            c = W // 2
            return img_lab[:, c, :].astype(float)
    except:
        H, W = img_lab.shape
        if axis == 1:  # row
            r = H // 2
            return img_lab[r, :].astype(float)
        elif axis == 0:  # column
            c = W // 2
            return img_lab[:, c].astype(float)
        
def filter_runs_by_gap_and_length(nums, max_gap=110, min_run_len=10):
    """
    Group nums into runs where consecutive elements differ by <= max_gap.
    Keep only runs whose length >= min_run_len.
    Returns:
      kept_nums: flattened list of numbers from kept runs
      kept_runs_count: number of runs kept
      kept_runs: list of (start_index, end_index, run)
    """
    if not nums:
        return [], 0, []

    kept = []
    kept_runs = []
    start = 0

    for i in range(1, len(nums) + 1):
        # End the current run if end of list OR gap > max_gap
        if i == len(nums) or nums[i] - nums[i - 1] > max_gap:
            run = nums[start:i]
            if len(run) >= min_run_len:
                kept.extend(run)
                kept_runs.append((start, i, run))
            start = i

    kept_runs_count = len(kept_runs)
    return kept, kept_runs_count, kept_runs

def get_n_pictures(image_gray,image_lab):
    
    col = center_line_lab(image_lab, 0)
    row = center_line_lab(image_lab, 1)
    grad_horizontal = np.gradient(row)
    grad_vertical = np.gradient(col)

    grad_L = np.gradient(col[:, 0])
    grad_a = np.gradient(col[:, 1])
    grad_b = np.gradient(col[:, 2])
    grad_vertical = np.sqrt(grad_L**2 + grad_a**2 + grad_b**2)
    
    grad_L = np.gradient(row[:, 0])
    grad_a = np.gradient(row[:, 1])
    grad_b = np.gradient(row[:, 2])
    grad_horizontal = np.sqrt(grad_L**2 + grad_a**2 + grad_b**2)

    big_vertical_grads = []
    for i, grad in enumerate(grad_vertical):
        if np.abs(grad) > 15:
            #print("Higher grad", grad)
            big_vertical_grads.append(i)


    if not big_vertical_grads:
        print('2 Images')
        return 2

    

    window = 5  # smooth over this many samples
    g = np.abs(np.diff(grad_horizontal))
    g = uniform_filter1d(g, size=window)


    first_col = image_lab[:,0]
    last_col = image_lab[:,w-1]
    c = np.concatenate((first_col, last_col))
    first_col_stdev = np.std(c)

    print(first_col_stdev)
    center_right_pixel = image_lab[h//2, 0]

    small_similar_grads = []
    for i,grad in enumerate(g):
        if grad > 5:
            continue

        current_pixel = image_lab[h // 2, i]

        # Compute Euclidean distance in Lab space (just a,b components here)
        diff = np.linalg.norm(current_pixel.astype(float) - center_right_pixel.astype(float))

        if diff > first_col_stdev:
            # print(f"Col {i}: diff={diff:.2f} → between 1× and 2×std")
            # print(i, image_lab[h//2, 0, 1:], image_lab[h//2, i, 1:])
            continue
        else:
            small_similar_grads.append(i)

    kept, n ,_ = filter_runs_by_gap_and_length(small_similar_grads)

    print(_)
    if n == 3: 
        return 2
    else:
        return 1

def first_last_255(x):
    positions = np.where(x == 255)[0]
    if positions.size == 0:
        return np.array([0, 0])  # if no 255 found
    return np.array([positions[0], positions[-1]])

def closest_from_center_255(x: np.ndarray):
    """
    x: 1D array (row) with 255 for foreground and 0 otherwise.
    Returns: np.array([left_idx, right_idx])
      - left_idx:  column index of the 255 closest to center from the LEFT half (or -1 if none)
      - right_idx: column index of the 255 closest to center from the RIGHT half (or -1 if none)

    Notes:
      - For even widths, the center is between the two middle columns.
        We treat the left half as <= floor(center) and the right half as >= ceil(center).
      - For odd widths, the exact center column belongs to BOTH sides; you’ll get that index
        for both left and right if it’s 255.
    """
    W = x.shape[0]
    center = (W - 1) / 2.0
    cL = int(np.floor(center))  # last column on the left half
    cR = int(np.ceil(center))   # first column on the right half

    cols_255 = np.flatnonzero(x == 255)
    if cols_255.size == 0:
        return np.array([-1, -1], dtype=int)

    # Left: the 255 with the largest column <= cL (closest to center from left)
    left_candidates = cols_255[cols_255 <= cL]
    left_idx = left_candidates.max() if left_candidates.size else -1

    # Right: the 255 with the smallest column >= cR (closest to center from right)
    right_candidates = cols_255[cols_255 >= cR]
    right_idx = right_candidates.min() if right_candidates.size else -1

    return np.array([left_idx, right_idx], dtype=int)

def cure_line(original_arr):
    arr = np.trim_zeros(original_arr)
    stdev = np.std(arr)
    median = np.median(arr)

    lista = []
    for value in arr:
        if value > (median + stdev) or value < (median-stdev):
            new_value = median
        else:
            new_value = value
        lista.append(new_value)

    x = range(len(arr)) 

    z = np.polyfit(x, lista, 1)

    p = np.poly1d(z)

    y = p(x)

    left_zeros = np.argmax(original_arr != 0)
    end = left_zeros + len(y)
    x = np.arange(left_zeros, end)

    return x,y

def get_lines(edges, axis):
    vertical_positions = np.apply_along_axis(first_last_255, axis=axis, arr=edges)


    vertical_positions = np.apply_along_axis(first_last_255, axis=axis, arr=edges)
    vertical_positions = np.moveaxis(vertical_positions, axis, -1)
    left = vertical_positions[:, 0] 
    right = vertical_positions[:, 1]

    for line in [left,right]:
        x,y = cure_line(line)

        if axis == 1:
            pts = np.column_stack((y, x)).astype(np.int32).reshape((-1, 1, 2))

            # Draw polyline in red (BGR color: red = (0,0,255))
            cv2.polylines(original_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        else:
            pts = np.column_stack((x, y)).astype(np.int32).reshape((-1, 1, 2))

            # Draw polyline in red (BGR color: red = (0,0,255))
            cv2.polylines(original_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imshow("Red Line", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def keep_first_last_nonzero_spans(a: np.ndarray):
    """
    Keep only the first and last contiguous non-zero chunks in a 1D array.
    Returns:
      kept: array like a, but zeros everywhere except the first and last chunk
      first_span: (start, end) indices for the first run, or (None, None) if none
      last_span:  (start, end) indices for the last run,  or (None, None) if none
    """
    if a.ndim != 1:
        raise ValueError("Input must be a 1D array")

    nz = a != 0
    if not np.any(nz):
        return np.zeros_like(a), (None, None), (None, None)

    nz_i = nz.astype(np.int8)
    starts = np.flatnonzero(np.diff(nz_i) == 1) + 1
    ends   = np.flatnonzero(np.diff(nz_i) == -1) + 1

    if nz[0]:
        starts = np.r_[0, starts]
    if nz[-1]:
        ends = np.r_[ends, a.size]

    first_start, first_end = int(starts[0]), int(ends[0])
    last_start,  last_end  = int(starts[-1]), int(ends[-1])

    kept = np.zeros_like(a)
    kept[first_start:first_end] = a[first_start:first_end]
    if (last_start, last_end) != (first_start, first_end):
        kept[last_start:last_end] = a[last_start:last_end]

    return kept, (first_start, first_end), (last_start, last_end)

def get_lines_n2(edges, axis):

    lines = []
    vertical_positions = np.apply_along_axis(first_last_255, axis=axis, arr=edges)
    vertical_positions = np.apply_along_axis(first_last_255, axis=axis, arr=edges)

    vertical_positions = np.moveaxis(vertical_positions, axis, -1)
    left = vertical_positions[:, 0] 
    right = vertical_positions[:, 1]
    if axis == 0:
        for line in (right,left):
            kept, first_info, last_info = keep_first_last_nonzero_spans(line)
            first_info_right = np.zeros_like(kept)
            first_info_right[first_info[0]:first_info[1]] = kept[first_info[0]:first_info[1]] 
            first_info_left = np.zeros_like(kept)
            first_info_left[last_info[0]:last_info[1]] = kept[last_info[0]:last_info[1]] 
            lines.append(first_info_right)
            lines.append(first_info_left)
        
    if axis == 1:
        lines.append(left)
        lines.append(right)
        center_positions = np.apply_along_axis(closest_from_center_255, axis=axis, arr=edges)
        center_positions_left = center_positions[:,0]
        center_positions_right = center_positions[:,1]
        lines.append(center_positions_left)
        lines.append(center_positions_right)

    for line in lines:

        x,y = cure_line(line)

        if axis == 1:
            pts = np.column_stack((y, x)).astype(np.int32).reshape((-1, 1, 2))

            # Draw polyline in red (BGR color: red = (0,0,255))
            cv2.polylines(original_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        else:
            pts = np.column_stack((x, y)).astype(np.int32).reshape((-1, 1, 2))

            # Draw polyline in red (BGR color: red = (0,0,255))
            cv2.polylines(original_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imshow("Red Line", original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

n_x_img = []
for image_path in glob.iglob(os.path.join(PATH, '*.jpg')):
    original_img = cv2.imread(image_path)

    img = cv2.medianBlur(original_img, 11)  

    if img is None:
        raise FileNotFoundError(image_path)

    h, w = img.shape[:2]

    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)

    # 1) Gradient edges (Sobel on L channel)
    L = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB)[:,:,0]
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(L)
    L = cv2.GaussianBlur(L, (9,9), 0)

    gx = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(L, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag8 = cv2.convertScaleAbs(mag)


    # 2) Binary edges + connect
    _, edges = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    n = get_n_pictures(image_gray,image_lab)
    print(f'Image: {image_path} has {n} pictures')
    n_x_img.append(n)

    if n == 1:
        get_lines(edges,axis=1)
        get_lines(edges,axis=0)
    if n == 2:
        get_lines_n2(edges, axis=1)
        get_lines_n2(edges, axis=0)


for i, (gt_n, p_n) in enumerate(zip(gt,n_x_img)):
    if gt_n == p_n:
        continue
    else:
        print(f'Image {i} has gt {gt_n} but pred {p_n}')

    # cv2.imshow("Edges", edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

