import os, glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


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

def plot_pr_curve(recalls_by_error,precisions_by_error, f1_by_error, errors):

    plt.figure()
    plt.plot(recalls_by_error, precisions_by_error, marker='o')
    for r, p, e in zip(recalls_by_error, precisions_by_error, errors):
        plt.annotate(str(e), (r, p), textcoords="offset points", xytext=(5,5))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall vs error")
    plt.grid(True)
    plt.show()

    # (optional) Also check which error gives best F1
    best_idx = int(np.argmax(f1_by_error))
    print(f"Best error by F1: {errors[best_idx]}  |  F1={f1_by_error[best_idx]:.4f}  "
        f"(P={precisions_by_error[best_idx]:.4f}, R={recalls_by_error[best_idx]:.4f})")


dir_path = 'datasets\qsd2_w1'
errors = [6] 
use_micro = True 
debug = False

recalls_by_error = []
precisions_by_error = []
f1_by_error = []

for err in errors:

    precision_list = []
    recall_list = []
    f1_list = []

    TP_total = FP_total = FN_total = 0

    for image_path in glob.iglob(os.path.join(dir_path, "*.jpg")):
        base, _ = os.path.splitext(image_path)
        image = cv.imread(image_path)
        A = cv.imread(base + '.png', cv.IMREAD_GRAYSCALE)

        lab = cv.cvtColor(image, cv.COLOR_BGR2Lab)
        h, w = image.shape[:2]

        first_col = lab[:, 0]
        last_col  = lab[:, w - 1]
        first_row = lab[0, :]
        last_row  = lab[h - 1, :]

        borders = np.concatenate((first_col, last_col, first_row, last_row), axis=0)
        avg_color = np.mean(borders, axis=0)
        std_dev_color = np.std(borders, axis=0) * 1.5

        colors = avg_color.tolist()
        th = np.array([avg_color - std_dev_color, avg_color + std_dev_color], dtype=np.float32).transpose([1, 0])

        if th.shape[0] == 3:
            th_ab = th[1:]  
        elif th.shape[0] == 2:
            th_ab = th     
        else:
            raise ValueError("th must have 2 (a,b) or 3 (L,a,b) rows.")

        lower = np.clip(np.round(th_ab[:, 0]), 0, 255).astype(np.uint8)
        upper = np.clip(np.round(th_ab[:, 1]), 0, 255).astype(np.uint8)

        ab = lab[:, :, 1:3]
        ch_ok = (ab >= lower.reshape(1, 1, 2)) & (ab <= upper.reshape(1, 1, 2))
        mask_all_ab = (np.all(ch_ok, axis=2).astype(np.uint8) * 255)
        mask_all_inv = cv.bitwise_not(mask_all_ab)
        
        num, labels, stats, _ = cv.connectedComponentsWithStats(mask_all_inv, connectivity=8)
        if num > 1:
            largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
        else:
            largest_label = 0
            
        mask = np.zeros_like(mask_all_inv)
        mask[labels == largest_label] = 255

        mask = np.apply_along_axis(fill_short_zero_runs, 0, mask)
        output_mask = np.apply_along_axis(fill_short_zero_runs, 1, mask)

        A_bin = (A > 127).astype(np.uint8)
        O_bin = (output_mask > 127).astype(np.uint8)

        image_name = os.path.basename(base)
        folder = "results"

        cv.imwrite(os.path.join(folder, f"{image_name}_img.jpg"), image)
        cv.imwrite(os.path.join(folder, f"{image_name}_output_mask.png"), mask)
        cv.imwrite(os.path.join(folder, f"{image_name}_annotation.png"), A)

        TP = np.sum((A_bin == 1) & (O_bin == 1))
        FP = np.sum((A_bin == 0) & (O_bin == 1))
        FN = np.sum((A_bin == 1) & (O_bin == 0))

        precision_i = TP / (TP + FP + 1e-8) #the 1e-8 is to avoid 0/0 divisions
        recall_i    = TP / (TP + FN + 1e-8)
        f1_i        = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-8)

        print(precision_i)
        if precision_i < 0.5:
            print(f'{image_path} has less than 50 precision: {precision_i}') 
        if recall_i < 0.5:
            print(f'{image_path} has less than 50 recall: {recall_i}')
        if f1_i < 0.5:
            print(f'{image_path} has less than 50 f1: {f1_i}') 
        precision_list.append(precision_i)
        recall_list.append(recall_i)
        f1_list.append(f1_i)

        # --- accumulate (for MICRO) ---
        TP_total += TP
        FP_total += FP
        FN_total += FN

    if use_micro:
        precision = TP_total / (TP_total + FP_total + 1e-8)
        recall    = TP_total / (TP_total + FN_total + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        precision = float(np.mean(precision_list))
        recall    = float(np.mean(recall_list))
        f1        = float(np.mean(f1_list))

    precisions_by_error.append(precision)
    recalls_by_error.append(recall)
    f1_by_error.append(f1)

if debug:
    plot_pr_curve(recalls_by_error,precisions_by_error, f1_by_error, errors)
