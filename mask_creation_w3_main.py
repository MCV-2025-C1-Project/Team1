import os
import glob
import cv2
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d

# --------------------
# Configuration / Defaults
# --------------------
DEFAULT_PATH = "qsd2_w3/"
GT = [1,1,1,1,1,1,1,1,2,1,1,2,2,1,1,2,1,1,1,1,1,1,2,1,1,2,2,1,2,2]


# --------------------
# Small utilities
# --------------------
def _maybe_show(title: str, img: np.ndarray, show: bool, wait: bool = True) -> None:
    if not show:
        return
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --------------------
# Core helpers (mostly your originals, tightened and de-globalized)
# --------------------


def center_line_lab(img_lab: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    img_lab: HxWx3 array in Lab (L in [0,100], a/b typically [-128,127])
    axis: 1 -> center row; 0 -> center column
    returns: Nx3 center line (float)
    """
    if img_lab.ndim == 3:  # color Lab
        H, W, _ = img_lab.shape
        if axis == 1:
            r = H // 2
            return img_lab[r, :, :].astype(float)
        elif axis == 0:
            c = W // 2
            return img_lab[:, c, :].astype(float)
    else:                  # grayscale fallback
        H, W = img_lab.shape
        if axis == 1:
            r = H // 2
            return img_lab[r, :].astype(float)
        elif axis == 0:
            c = W // 2
            return img_lab[:, c].astype(float)
    raise ValueError("axis must be 0 (column) or 1 (row)")


def filter_runs_by_gap_and_length(nums,
                                  max_gap=15,
                                  min_run_len_borders=10,
                                  min_run_len_inside=170):
    """
    Group nums into runs where consecutive elements differ by <= max_gap.
    Keep border runs if length >= min_run_len_borders.
    Keep inside runs if length >= min_run_len_inside.
    Returns:
      (n_definite_runs, definite_runs)
        where definite_runs = list of tuples (start_index_in_nums, end_index_in_nums, run_values_list)
    """
    if not nums:
        return 0, []

    kept_runs = []
    start = 0
    for i in range(1, len(nums) + 1):
        if i == len(nums) or nums[i] - nums[i - 1] > max_gap:
            run = nums[start:i]
            if len(run) >= min_run_len_borders:
                kept_runs.append((start, i, run))
            start = i

    if not kept_runs:
        return 0, []

    # Always keep first & last kept runs; keep inner ones if long enough
    definite_runs = [kept_runs[0], kept_runs[-1]]
    for section in kept_runs[1:-1]:
        if len(section[2]) > min_run_len_inside:
            definite_runs.append(section)

    n = len(definite_runs)
    print(definite_runs)
    return n, definite_runs

def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    """
    Overlays a binary mask on the image with transparency.

    Args:
        image (np.ndarray): Original BGR image.
        mask (np.ndarray): Binary mask (0/255).
        color (tuple): Color for the overlay (B,G,R).
        alpha (float): Transparency factor (0.0 = invisible, 1.0 = fully opaque).

    Returns:
        np.ndarray: Image with mask overlay.
    """
    # ensure both same size
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # convert to color if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # make colored overlay
    overlay = image.copy()
    overlay[mask > 127] = color  # apply color where mask is white

    # blend images
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended

def get_n_pictures(image_lab: np.ndarray) -> int:
    """
    Heuristic to decide whether the image likely contains 1 or 2 pictures.
    """
    H, W, _ = image_lab.shape

    col = center_line_lab(image_lab, 0)  # center column (H x 3)
    row = center_line_lab(image_lab, 1)  # center row (W x 3)

    # vertical gradient along the center column
    grad_vertical = np.sqrt(
        np.gradient(col[:, 0])**2 +
        np.gradient(col[:, 1])**2 +
        np.gradient(col[:, 2])**2
    )

    # horizontal gradient along the center row
    grad_horizontal = np.sqrt(
        np.gradient(row[:, 0])**2 +
        np.gradient(row[:, 1])**2 +
        np.gradient(row[:, 2])**2
    )

    big_vertical_grads = [i for i, g in enumerate(grad_vertical) if abs(g) > 15]

    if not big_vertical_grads:
        print('2 Images')
        return 2

    # Smooth the derivative of horizontal gradients
    window = 5
    g = np.abs(np.diff(grad_horizontal))
    g = uniform_filter1d(g, size=window)

    # Compare center-left pixel similarity against border variability
    first_col = image_lab[:, 0, :]
    last_col  = image_lab[:, W - 1, :]
    border_stack = np.concatenate((first_col, last_col), axis=0)
    first_col_stdev = np.std(border_stack)

    center_left_pixel = image_lab[H // 2, 0, :]

    small_similar_grads = []
    for i, gi in enumerate(g):
        if gi > 5:
            continue

        current_pixel = image_lab[H // 2, i, :]
        diff = np.linalg.norm(current_pixel.astype(float) - center_left_pixel.astype(float))
        if diff <= first_col_stdev:
            small_similar_grads.append(i)

    n, _ = filter_runs_by_gap_and_length(small_similar_grads)
    return 2 if n == 3 else 1


def draw_hough_lines(edges, orig_img=None, use_probabilistic=True,
                     rho=1.0, theta=np.pi/180.0, threshold=120,
                     minLineLength=80, maxLineGap=10, show=True, save_path=None):
    # Ensure edges are single-channel uint8
    edges_gray = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY) if edges.ndim == 3 else edges.copy()
    edges_gray = (edges_gray > 0).astype(np.uint8) * 255

    canvas = (cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR) if orig_img is not None and len(orig_img.shape) == 2
              else orig_img.copy() if orig_img is not None
              else cv2.cvtColor(edges_gray, cv2.COLOR_GRAY2BGR))

    if use_probabilistic:
        lines = cv2.HoughLinesP(edges_gray, rho, theta, threshold,
                                minLineLength=minLineLength, maxLineGap=maxLineGap)
        if lines is not None:
            for (x1, y1, x2, y2) in lines[:, 0]:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    else:
        lines = cv2.HoughLines(edges_gray, rho, theta, threshold)
        if lines is not None:
            for (r, th) in lines[:, 0]:
                a, b = np.cos(th), np.sin(th)
                x0, y0 = a * r, b * r
                x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
                x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(save_path, canvas)
        print(f"[INFO] Hough lines image saved to: {save_path}")

    if show:
        _maybe_show('Hough Lines', canvas, show=True)

    return canvas


def draw_frame_from_edges(edges,
                          orig_img=None,
                          rho=1,
                          theta=np.pi/180,
                          threshold=120,
                          maxLineGap=15,
                          vertical_ratio=1.0,
                          thickness=2,
                          color=(0, 255, 0)):
    """
    Detect lines from a binary edge map and keep 4 framing lines:
      - leftmost & rightmost among vertical-ish
      - topmost  & bottommost among horizontal-ish
    Then draw the quadrilateral (their intersections) on a canvas.

    Returns:
        canvas, quad_points (4x2 float) or None
    """
    if edges.ndim == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)
    edges = (edges > 0).astype(np.uint8) * 255
    H, W = edges.shape[:2]

    canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if orig_img is None else orig_img.copy()

    minLineLength = int(0.2 * min(H, W))
    linesP = cv2.HoughLinesP(edges, rho, theta, threshold,
                             minLineLength=minLineLength, maxLineGap=maxLineGap)

    if linesP is None or len(linesP) == 0:
        print("[WARN] No lines detected.")
        return canvas, None

    verticals, horizontals = [], []
    for x1, y1, x2, y2 in linesP[:, 0, :]:
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < vertical_ratio * abs(dy):
            avg_x = 0.5 * (x1 + x2)
            verticals.append(((x1, y1, x2, y2), avg_x))
        else:
            avg_y = 0.5 * (y1 + y2)
            horizontals.append(((x1, y1, x2, y2), avg_y))

    if len(verticals) == 0 or len(horizontals) == 0:
        print("[WARN] Not enough vertical/horizontal candidates.")
        return canvas, None

    verticals.sort(key=lambda v: v[1])
    horizontals.sort(key=lambda h: h[1])

    left_seg  = verticals[0][0]
    right_seg = verticals[-1][0]
    top_seg   = horizontals[0][0]
    bottom_seg= horizontals[-1][0]

    def line_from_segment(seg):
        x1, y1, x2, y2 = map(float, seg)
        a = y1 - y2
        b = x2 - x1
        c = x1*y2 - x2*y1
        n = np.hypot(a, b) + 1e-12
        return a / n, b / n, c / n

    L_left   = line_from_segment(left_seg)
    L_right  = line_from_segment(right_seg)
    L_top    = line_from_segment(top_seg)
    L_bottom = line_from_segment(bottom_seg)

    def intersect(L1, L2):
        a1, b1, c1 = L1
        a2, b2, c2 = L2
        d = a1*b2 - a2*b1
        if abs(d) < 1e-9:
            return None
        x = (b1*c2 - b2*c1) / d
        y = (c1*a2 - c2*a1) / d
        return np.array([x, y], dtype=np.float32)

    TL = intersect(L_top, L_left)
    TR = intersect(L_top, L_right)
    BR = intersect(L_bottom, L_right)
    BL = intersect(L_bottom, L_left)

    quad = [TL, TR, BR, BL]
    if any(p is None or not np.isfinite(p).all() for p in quad):
        print("[WARN] Intersections unstable; drawing the 4 selected segments only.")
        for seg, col in zip([left_seg, right_seg, top_seg, bottom_seg],
                            [(0,255,0), (0,255,0), (255,0,0), (255,0,0)]):
            x1,y1,x2,y2 = map(int, seg)
            cv2.line(canvas, (x1,y1), (x2,y2), col, thickness, cv2.LINE_AA)
        return canvas, None

    quad = np.stack(quad, axis=0).astype(np.float32)

    # Draw the 4 infinite lines for visuals
    def draw_infinite(L, color_):
        a,b,c = L
        pts = []
        if abs(a) > 1e-9:
            x = -c / a
            if 0 <= x <= W-1: pts.append((int(x), 0))
            x = -(c + b*(H-1)) / a
            if 0 <= x <= W-1: pts.append((int(x), H-1))
        if abs(b) > 1e-9:
            y = -c / b
            if 0 <= y <= H-1: pts.append((0, int(y)))
            y = -(c + a*(W-1)) / b
            if 0 <= y <= H-1: pts.append((W-1, int(y)))
        if len(pts) >= 2:
            pts = np.array(pts, dtype=np.int32)
            best_i, best_j, best_d = 0, 1, cv2.norm(pts[0]-pts[1])
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    dd = cv2.norm(pts[i]-pts[j])
                    if dd > best_d:
                        best_d, best_i, best_j = dd, i, j
            p1, p2 = tuple(pts[best_i]), tuple(pts[best_j])
            cv2.line(canvas, p1, p2, color_, thickness, cv2.LINE_AA)

    for L, col in zip([L_left, L_right, L_top, L_bottom],
                      [(0,255,0), (0,255,0), (255,0,0), (255,0,0)]):
        draw_infinite(L, col)

    quad_i = quad.astype(np.int32).reshape(-1,1,2)
    cv2.polylines(canvas, [quad_i], isClosed=True, color=(0,255,255), thickness=2, lineType=cv2.LINE_AA)

    return canvas, quad


def mask_to_boundary(mask, ksize=5, mode="gradient"):
    mask = (mask > 0).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    if mode == "gradient":
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)
    elif mode == "erode_diff":
        er = cv2.erode(mask, k, iterations=1)
        edges = cv2.absdiff(mask, er)
    else:
        raise ValueError("mode must be 'gradient' or 'erode_diff'")
    edges = cv2.medianBlur(edges, 3)
    return edges


def fill_short_zero_runs(arr, max_zeros=5, allow_cross_mid=True, mid_index=None):
    x = np.asarray(arr, dtype=float).copy()
    n = x.size
    if n == 0:
        return x
    mid = n // 2 if mid_index is None else int(mid_index)
    i = 0
    while i < n:
        if x[i] != 0:
            i += 1
            continue
        start = i
        while i < n and x[i] == 0:
            i += 1
        end = i
        L = end - start
        interior_ok = (start > 0 and end < n and x[start-1] != 0 and x[end] != 0)
        short_ok = (L <= max_zeros)
        crosses_mid = (start < mid < end)
        cross_ok = (allow_cross_mid or not crosses_mid)
        if interior_ok and short_ok and cross_ok:
            left, right = x[start-1], x[end]
            x[start:end] = np.linspace(left, right, L + 2)[1:-1]
    return x


def quads_to_mask(quads, shape, offsets=None, value=255):
    H, W = shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    if quads is None:
        return mask
    if not isinstance(quads, (list, tuple)):
        quads = [quads]
    if offsets is None:
        offsets = [(0, 0)] * len(quads)

    for q, (ox, oy) in zip(quads, offsets):
        if q is None:
            continue
        q = np.asarray(q, dtype=np.float32) + np.array([ox, oy], dtype=np.float32)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        cv2.fillPoly(mask, [q.astype(np.int32).reshape(-1, 1, 2)], value)
    return mask


def get_rid_of_shadows(gray_img: np.ndarray, axis: int, original_img: np.ndarray, show: bool = False):
    """
    gray_img: single-channel image (uint8) where non-interest areas are mid-gray.
    axis: axis along which to compute gradient.
    """
    H, W = gray_img.shape[:2]
    grad = np.gradient(gray_img.astype(float), axis=axis)
    mask_zero = np.abs(grad) <= 50

    image_filtered = gray_img.copy()
    image_filtered[mask_zero] = 0

    filled0 = np.apply_along_axis(fill_short_zero_runs, axis=0, arr=image_filtered, max_zeros=1200)
    filled1 = np.apply_along_axis(fill_short_zero_runs, axis=1, arr=filled0, max_zeros=120000)
    edges = mask_to_boundary(filled1)
    gray_blur = cv2.GaussianBlur(edges, (3,3), 0).astype(np.uint8)

    _maybe_show("Blurred", gray_blur, show)

    # draw approximate frame
    overlay, quad = draw_frame_from_edges(gray_blur, orig_img=original_img)
    _maybe_show("Drawn", overlay, show)
    return quad


def process_shadows(mask: np.ndarray, axis: int, original_img: np.ndarray, show: bool = False):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped_img  = original_img[y_min:y_max+1, x_min:x_max+1]
    cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]

    cropped_visible = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)

    full_sized_crop = np.zeros_like(original_img)
    full_sized_crop[y_min:y_max+1, x_min:x_max+1] = cropped_visible
    full_gray = cv2.cvtColor(full_sized_crop, cv2.COLOR_BGR2GRAY)
    full_gray[full_gray == 0] = 255 // 2
    full_gray = cv2.equalizeHist(full_gray)

    _maybe_show("Gray", full_gray, show)

    quad = get_rid_of_shadows(full_gray, axis, original_img, show=show)
    return quad


# --------------------
# Main pipeline
# --------------------
def process_single_image(image_path: str,
                         out_dir: str,
                         evaluate: bool = False,
                         show: bool = False):
    """
    Processes a single image; returns (n_pictures, mask, optional_metrics_dict)
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{filename}.png")

    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(image_path)

    img = cv2.medianBlur(original_img, 11)
    H, W = img.shape[:2]

    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 1) Gradient edges (Sobel/Scharr on L channel)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
    L = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(L)
    L = cv2.GaussianBlur(L, (9, 9), 0)

    gx = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(L, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    mag8 = cv2.convertScaleAbs(mag)

    # 2) Binary edges + connect
    _, edges = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    _maybe_show("Edges", edges, show)

    n = get_n_pictures(image_lab)
    print(n)

    mask = np.zeros((H, W), dtype=np.uint8)
    quads = []
    if n == 1:
        edges_f = np.apply_along_axis(fill_short_zero_runs, axis=1, arr=edges)
        edges_f = np.apply_along_axis(fill_short_zero_runs, axis=0, arr=edges_f)
        edges_b = mask_to_boundary(edges_f)

        overlay, quad = draw_frame_from_edges(edges_b, orig_img=original_img)
        _maybe_show("1 Image (initial quad)", overlay, show)
        mask = quads_to_mask(quad, (H, W))
        cv2.imwrite(out_path, mask)

        new_quad = process_shadows(mask, axis=0, original_img=original_img, show=show)
        mask = quads_to_mask(new_quad, (H, W))
        cv2.imwrite(out_path, mask)

        # overlayed = overlay_mask_on_image(original_img, mask, color=(0, 0, 255), alpha=0.4)
        # cv2.imwrite(out_path.replace('.png', '_overlay.png'), overlayed)
        quads.append(new_quad)
    elif n == 2:
        # Select two largest components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(1.5 * H // 20)))
        edges_d = cv2.dilate(edges, kernel, iterations=1)
        edges_d = (edges_d > 0).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges_d, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_ids = np.argsort(areas)[-2:] + 1
        print("Largest component labels:", largest_ids)

        # Determine left/right by centroid x
        def bbox_center_x(stat_row):
            x, y, w, h, _ = stat_row
            return x + w / 2

        left_id, right_id = largest_ids
        if bbox_center_x(stats[left_id]) > bbox_center_x(stats[right_id]):
            left_id, right_id = right_id, left_id

        x1, y1, w1, h1, _ = stats[left_id]
        x2, y2, w2, h2, _ = stats[right_id]

        crop_left_mask  = edges[y1:y1+h1, x1:x1+w1]
        crop_right_mask = edges[y2:y2+h2, x2:x2+w2]
        crop_left_img   = original_img[y1:y1+h1, x1:x1+w1]
        crop_right_img  = original_img[y2:y2+h2, x2:x2+w2]

        crop_left_mask  = mask_to_boundary(crop_left_mask)
        crop_right_mask = mask_to_boundary(crop_right_mask)

        overlay_left,  quad_left  = draw_frame_from_edges(crop_left_mask,  orig_img=crop_left_img)
        overlay_right, quad_right = draw_frame_from_edges(crop_right_mask, orig_img=crop_right_img)
        _maybe_show("Left overlay", overlay_left, show)
        _maybe_show("Right overlay", overlay_right, show)

        mask1 = quads_to_mask(quad_left,  (H, W), offsets=[(x1, y1)])
        mask2 = quads_to_mask(quad_right, (H, W), offsets=[(x2, y2)])

        quad1 = process_shadows(mask1, axis=0, original_img=original_img, show=show)
        mask1 = quads_to_mask(quad1, (H, W))

        quad2 = process_shadows(mask2, axis=0, original_img=original_img, show=show)
        mask2 = quads_to_mask(quad2, (H, W))

        print(quad1)
        print(quad2)

        mask = quads_to_mask([quad1, quad2], (H, W))
        cv2.imwrite(out_path, mask)
        # overlayed = overlay_mask_on_image(original_img, mask, color=(0, 0, 255), alpha=0.4)
        # cv2.imwrite(out_path.replace('.png', '_overlay.png'), overlayed)
        quads.append(quad1)
        quads.append(quad2)
    # Evaluation block (optional)
    metrics = None
    if evaluate:
        base, _ = os.path.splitext(image_path)
        A = cv2.imread(base + '.png', cv2.IMREAD_GRAYSCALE)
        if A is not None:
            A_bin = (A > 127).astype(np.uint8)
            O_bin = (mask > 127).astype(np.uint8)
            TP = int(np.sum((A_bin == 1) & (O_bin == 1)))
            FP = int(np.sum((A_bin == 0) & (O_bin == 1)))
            FN = int(np.sum((A_bin == 1) & (O_bin == 0)))
            precision_i = TP / (TP + FP + 1e-8)
            recall_i    = TP / (TP + FN + 1e-8)
            f1_i        = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-8)
            metrics = dict(TP=TP, FP=FP, FN=FN,
                           precision=precision_i, recall=recall_i, f1=f1_i)
    return n, mask, metrics, quads


def process_dataset(path: str = DEFAULT_PATH,
                    out_dir: str = "./results",
                    evaluate: bool = False,
                    show: bool = False,
                    gt_list = None,
                    use_micro: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    n_x_img = []
    TP_total = FP_total = FN_total = 0
    precision_list, recall_list, f1_list = [], [], []
    masks = []
    quads = []

    for image_path in glob.iglob(os.path.join(path, '*.jpg')):
        n, mask, metrics, quad = process_single_image(image_path, out_dir, evaluate=evaluate, show=show)
        masks.append(mask)
        quads.append(quad)
        print(f'Image: {image_path} has {n} pictures')
        n_x_img.append(n)

        if evaluate and metrics is not None:
            precision_list.append(metrics["precision"])
            recall_list.append(metrics["recall"])
            f1_list.append(metrics["f1"])
            TP_total += metrics["TP"]
            FP_total += metrics["FP"]
            FN_total += metrics["FN"]

        if use_micro and evaluate:
            precision = TP_total / (TP_total + FP_total + 1e-8)
            recall    = TP_total / (TP_total + FN_total + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            print(f'[MICRO] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        elif evaluate:
            precision = float(np.mean(precision_list)) if precision_list else 0.0
            recall    = float(np.mean(recall_list))    if recall_list else 0.0
            f1        = float(np.mean(f1_list))        if f1_list else 0.0
            print(f'[MACRO] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

    # Optional GT consistency check at end
    if gt_list is not None:
        for i, (n_gt, n_pred) in enumerate(zip(gt_list, n_x_img)):
            if n_gt != n_pred:
                print(f'Image {i} has gt {n_gt} but labeled as {n_pred}')

    return masks, quads


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha=0.5) -> np.ndarray:
    """
    Overlays a binary mask on the image with transparency.

    Args:
        image (np.ndarray): Original BGR image.
        mask (np.ndarray): Binary mask (0/255).
        color (tuple): Color for the overlay (B,G,R).
        alpha (float): Transparency factor (0.0 = invisible, 1.0 = fully opaque).

    Returns:
        np.ndarray: Image with mask overlay.
    """
    # ensure both same size
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # convert to color if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # make colored overlay
    overlay = image.copy()
    overlay[mask > 127] = color  # apply color where mask is white

    # blend images
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended


def main():
    # lightweight arg parsing without argparse (keep dependencies minimal)
    path = DEFAULT_PATH
    evaluate = False
    show = False
    use_micro = True

    for arg in sys.argv[1:]:
        if arg.startswith("--path="):
            path = arg.split("=", 1)[1]
        elif arg == "--eval":
            evaluate = True
        elif arg == "--show":
            show = True
        elif arg == "--macro":
            use_micro = False

    masks, quads = process_dataset(path=path,
                    out_dir="./results",
                    evaluate=evaluate,
                    show=show,
                    gt_list=GT,
                    use_micro=use_micro)


if __name__ == "__main__":
    main()
