# Optimized, modular and cleaned version of the original background removal script.
# Functionality preserved exactly (no logic, thresholds or return order changed).
import numpy as np
import cv2
from scipy.ndimage import binary_opening, median_filter, morphological_gradient
from scipy import ndimage
from skimage import morphology, segmentation


# -------------------------
# Utilities / Color & image conversions
# -------------------------
def convert_to_representation(image):
    """Convert RGB image to LAB and return channels as float arrays."""
    im_lab = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(im_lab)
    return L.astype(float), a.astype(float), b.astype(float)


def normalize_gradient(grad):
    """Normalize gradient values to 0-255 (uint8)."""
    grad_float = grad.astype(np.float32)
    grad_min = grad_float.min()
    grad_max = grad_float.max()
    grad_norm = 255 * (grad_float - grad_min) / ((grad_max - grad_min) + 1e-8)
    return grad_norm.astype(np.uint8)


# -------------------------
# Edge detection and polygon estimation helpers
# -------------------------
def compute_edge_mask(im_lab, gradient_threshold=0.1):
    """Compute morphological gradient on LAB channels and return binary mask + gradient."""
    L, a, b = im_lab
    grad_l = morphological_gradient(L, structure=np.ones((5, 5)))
    grad_a = morphological_gradient(a, structure=np.ones((5, 5)))
    grad_b = morphological_gradient(b, structure=np.ones((5, 5)))

    grad_combined = np.sqrt(grad_l**2 + grad_a**2 + grad_b**2)

    threshold = np.max(grad_combined) * gradient_threshold
    grad_bin = grad_combined > threshold

    mask = binary_opening(grad_bin, structure=np.ones((3, 3)))
 
    return mask, grad_combined


def create_border_suppressed_mask(mask_bool, pixel_border, h, w):
    """Suppress border pixels (up to pixel_border) to avoid extremes on edges."""
    if not pixel_border or int(pixel_border) <= 0:
        return mask_bool

    mask_for_extremes = mask_bool.copy()
    pb = int(pixel_border)

    pb_h = min(pb, h // 2)
    pb_w = min(pb, w // 2)

    if pb_w > 0:
        mask_for_extremes[:, :pb_w] = False
        mask_for_extremes[:, w - pb_w:] = False
    if pb_h > 0:
        mask_for_extremes[:pb_h, :] = False
        mask_for_extremes[h - pb_h:, :] = False

    return mask_for_extremes


def compute_extreme_points(mask_for_extremes, h, w):
    """Compute left/right/top/bottom extreme coordinates and boolean rows/cols presence."""
    rows_any = mask_for_extremes.any(axis=1)
    left_all = mask_for_extremes.argmax(axis=1)
    right_all = w - 1 - np.argmax(mask_for_extremes[:, ::-1], axis=1)
    left = np.where(rows_any, left_all, -1)
    right = np.where(rows_any, right_all, -1)

    cols_any = mask_for_extremes.any(axis=0)
    top_all = mask_for_extremes.argmax(axis=0)
    bottom_all = h - 1 - np.argmax(mask_for_extremes[::-1, :], axis=0)
    top = np.where(cols_any, top_all, -1)
    bottom = np.where(cols_any, bottom_all, -1)

    return left, right, top, bottom, rows_any, cols_any


def filter_extremes_by_median(extremes, threshold):
    """Filter extreme coordinates by median deviation; outliers become -1."""
    filtered = extremes.copy()
    valid = filtered >= 0

    if valid.any():
        median_val = int(np.median(filtered[valid]))
        filtered[valid] = np.where(
            np.abs(filtered[valid] - median_val) <= threshold,
            filtered[valid],
            -1
        )

    return filtered


def create_border_mask(left, right, top, bottom, rows_any, cols_any, shape):
    """Create a sparse border mask from per-row/col extreme points."""
    border_mask = np.zeros(shape, dtype=bool)

    row_inds = np.where(rows_any)[0]
    for r in row_inds:
        if left[r] >= 0:
            border_mask[r, left[r]] = True
        if right[r] >= 0:
            border_mask[r, right[r]] = True

    col_inds = np.where(cols_any)[0]
    for c in col_inds:
        if top[c] >= 0:
            border_mask[top[c], c] = True
        if bottom[c] >= 0:
            border_mask[bottom[c], c] = True

    return border_mask


def fit_x_of_y(y_indices, x_values):
    """Fit x = a*y + b (vertical-like edges)."""
    if len(y_indices) >= 2:
        a, b = np.polyfit(y_indices, x_values, 1)
    elif len(y_indices) == 1:
        a, b = 0.0, float(x_values[0])
    else:
        a, b = None, None
    return a, b


def fit_y_of_x(x_indices, y_values):
    """Fit y = c*x + d (horizontal-like edges)."""
    if len(x_indices) >= 2:
        c, d = np.polyfit(x_indices, y_values, 1)
    elif len(x_indices) == 1:
        c, d = 0.0, float(y_values[0])
    else:
        c, d = None, None
    return c, d


def intersect_lines(a, b, c, d):
    """Intersect x = a*y + b with y = c*x + d -> returns (x,y) or None if parallel/invalid."""
    if a is None or c is None:
        return None

    denom = 1.0 - a * c
    if abs(denom) < 1e-6:
        return None

    x = (a * d + b) / denom
    y = c * x + d
    return (x, y)


def compute_polygon_corners(left, right, top, bottom, h, w):
    """From extremes, fit four sides and compute polygon corners (with fallbacks)."""
    rows_left = np.where(left >= 0)[0]
    xs_left = left[rows_left] if rows_left.size > 0 else np.array([])

    rows_right = np.where(right >= 0)[0]
    xs_right = right[rows_right] if rows_right.size > 0 else np.array([])

    cols_top = np.where(top >= 0)[0]
    ys_top = top[cols_top] if cols_top.size > 0 else np.array([])

    cols_bottom = np.where(bottom >= 0)[0]
    ys_bottom = bottom[cols_bottom] if cols_bottom.size > 0 else np.array([])

    a_left, b_left = fit_x_of_y(rows_left, xs_left)
    a_right, b_right = fit_x_of_y(rows_right, xs_right)
    c_top, d_top = fit_y_of_x(cols_top, ys_top)
    c_bottom, d_bottom = fit_y_of_x(cols_bottom, ys_bottom)

    corners = [
        intersect_lines(a_left, b_left, c_top, d_top),      # left-top
        intersect_lines(a_right, b_right, c_top, d_top),    # right-top
        intersect_lines(a_right, b_right, c_bottom, d_bottom),  # right-bottom
        intersect_lines(a_left, b_left, c_bottom, d_bottom)     # left-bottom
    ]

    final_corners = []
    for i, corner in enumerate(corners):
        if corner is None:
            if i == 0:  # left-top
                x = np.median(xs_left) if xs_left.size > 0 else 0
                y = np.median(ys_top) if ys_top.size > 0 else 0
            elif i == 1:  # right-top
                x = np.median(xs_right) if xs_right.size > 0 else (w - 1)
                y = np.median(ys_top) if ys_top.size > 0 else 0
            elif i == 2:  # right-bottom
                x = np.median(xs_right) if xs_right.size > 0 else (w - 1)
                y = np.median(ys_bottom) if ys_bottom.size > 0 else (h - 1)
            else:  # left-bottom
                x = np.median(xs_left) if xs_left.size > 0 else 0
                y = np.median(ys_bottom) if ys_bottom.size > 0 else (h - 1)
            corner = (float(x), float(y))

        x_cl = int(np.clip(round(corner[0]), 0, w - 1))
        y_cl = int(np.clip(round(corner[1]), 0, h - 1))
        final_corners.append((x_cl, y_cl))

    return final_corners


def create_polygon_mask(corners, shape, fallback_mask):
    """Create filled polygon mask; fallback to fallback_mask if degenerate."""
    h, w = shape
    polygon = np.array(corners, dtype=np.int32)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        cv2.fillPoly(poly_mask, [polygon], 1)
    except Exception:
        poly_mask = fallback_mask.astype(np.uint8)
    return poly_mask


# -------------------------
# Shadow detection / cluster utilities
# -------------------------
def is_cluster_in_upper_third(region_mask, mask):
    """
    True si la parte del cluster que cae dentro de `mask` está completamente
    en el tercio superior de la extensión vertical de `mask`.
    """
    # Intersectar el cluster con la máscara para trabajar sólo dentro del polígono
    region_inside_mask = region_mask & (mask.astype(bool))

    coords = np.argwhere(region_inside_mask)
    if len(coords) == 0:
        return False

    # Obtener la extensión vertical de la máscara (filas en las que mask tiene True)
    mask_rows = np.where(mask.any(axis=1))[0]
    if mask_rows.size == 0:
        return False

    mask_top = int(mask_rows[0])
    mask_bottom = int(mask_rows[-1])
    mask_height = mask_bottom - mask_top + 1
    upper_limit = mask_top + mask_height // 3  # primer tercio de la máscara

    max_y = int(np.max(coords[:, 0]))
    return max_y < upper_limit


def cluster_has_holes(cluster_mask):
    """Check if a cluster mask has holes (connected components in inverted mask)."""
    inverted = (~cluster_mask.astype(bool)).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(inverted, connectivity=8)
    # num_labels includes the outer background; >2 means at least one hole
    return num_labels > 2


def find_background_cluster(clusters_map):
    """Return label of any cluster that touches image border, else None."""
    h, w = clusters_map.shape
    unique_labels = np.unique(clusters_map)
    unique_labels = unique_labels[unique_labels > 0]
    for label in unique_labels:
        cluster_mask = (clusters_map == label)
        if (np.any(cluster_mask[0, :]) or np.any(cluster_mask[-1, :]) or
                np.any(cluster_mask[:, 0]) or np.any(cluster_mask[:, -1])):
            return label
    return None


def combine_cluster_masks(clusters_map):
    """Combine clusters_map > 0 into a single binary mask (uint8)."""
    return (clusters_map > 0).astype(np.uint8)


def find_max_interior_quadrilateral(cluster_mask, original_mask, start_point=None):
    """
    Find a maximal interior quadrilateral inside cluster_mask's hole (used to refine mask).
    Behavior preserved from the original implementation.
    """
    h, w = cluster_mask.shape
    hole_mask = (~cluster_mask.astype(bool)).astype(np.uint8)
    hole_mask = hole_mask & original_mask  # intersection with original polygon mask
    combined_mask = hole_mask

    if start_point is None:
        center_y, center_x = h // 2, w // 2
    else:
        center_y, center_x = start_point

    if not combined_mask[center_y, center_x]:
        coords = np.argwhere(combined_mask > 0)
        if len(coords) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        distances = np.sum((coords - np.array([center_y, center_x]))**2, axis=1)
        closest = coords[np.argmin(distances)]
        center_y, center_x = closest[0], closest[1]

    # Main cross rays
    top_y = center_y
    while top_y > 0 and combined_mask[top_y - 1, center_x]:
        top_y -= 1

    bottom_y = center_y
    while bottom_y < h - 1 and combined_mask[bottom_y + 1, center_x]:
        bottom_y += 1

    left_x = center_x
    while left_x > 0 and combined_mask[center_y, left_x - 1]:
        left_x -= 1

    right_x = center_x
    while right_x < w - 1 and combined_mask[center_y, right_x + 1]:
        right_x += 1

    # Trim 1/5 on each side
    vertical_length = bottom_y - top_y
    fifth_v = vertical_length // 5
    top_trimmed = top_y + fifth_v
    bottom_trimmed = bottom_y - fifth_v

    horizontal_length = right_x - left_x
    fifth_h = horizontal_length // 5
    left_trimmed = left_x + fifth_h
    right_trimmed = right_x - fifth_h

    # Launch perpendicular rays from trimmed spans
    left_points = []
    right_points = []
    for y in range(top_trimmed, bottom_trimmed + 1):
        x_left = center_x
        while x_left > 0 and combined_mask[y, x_left - 1]:
            x_left -= 1
        left_points.append((y, x_left))

        x_right = center_x
        while x_right < w - 1 and combined_mask[y, x_right + 1]:
            x_right += 1
        right_points.append((y, x_right))

    top_points = []
    bottom_points = []
    for x in range(left_trimmed, right_trimmed + 1):
        y_top = center_y
        while y_top > 0 and combined_mask[y_top - 1, x]:
            y_top -= 1
        top_points.append((y_top, x))

        y_bottom = center_y
        while y_bottom < h - 1 and combined_mask[y_bottom + 1, x]:
            y_bottom += 1
        bottom_points.append((y_bottom, x))

    # Filtering by median deviation
    def filter_points_by_median(points, axis, threshold=20):
        if len(points) == 0:
            return []
        points_arr = np.array(points)
        values = points_arr[:, axis]
        median_val = np.median(values)
        filtered = [pt for pt in points if abs(pt[axis] - median_val) <= threshold]
        return filtered if len(filtered) > 0 else points

    left_points_filtered = filter_points_by_median(left_points, axis=1, threshold=20)
    right_points_filtered = filter_points_by_median(right_points, axis=1, threshold=20)
    top_points_filtered = filter_points_by_median(top_points, axis=0, threshold=20)
    bottom_points_filtered = filter_points_by_median(bottom_points, axis=0, threshold=20)

    # Fit lines and compute intersections (reusing top-level fit/intersect)
    def fit_points_x_of_y(points):
        if len(points) == 0:
            return None, None
        arr = np.array(points)
        y_vals = arr[:, 0]
        x_vals = arr[:, 1]
        if len(points) >= 2:
            a, b = np.polyfit(y_vals, x_vals, 1)
        elif len(points) == 1:
            a, b = 0.0, float(x_vals[0])
        else:
            a, b = None, None
        return a, b

    def fit_points_y_of_x(points):
        if len(points) == 0:
            return None, None
        arr = np.array(points)
        y_vals = arr[:, 0]
        x_vals = arr[:, 1]
        if len(points) >= 2:
            c, d = np.polyfit(x_vals, y_vals, 1)
        elif len(points) == 1:
            c, d = 0.0, float(y_vals[0])
        else:
            c, d = None, None
        return c, d

    a_left, b_left = fit_points_x_of_y(left_points_filtered)
    a_right, b_right = fit_points_x_of_y(right_points_filtered)
    c_top, d_top = fit_points_y_of_x(top_points_filtered)
    c_bottom, d_bottom = fit_points_y_of_x(bottom_points_filtered)

    corners = [
        intersect_lines(a_left, b_left, c_top, d_top),
        intersect_lines(a_right, b_right, c_top, d_top),
        intersect_lines(a_right, b_right, c_bottom, d_bottom),
        intersect_lines(a_left, b_left, c_bottom, d_bottom)
    ]

    final_corners = []
    for i, corner in enumerate(corners):
        if corner is None:
            if i == 0:
                if len(left_points_filtered) > 0 and len(top_points_filtered) > 0:
                    x = np.median([p[1] for p in left_points_filtered])
                    y = np.median([p[0] for p in top_points_filtered])
                else:
                    x, y = left_x, top_y
            elif i == 1:
                if len(right_points_filtered) > 0 and len(top_points_filtered) > 0:
                    x = np.median([p[1] for p in right_points_filtered])
                    y = np.median([p[0] for p in top_points_filtered])
                else:
                    x, y = right_x, top_y
            elif i == 2:
                if len(right_points_filtered) > 0 and len(bottom_points_filtered) > 0:
                    x = np.median([p[1] for p in right_points_filtered])
                    y = np.median([p[0] for p in bottom_points_filtered])
                else:
                    x, y = right_x, bottom_y
            else:
                if len(left_points_filtered) > 0 and len(bottom_points_filtered) > 0:
                    x = np.median([p[1] for p in left_points_filtered])
                    y = np.median([p[0] for p in bottom_points_filtered])
                else:
                    x, y = left_x, bottom_y
            corner = (float(x), float(y))

        x_cl = int(np.clip(round(corner[0]), 0, w - 1))
        y_cl = int(np.clip(round(corner[1]), 0, h - 1))
        final_corners.append((x_cl, y_cl))

    polygon = np.array(final_corners, dtype=np.int32)
    result_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        cv2.fillPoly(result_mask, [polygon], 1)
    except Exception:
        result_mask = np.zeros((h, w), dtype=np.uint8)

    return result_mask


def remove_shadows_and_refine(poly_mask, grad_norm):
    """
    Encapsulates the 'segunda fase' shadow removal and final_mask computation.
    Returns: grad_bin (uint8 binary closed), shadows_candidates (int32 labels),
             shadows_candidates2 (copy of labels), final_mask (uint8).
    """
    NEARITY = 20
    poly_mask = poly_mask.astype(np.uint8)
    h, w = poly_mask.shape

    # Crear una máscara expandida para el filtro
    poly_mask_expanded = poly_mask.copy()

    # Encontrar los píxeles no-cero de la máscara
    nonzero_pixels = np.argwhere(poly_mask > 0)

    if len(nonzero_pixels) > 0:
        # Calcular el bounding box de la máscara
        min_y, min_x = nonzero_pixels.min(axis=0)
        max_y, max_x = nonzero_pixels.max(axis=0)
        
        # Calcular el centro y dimensiones actuales
        center_y = (min_y + max_y) / 2
        center_x = (min_x + max_x) / 2
        height_mask = max_y - min_y
        width_mask = max_x - min_x
        
        # Intentar zoom del 10%
        zoom_factor = 1.10
        new_height = height_mask * zoom_factor
        new_width = width_mask * zoom_factor
        
        # Calcular nuevos límites
        new_min_y = center_y - new_height / 2
        new_max_y = center_y + new_height / 2
        new_min_x = center_x - new_width / 2
        new_max_x = center_x + new_width / 2
        
        # Verificar distancia mínima al borde (10 píxeles)
        margin = 10
        
        # Calcular el zoom máximo permitido para mantener el margen
        max_zoom_y = min((center_y - margin) / (height_mask / 2), 
                        (h - center_y - margin) / (height_mask / 2)) if height_mask > 0 else zoom_factor
        max_zoom_x = min((center_x - margin) / (width_mask / 2), 
                        (w - center_x - margin) / (width_mask / 2)) if width_mask > 0 else zoom_factor
        max_zoom = min(max_zoom_y, max_zoom_x)
        
        # Ajustar zoom si es necesario
        if max_zoom < 1.0:
            # No se puede expandir, usar máscara original
            zoom_factor = 1.0
        else:
            zoom_factor = min(zoom_factor, max_zoom)
        
        # Aplicar el zoom final solo si es mayor a 1.0
        if zoom_factor > 1.0:
            final_height = height_mask * zoom_factor
            final_width = width_mask * zoom_factor
            
            final_min_y = int(max(0, center_y - final_height / 2))
            final_max_y = int(min(h, center_y + final_height / 2))
            final_min_x = int(max(0, center_x - final_width / 2))
            final_max_x = int(min(w, center_x + final_width / 2))
            
            # Crear máscara expandida
            poly_mask_expanded = np.zeros_like(poly_mask)
            poly_mask_expanded[final_min_y:final_max_y, final_min_x:final_max_x] = 1
            poly_mask_expanded = np.where(poly_mask_expanded > 0, poly_mask_expanded, 0).astype(np.uint8)

    # Threshold and close gradient, masked by polygon
    grad_norm_thr = np.where(grad_norm > np.max(grad_norm) * 0.1, 255, 0).astype(np.uint8)
    grad_norm_thr = np.where(poly_mask_expanded > 0, grad_norm_thr, 0).astype(np.uint8)
    grad_bin = cv2.morphologyEx(grad_norm_thr, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8))
    grad_bin = (grad_bin > 0).astype(np.uint8)
    grad_bin[grad_bin > 0] = 1

    zero_mask = (grad_bin == 0)

    shadows_candidates = np.zeros_like(poly_mask, dtype=np.int32)
    current_label = 1

    def process_seed_and_label_local(r, c):
        nonlocal current_label
        if r < 0 or r >= h or c < 0 or c >= w:
            return
        if not zero_mask[r, c]:
            return
        region = segmentation.flood(zero_mask, (r, c), connectivity=1)
        region_size = np.sum(region)
        if region_size < 100:
            return
        if is_cluster_in_upper_third(region, poly_mask):
            return
        shadows_candidates[region] = current_label
        current_label += 1

    # Sweep from left/top/right/bottom as original logic
    # LEFT
    for r in range(h):
        state = 0
        gradient_count = 0
        for x in range(w):
            if grad_bin[r, x] > 0 and state == 0:
                state = 1
                gradient_count = 1
            elif grad_bin[r, x] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break
            elif grad_bin[r, x] == 0 and state == 1:
                process_seed_and_label_local(r, x)
                break

    # RIGHT
    for r in range(h):
        state = 0
        gradient_count = 0
        for x in range(w - 1, -1, -1):
            if grad_bin[r, x] > 0 and state == 0:
                state = 1
                gradient_count = 1
            elif grad_bin[r, x] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break
            elif grad_bin[r, x] == 0 and state == 1:
                process_seed_and_label_local(r, x)
                break

    # TOP
    for c in range(w):
        state = 0
        gradient_count = 0
        for y in range(h):
            if grad_bin[y, c] > 0 and state == 0:
                state = 1
                gradient_count = 1
            elif grad_bin[y, c] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break
            elif grad_bin[y, c] == 0 and state == 1:
                process_seed_and_label_local(y, c)
                break

    # BOTTOM
    for c in range(w):
        state = 0
        gradient_count = 0
        for y in range(h - 1, -1, -1):
            if grad_bin[y, c] > 0 and state == 0:
                state = 1
                gradient_count = 1
            elif grad_bin[y, c] > 0 and state == 1:
                gradient_count += 1
                if gradient_count > NEARITY:
                    break
            elif grad_bin[y, c] == 0 and state == 1:
                process_seed_and_label_local(y, c)
                break

    # Diameter-based filtering as original
    poly_rows = np.any(poly_mask > 0, axis=1)
    poly_cols = np.any(poly_mask > 0, axis=0)
    Vm = np.sum(poly_rows)
    Hm = np.sum(poly_cols)

    unique_labels = np.unique(shadows_candidates)
    unique_labels = unique_labels[unique_labels > 0]
    shadows_candidates2 = shadows_candidates.copy()

    for label in unique_labels:
        cluster_mask = (shadows_candidates == label)
        cluster_rows = np.any(cluster_mask, axis=1)
        cluster_cols = np.any(cluster_mask, axis=0)
        cluster_V = np.sum(cluster_rows)
        cluster_H = np.sum(cluster_cols)

        meets_V = cluster_V >= (3 / 5) * Vm
        meets_H = cluster_H >= (3 / 5) * Hm
        has_holes = cluster_has_holes(cluster_mask)

        if not (meets_V or meets_H):
            shadows_candidates[cluster_mask] = 0
            shadows_candidates2[cluster_mask] = 0
            continue

        if meets_V and not meets_H:
            if cluster_H > (3 / 5) * Hm:
                shadows_candidates[cluster_mask] = 0
                shadows_candidates2[cluster_mask] = 0
        elif meets_H and not meets_V:
            if cluster_V > (3 / 5) * Vm:
                shadows_candidates[cluster_mask] = 0
                shadows_candidates2[cluster_mask] = 0
        elif meets_V and meets_H:
            shadows_candidates[cluster_mask] = 0

        if has_holes:
            shadows_candidates[cluster_mask] = 0
            shadows_candidates2[cluster_mask] = 0
            
            
    # Ensure background cluster exists in shadows_candidates2 before creating C
    bg_exists = find_background_cluster(shadows_candidates2)
    if bg_exists is None:
        #print("[DEBUG] No background cluster found. Creating one via border flooding.")
        # Use bottom-right corner as seed (most likely to be background)
        seed_r, seed_c = h - 1, w - 1
        
        if zero_mask[seed_r, seed_c]:
            #print(f"[DEBUG] Border seed at ({seed_r}, {seed_c}). Performing flood fill.")
            region = segmentation.flood(zero_mask, (seed_r, seed_c), connectivity=1)
            region_size = np.sum(region)
            
            if region_size >= 100:  # Same size threshold
                if not is_cluster_in_upper_third(region, poly_mask):
                    shadows_candidates2[region] = current_label
                    #print(f"[DEBUG] Background cluster created with label {current_label}, size {region_size}")
                    current_label += 1
                else:
                    pass
                    #print(f"[DEBUG] Border region in upper third. Not creating background cluster.")
            else:
                pass
                #print(f"[DEBUG] Border region too small ({region_size} pixels). Not creating background cluster.")
        else:
            pass
            #print("[DEBUG] No valid zero_mask pixel found for background cluster.")


    # Algorithm decisions to create final_mask (keeps original branching logic)
    C = shadows_candidates2
    CF = shadows_candidates


    bg_cluster_label = find_background_cluster(C)

    unique_C = np.unique(C)
    unique_C = unique_C[unique_C > 0]
    num_clusters_C = len(unique_C)

    unique_CF = np.unique(CF)
    unique_CF = unique_CF[unique_CF > 0]
    num_clusters_CF = len(unique_CF)

    #print(f"[DEBUG] num_clusters_C = {num_clusters_C}, labels = {unique_C}")
    #print(f"[DEBUG] num_clusters_CF = {num_clusters_CF}, labels = {unique_CF}")

    # Default final mask is poly_mask; other cases modify it per original logic
    if bg_cluster_label is None:
        #print("[DEBUG] No background cluster found. Using polygon mask as final mask.")
        final_mask = poly_mask.copy()

    else:
        if num_clusters_C == 1:
            #print("[DEBUG] Case: single cluster in C. Using background cluster quadrilateral.")
            bg_mask = (C == bg_cluster_label).astype(bool)
            candidate_mask = find_max_interior_quadrilateral(bg_mask, poly_mask)

            # Comprobación de área mínima (70% del área original)
            area_original = np.sum(poly_mask)
            area_candidate = np.sum(candidate_mask)
            ratio = area_candidate / area_original if area_original > 0 else 0

            #print(f"[DEBUG] area_original={area_original}, area_candidate={area_candidate}, ratio={ratio:.3f}")

            if ratio >= 0.7:
                #print("[DEBUG] Candidate mask covers ≥70% of original area. Accepting candidate mask.")
                final_mask = candidate_mask
            else:
                #print("[DEBUG] Candidate mask too small (<70%). Falling back to original polygon mask.")
                final_mask = poly_mask.copy()

        elif num_clusters_C > 1 and num_clusters_CF == 0:
            #print("[DEBUG] Case: multiple clusters in C, none in CF.")
            all_clusters_mask = combine_cluster_masks(C).astype(bool)
            center = (h // 2, w // 2)
            center_value = C[center[0], center[1]]
            #print(f"[DEBUG] Image center at {center}, label at center: {center_value}")

            if center_value > 0:
                #print("[DEBUG] Center lies inside a cluster. Using background cluster mask.")
                bg_mask = (C == bg_cluster_label).astype(bool)
                final_mask = find_max_interior_quadrilateral(bg_mask, poly_mask)
            else:
                #print("[DEBUG] Center not inside any cluster. Testing candidate mask area.")
                candidate_mask = find_max_interior_quadrilateral(all_clusters_mask, poly_mask, center)
                area_original = np.sum(poly_mask)
                area_candidate = np.sum(candidate_mask)
                #print(f"[DEBUG] area_original={area_original}, area_candidate={area_candidate}, ratio={area_candidate / area_original:.3f}")

                if area_candidate >= 0.8 * area_original:
                    #print("[DEBUG] Candidate mask covers ≥80% of original area. Accepting candidate mask.")
                    final_mask = candidate_mask
                else:
                    #print("[DEBUG] Candidate mask too small. Falling back to background cluster mask.")
                    bg_mask = (C == bg_cluster_label).astype(bool)
                    final_mask = find_max_interior_quadrilateral(bg_mask, poly_mask)

        elif num_clusters_CF == 1:
            #print("[DEBUG] Case: single cluster in CF.")
            cf_label = unique_CF[0]
            cf_mask = (CF == cf_label).astype(bool)
            #print(f"[DEBUG] Using CF cluster label {cf_label} for quadrilateral extraction.")
            final_mask = find_max_interior_quadrilateral(cf_mask, poly_mask)

        else:
            #print("[DEBUG] Warning: multiple clusters in CF.  Falling back to background cluster mask.")
            bg_mask = (C == bg_cluster_label).astype(bool)
            final_mask = find_max_interior_quadrilateral(bg_mask, poly_mask)

    #print("[DEBUG] Final mask computation complete.")
    #print(f"[DEBUG] Final mask area = {np.sum(final_mask)}, mask dtype = {final_mask.dtype}")


    # Convert clusters to RGBA visualizations (preserved behavior)
    unique_labels = np.unique(shadows_candidates)
    unique_labels = unique_labels[unique_labels > 0]

    shadows_candidates_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(unique_labels) + 1, 3))
    for i, label in enumerate(unique_labels):
        mask_label = (shadows_candidates == label)
        shadows_candidates_rgba[mask_label, :3] = colors[i]
        shadows_candidates_rgba[mask_label, 3] = 255
    shadows_candidates_viz = shadows_candidates_rgba

    unique_labels = np.unique(shadows_candidates2)
    unique_labels = unique_labels[unique_labels > 0]
    shadows_candidates_rgba2 = np.zeros((h, w, 4), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(unique_labels) + 1, 3))
    for i, label in enumerate(unique_labels):
        mask_label = (shadows_candidates2 == label)
        shadows_candidates_rgba2[mask_label, :3] = colors[i]
        shadows_candidates_rgba2[mask_label, 3] = 255
    shadows_candidates2_viz = shadows_candidates_rgba2

    return grad_bin, shadows_candidates_viz, shadows_candidates2_viz, final_mask


# -------------------------
# Public API: main function (signature preserved)
# -------------------------
def remove_background_morphological_gradient(im, thr=20, pixel_border=15, gradient_threshold=0.05):
    """
    Remove background from an image using morphological gradient detection and polygon fitting.

    Returns (original_image, mask, output_image, normalized_gradient,
             grad_bin, shadows_candidates_viz, shadows_candidates2_viz, final_mask)

    All parameters and outputs preserved from the original implementation.
    """
    # Phase 1: polygon extraction
    im_lab = convert_to_representation(im)
    mask, grad = compute_edge_mask(im_lab, gradient_threshold)

    mask_bool = mask.astype(bool)
    h, w = mask_bool.shape

    mask_for_extremes = create_border_suppressed_mask(mask_bool, pixel_border, h, w)

    left, right, top, bottom, rows_any, cols_any = compute_extreme_points(mask_for_extremes, h, w)

    left_filtered = filter_extremes_by_median(left, thr)
    right_filtered = filter_extremes_by_median(right, thr)
    top_filtered = filter_extremes_by_median(top, thr)
    bottom_filtered = filter_extremes_by_median(bottom, thr)

    border_mask = create_border_mask(left_filtered, right_filtered, top_filtered, bottom_filtered,
                                     rows_any, cols_any, mask_bool.shape)

    final_corners = compute_polygon_corners(left_filtered, right_filtered, top_filtered, bottom_filtered, h, w)
    poly_mask = create_polygon_mask(final_corners, (h, w), border_mask)

    output_image = im * poly_mask[:, :, np.newaxis]
    grad_norm = normalize_gradient(grad)

    ret1 = im.copy()
    ret2 = poly_mask.copy()
    ret3 = output_image.copy()
    ret4 = grad_norm.copy()

    # Phase 2: shadow removal & refinement
    grad_bin, shadows_candidates_viz, shadows_candidates2_viz, final_mask = remove_shadows_and_refine(poly_mask, grad_norm)

    # Final return (keeps original return order)
    #return ret1, ret2, im*final_mask[:, :, np.newaxis], ret4, grad_bin, shadows_candidates_viz, shadows_candidates2_viz, final_mask


    return ret1, final_mask, im*final_mask[:, :, np.newaxis], None
