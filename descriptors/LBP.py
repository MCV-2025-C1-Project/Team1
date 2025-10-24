import cv2
import numpy as np
from math import ceil

def divide_image(image, blocks: int):
    num_dims = image.ndim
    if num_dims == 2:
        H, W = image.shape
        image = image.reshape([H, W, 1])
    H, W, C = image.shape
    
    height_padding = blocks - (H % blocks)
    width_padding = blocks - (W % blocks)

    new_image = cv2.copyMakeBorder(image, 0, height_padding, 0, width_padding, cv2.BORDER_REFLECT)
    if num_dims == 2:
        H, W, = new_image.shape
        new_image = new_image.reshape([H, W, 1])
    H, W, C = new_image.shape

    new_image = np.reshape(new_image, [blocks, H // blocks, blocks, W // blocks, C]).transpose([0, 2, 1, 3, 4]).reshape(-1, H // blocks, W // blocks, C)
    if num_dims == 2:
        new_image = new_image.reshape([-1, H // blocks, W // blocks])
    return new_image


def divide_image_no_pad_balanced(image, blocks: int):
    H, W = image.shape[:2]
    y_parts = np.array_split(np.arange(H), blocks)
    x_parts = np.array_split(np.arange(W), blocks)

    tiles, coords = [], []
    for ys in y_parts:
        for xs in x_parts:
            y0, y1 = ys[0], ys[-1] + 1
            x0, x1 = xs[0], xs[-1] + 1
            tiles.append(image[y0:y1, x0:x1])
            coords.append((y0, y1, x0, x1))
    return tiles, coords


def divide_image_mirror_pad(image, blocks: int):
    """
    Mirror-pad the image on each side so (H % blocks == 0) and (W % blocks == 0),
    then split into blocks×blocks tiles and return (tiles, coords).

    tiles:  list of np.ndarrays (each tile)
    coords: list of (y0, y1, x0, x1) in the *padded* image coordinates
    """
    H, W = image.shape[:2]

    # how much we need to add to make divisible by `blocks`
    pad_h = (blocks - (H % blocks)) % blocks
    pad_w = (blocks - (W % blocks)) % blocks

    # split padding across both sides
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left

    # mirror pad on each side (works for 2D or 3D images)
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_REFLECT)

    H2, W2 = image.shape[:2]

    # now H2 and W2 are divisible by blocks → equal-sized tiles
    y_parts = np.array_split(np.arange(H2), blocks)
    x_parts = np.array_split(np.arange(W2), blocks)

    tiles, coords = [], []
    for ys in y_parts:
        for xs in x_parts:
            y0, y1 = ys[0], ys[-1] + 1
            x0, x1 = xs[0], xs[-1] + 1
            tiles.append(image[y0:y1, x0:x1])
            coords.append((y0, y1, x0, x1))

    return tiles, coords




def get_LBP_hist_old(image, bins: int, blocks: int):
    """
    Split image into blocks×blocks tiles (balanced, no padding).
    For each tile and each channel:
      - compute LBP (P=8, R=1) with neighbor order: W, SW, S, SE, E, NE, N, NW
      - convert 8-bit pattern to decimal in [0,255]
      - build L1-normalized histogram with `bins` bins over [0,256)
    Concatenate per-tile histograms and per-channel histograms to form a single descriptor.
    Returns: 1D numpy array.
    """
    # Ensure 3D
    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape

    tiles, _ = divide_image_no_pad_balanced(image, blocks)

    # neighbor order starting at LEFT and going clockwise:
    kernel = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]

    all_hists = []
    for c in range(C):
        for tile in tiles:
            t = tile[..., c]
            h, w = t.shape

            # If tile too small for 3x3 neighborhood, append zero hist
            if h < 3 or w < 3:
                hist = np.zeros(bins, dtype=np.float32)
                all_hists.append(hist)
                continue

            codes = []
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    code = 0
                    for (di, dj) in kernel:
                        code = (code << 1) | (1 if t[i + di, j + dj] >= t[i, j] else 0)
                    codes.append(code)

            hist, _ = np.histogram(codes, bins=bins, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-12)  # L1 normalize
            all_hists.append(hist)

    # Concatenate everything into a single vector
    return np.concatenate(all_hists, axis=0)


import numpy as np

def get_LBP_hist(image, bins: int, blocks: int):
    """
    Same API, much faster:
    - vectorized LBP (P=8, R=1) over the full image/channel at once
    - per-tile histograms taken from the precomputed code map
    - L1-normalized per tile, concatenated over tiles and channels
    """
    # --- ensure 3D ---
    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape
    img = image.astype(np.uint8, copy=False)

    # --- helper: coords only (avoid materializing tiles) ---
    # balanced split indices as in divide_image_no_pad_balanced
    def tile_coords(H, W, blocks):
        y_parts = np.array_split(np.arange(H), blocks)
        x_parts = np.array_split(np.arange(W), blocks)
        coords = []
        for ys in y_parts:
            for xs in x_parts:
                y0, y1 = ys[0], ys[-1] + 1
                x0, x1 = xs[0], xs[-1] + 1
                coords.append((y0, y1, x0, x1))
        return coords

    coords = tile_coords(H, W, blocks)

    # --- precompute LBP code map per channel (vectorized) ---
    # For R=1, integer neighbors: W, SW, S, SE, E, NE, N, NW in that order.
    # We compute codes only where 3x3 exists: interior 1..H-2, 1..W-2
    all_hists = []
    for c in range(C):
        ch = img[..., c]

        # center region
        center = ch[1:H-1, 1:W-1]

        # neighbors via slicing (no copies)
        nW  = ch[1:H-1, 0:W-2]     # (y, x-1)
        nSW = ch[2:H,   0:W-2]     # (y+1, x-1)
        nS  = ch[2:H,   1:W-1]     # (y+1, x)
        nSE = ch[2:H,   2:W]       # (y+1, x+1)
        nE  = ch[1:H-1, 2:W]       # (y, x+1)
        nNE = ch[0:H-2, 2:W]       # (y-1, x+1)
        nN  = ch[0:H-2, 1:W-1]     # (y-1, x)
        nNW = ch[0:H-2, 0:W-2]     # (y-1, x-1)

        # comparisons (uint8, so >= is fine)
        b7 = (nW  >= center).astype(np.uint8)  # MSB first to match your order
        b6 = (nSW >= center).astype(np.uint8)
        b5 = (nS  >= center).astype(np.uint8)
        b4 = (nSE >= center).astype(np.uint8)
        b3 = (nE  >= center).astype(np.uint8)
        b2 = (nNE >= center).astype(np.uint8)
        b1 = (nN  >= center).astype(np.uint8)
        b0 = (nNW >= center).astype(np.uint8)  # LSB

        # pack bits into 0..255
        codes = (
            (b7 << 7) | (b6 << 6) | (b5 << 5) | (b4 << 4) |
            (b3 << 3) | (b2 << 2) | (b1 << 1) | b0
        )  # shape: (H-2, W-2), corresponds to centers (1..H-2, 1..W-2)

        # --- per-tile histograms by slicing the code map ---
        for (y0, y1, x0, x1) in coords:
            # original scalar code loop skipped the tile border (i=1..h-2, j=1..w-2)
            # equivalent here: shift coords into codes map and shrink by 1 pixel on each side
            ys0 = max(y0+1, 1)
            ys1 = min(y1-1, H-1)
            xs0 = max(x0+1, 1)
            xs1 = min(x1-1, W-1)

            # If the effective region is too small (<1×<1), return zeros (as before)
            if ys1 <= ys0 or xs1 <= xs0:
                hist = np.zeros(bins, dtype=np.float32)
                all_hists.append(hist)
                continue

            tile_codes = codes[ys0-1:ys1-1, xs0-1:xs1-1].ravel()  # align to codes map

            if bins == 256:
                # exact 256-bin histogram is faster with bincount
                hist = np.bincount(tile_codes, minlength=256).astype(np.float32)
                if bins != 256:  # (kept for safety, but condition is False)
                    hist = hist[:bins]
            else:
                # quantize 0..255 into `bins` equal bins, then bincount
                # bin_idx = floor(code * bins / 256)
                bin_idx = (tile_codes.astype(np.uint16) * bins) >> 8  # fast integer scale
                # guard: codes==255 → bin_idx==bins-1 (OK)
                hist = np.bincount(bin_idx, minlength=bins).astype(np.float32)

            # L1 normalize per tile
            s = hist.sum()
            if s > 0:
                hist /= (s + 1e-12)
            all_hists.append(hist)

    return np.concatenate(all_hists, axis=0)


def get_Multiscale_LBP_hist_old(image, bins: int, blocks: int, scales=None):
    """
    Multiscale LBP with bilinear interpolation, concatenated output.

    - Split image into blocks×blocks tiles (balanced, no global padding).
    - For each tile and each channel:
        * for each (P,R) in `scales`, compute LBP codes with neighbor order
          starting at LEFT (W) and going CLOCKWISE
        * build an L1-normalized histogram with `bins` bins over [0, 2**P)
    - Concatenate per-scale histograms, then across tiles and channels,
      returning a single 1D descriptor.

    Returns: 1D numpy array.
    """
    print(scales)
    if scales is None:
        scales = [(8, 1.0)]  # default classic LBP

    # Ensure 3D
    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape

    tiles, _ = divide_image_no_pad_balanced(image, blocks)

    def bilinear(pad_img, y, x):
        """Bilinear sampling at float (y,x) on 2D array pad_img."""
        y0 = int(np.floor(y)); x0 = int(np.floor(x))
        y1 = y0 + 1;           x1 = x0 + 1
        wy = y - y0;           wx = x - x0
        v00 = pad_img[y0, x0]; v01 = pad_img[y0, x1]
        v10 = pad_img[y1, x0]; v11 = pad_img[y1, x1]
        return (1-wy) * ((1-wx) * v00 + wx * v01) + wy * ((1-wx) * v10 + wx * v11)

    all_hists = []

    for c in range(C):
        for tile in tiles:
            t = tile[..., c].astype(np.float32)
            h, w = t.shape

            maxR = int(np.ceil(max(r for (_P, r) in scales)))
            pad = maxR + 2
            p = np.pad(t, pad, mode='reflect')

            tile_parts = []
            for P, R in scales:
                thetas = np.pi + (-2.0 * np.pi) * (np.arange(P, dtype=np.float32) / P)
                ox = R * np.cos(thetas)
                oy = R * np.sin(thetas)

                codes = []
                for i in range(h):
                    for j in range(w):
                        ci, cj = i + pad, j + pad
                        center = p[ci, cj]
                        code = 0
                        for k in range(P):
                            yk = ci + oy[k]
                            xk = cj + ox[k]
                            nb = bilinear(p, yk, xk)
                            code = (code << 1) | (1 if nb >= center else 0)
                        codes.append(code)

                hist, _ = np.histogram(codes, bins=bins, range=(0, 2 ** P))
                hist = hist.astype(np.float32)
                hist /= (hist.sum() + 1e-12)
                tile_parts.append(hist)

            all_hists.append(np.concatenate(tile_parts, axis=0))

    return np.concatenate(all_hists, axis=0)

import numpy as np

def get_Multiscale_LBP_hist(image, bins: int, blocks: int, scales=None, sampling: str = 'nearest'):
    """
    Multiscale LBP with selectable sampling ('bilinear' | 'nearest'), concatenated output.
    """
    if scales is None:
        scales = [(8, 1.0)]  # default classic LBP

    # Ensure 3D
    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape

    tiles, _ = divide_image_no_pad_balanced(image, blocks)

    # ---- Precompute across scales ----
    maxR = int(np.ceil(max(float(r) for (_P, r) in scales)))
    pad = maxR + 2

    scale_data = []
    for P, R in scales:
        P = int(P); R = float(R)
        thetas = np.pi + (-2.0 * np.pi) * (np.arange(P, dtype=np.float32) / P)  # start LEFT, clockwise
        ox = (R * np.cos(thetas)).astype(np.float32)  # (P,)
        oy = (R * np.sin(thetas)).astype(np.float32)  # (P,)
        weights = (1 << np.arange(P-1, -1, -1, dtype=np.int32)).astype(np.int32)  # MSB->LSB
        code_max = 1 << P
        scale_data.append((P, R, ox, oy, weights, code_max))

    # ---- Samplers ----
    def bilinear_sample(pad_img, y, x):
        y0 = np.floor(y).astype(np.int32); x0 = np.floor(x).astype(np.int32)
        y1 = y0 + 1;                        x1 = x0 + 1
        wy = (y - y0).astype(np.float32);   wx = (x - x0).astype(np.float32)
        v00 = pad_img[y0, x0]; v01 = pad_img[y0, x1]
        v10 = pad_img[y1, x0]; v11 = pad_img[y1, x1]
        return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)

    def nearest_sample(pad_img, y, x):
        yi = np.rint(y).astype(np.int32)
        xi = np.rint(x).astype(np.int32)
        return pad_img[yi, xi]


    sample_fn = bilinear_sample if sampling == 'bilinear' else nearest_sample

    all_hists = []

    for c in range(C):
        for tile in tiles:
            t = tile[..., c].astype(np.float32)
            h, w = t.shape

            p = np.pad(t, pad, mode='reflect')

            yy, xx = np.mgrid[0:h, 0:w]
            ci = (yy + pad).astype(np.float32)  # (h,w)
            cj = (xx + pad).astype(np.float32)  # (h,w)
            center = p[ci.astype(np.int32), cj.astype(np.int32)]

            tile_parts = []
            for (P, R, ox, oy, weights, code_max) in scale_data:
                # neighbor coords (P,h,w)
                yk = ci[None, :, :] + oy[:, None, None]
                xk = cj[None, :, :] + ox[:, None, None]

                nb = sample_fn(p, yk, xk)  # (P,h,w)

                bits = (nb >= center[None, :, :]).astype(np.int32)  # (P,h,w)
                codes = (bits * weights[:, None, None]).sum(axis=0).ravel().astype(np.int32)

                # histogram
                if bins == code_max:
                    hist = np.bincount(codes, minlength=code_max).astype(np.float32)[:bins]
                else:
                    bin_idx = (codes.astype(np.int64) * bins) // code_max
                    bin_idx = np.minimum(bin_idx, bins - 1).astype(np.int32)
                    hist = np.bincount(bin_idx, minlength=bins).astype(np.float32)

                s = hist.sum()
                if s > 0:
                    hist /= (s + 1e-12)

                tile_parts.append(hist)

            all_hists.append(np.concatenate(tile_parts, axis=0))

    return np.concatenate(all_hists, axis=0)


def get_OCLBP_hists_old(image,
                    bins,
                    blocks,
                    P,
                    R,
                    use_uniform_u2=True):
    """
    Over-complete LBP (OCLBP) with bilinear interpolation and overlapping blocks.

    - configs: list of tuples (block_h, block_w, vert_step, hor_step, P, R).
      If None, defaults to the three example configurations from the OCLBP paper,
      implemented with half-overlap (step = block_size // 2):
        [(10,10,5,5,8,1), (14,14,7,7,8,2), (18,18,9,9,8,3)]
    - use_uniform_u2: if True use uniform-u2 mapping -> bins = P + 2 per scale.
                      (This is the common OCLBP choice.)
                      else: bins

    Returns: list of numpy arrays [feat_ch0, feat_ch1, ...] (one 1D array per channel).
    """
    
    bh_grid = max(1, ceil(H / blocks))
    bw_grid = max(1, ceil(W / blocks))
    step_h = bh_grid*0.5
    step_w = bw_grid*0.5
     
    configs = [(bh_grid, bw_grid, step_h, step_w, P, R)]
            

    # ensure 3D
    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape

    # bilinear sampler (same as original)
    def bilinear(pad_img, y, x):
        y0 = int(np.floor(y)); x0 = int(np.floor(x))
        y1 = y0 + 1;           x1 = x0 + 1
        wy = y - y0;           wx = x - x0
        v00 = pad_img[y0, x0]; v01 = pad_img[y0, x1]
        v10 = pad_img[y1, x0]; v11 = pad_img[y1, x1]
        return (1-wy) * ((1-wx) * v00 + wx * v01) + wy * ((1-wx) * v10 + wx * v11)

    # uniform-u2 mapping function for a given P
    # returns an array map[code] -> mapped_index (0..P or P+1)
    def uniform_u2_map(P):
        map_arr = np.zeros(2 ** P, dtype=np.int32)
        for code in range(2 ** P):
            bits = [(code >> (P - 1 - b)) & 1 for b in range(P)]  # MSB-first same as LBP code build
            transitions = sum(bits[i] != bits[(i + 1) % P] for i in range(P))
            if transitions <= 2:
                # uniform -> map to number of ones (0..P)
                map_arr[code] = sum(bits)
            else:
                # non-uniform -> map to last bin (P+1)
                map_arr[code] = P + 1
        return map_arr  # length 2**P, values in 0..P+1

    per_channel_features = []

    # For each channel, compute descriptor
    for c in range(C):
        ch = image[..., c].astype(np.float32)

        channel_parts = []  # will hold arrays of histograms (per config, per tile)

        for (bh, bw, vert_step, hor_step, P, R) in configs:
            # determine tile top-left coordinates with coverage to image end
            y_starts = list(range(0, max(1, H - bh + 1), vert_step))
            if y_starts[-1] + bh < H:
                y_starts.append(H - bh)
            x_starts = list(range(0, max(1, W - bw + 1), hor_step))
            if x_starts[-1] + bw < W:
                x_starts.append(W - bw)

            # compute padding needed for this radius
            pad_r = int(np.ceil(R))
            pad = pad_r + 2
            # pad the whole channel once per config (faster than per tile)
            p_ch = np.pad(ch, pad, mode='reflect')

            # precompute neighbor offsets (angles start LEFT, clockwise)
            thetas = np.pi + (-2.0 * np.pi) * (np.arange(P, dtype=np.float32) / P)
            ox = R * np.cos(thetas)
            oy = R * np.sin(thetas)

            # uniform mapping (if used)
            if use_uniform_u2:
                u_map = uniform_u2_map(P)
                num_bins = P + 2
            else:
                u_map = None
                num_bins = bins

            # iterate tiles (top-to-bottom, left-to-right)
            tile_hists = []
            for ys in y_starts:
                for xs in x_starts:
                    tile = ch[ys:ys + bh, xs:xs + bw].astype(np.float32)
                    h, w = tile.shape

                    # we will sample from padded ch using shifts
                    # top-left of tile in padded coords:
                    top = ys + pad
                    left = xs + pad

                    codes = []
                    for i in range(h):
                        for j in range(w):
                            ci = top + i
                            cj = left + j
                            center = p_ch[ci, cj]
                            code = 0
                            for k in range(P):  # MSB-first; k=0 corresponds to LEFT
                                yk = ci + oy[k]
                                xk = cj + ox[k]
                                nb = bilinear(p_ch, yk, xk)
                                code = (code << 1) | (1 if nb >= center else 0)
                            codes.append(code)

                    codes = np.asarray(codes, dtype=np.int32)

                    if use_uniform_u2:
                        mapped = u_map[codes]          # values in 0..P+1
                        hist = np.bincount(mapped, minlength=num_bins).astype(np.float32)
                    else:
                        # normal histogram over [0, 2**P)
                        hist, _ = np.histogram(codes, bins=num_bins, range=(0, 2 ** P))
                        hist = hist.astype(np.float32)

                    # L1 normalize
                    s = hist.sum()
                    if s > 0:
                        hist /= (s + 1e-12)
                    tile_hists.append(hist)

            # concatenate all tile histograms for this config (tile order preserved)
            if tile_hists:
                config_vec = np.concatenate(tile_hists, axis=0)
            else:
                config_vec = np.zeros(0, dtype=np.float32)
            channel_parts.append(config_vec)

        # finally concatenate across configs for this channel
        if channel_parts:
            ch_vec = np.concatenate(channel_parts, axis=0)
        else:
            ch_vec = np.zeros(0, dtype=np.float32)
        per_channel_features.append(ch_vec)

    return per_channel_features

def get_OCLBP_hist(image,
                   bins,
                   blocks,
                   P,
                   R,
                   use_uniform_u2=False,
                   sampling: str = 'nearest'):
    """
    Over-Complete LBP (OCLBP) descriptor with overlapped tiles.
    - Uses a fixed number of tile start positions per axis (n = 2*blocks - 1)
      computed via linspace so the descriptor length is constant across images
      of different sizes (no need to resize images).
    - Supports 'bilinear' or 'nearest' sampling for neighbor intensities.
    - If use_uniform_u2=True, applies the uniform-u2 mapping (bins = P+2).
      Otherwise, builds an exact histogram if bins == 2**P, or a uniformly
      rebinned histogram to the requested 'bins'.

    Parameters
    ----------
    image : (H,W) or (H,W,C) ndarray
        Input image, grayscale or color.
    bins : int
        Number of histogram bins when uniform-u2 is disabled.
    blocks : int
        Controls the tile size: tile_h = ceil(H/blocks), tile_w = ceil(W/blocks).
        The function places n = 2*blocks - 1 tile positions per axis (approx 50% overlap).
    P : int
        Number of neighbors for LBP.
    R : float
        Sampling radius.
    use_uniform_u2 : bool, default False
        If True, uses uniform-u2 mapping (bins per sub-hist = P + 2).
    sampling : {'bilinear','nearest'}, default 'nearest'
        Interpolation used to read neighbor intensities.

    Returns
    -------
    desc : (D,) ndarray
        Concatenated descriptor over all tiles and channels.
    """

    # Ensure 3D (add channel dim if needed) and float32 to avoid repeated casting
    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape
    img = image.astype(np.float32, copy=False)

    # Helper: fixed tile starts per axis (n = 2*blocks - 1) via linspace
    # This makes the number of tiles per axis independent of H/W (stable descriptor length).
    def fixed_starts_linspace(L, blocks):
        # Tile size for that axis
        bL = int(max(1, ceil(L / blocks)))
        # Number of start positions (with ~50% overlap idea)
        n = 2*blocks - 1
        if n == 1:
            return np.array([0], dtype=int), bL
        # Equally-spaced starts from 0 to L - bL (inclusive end approximated)
        starts = np.round(np.linspace(0, max(0, L - bL), n)).astype(int)
        return starts, bL

    # Tile starts and tile sizes for both axes
    y_starts, bh = fixed_starts_linspace(H, blocks)
    x_starts, bw = fixed_starts_linspace(W, blocks)

    # Padding to safely sample fractional coordinates near borders
    pad_r = int(np.ceil(R))
    pad = pad_r + 2

    # Neighbor offsets (start from LEFT, go clockwise)
    thetas = np.pi + (-2.0 * np.pi) * (np.arange(P, dtype=np.float32) / P)
    ox = (R * np.cos(thetas)).astype(np.float32)  # (P,)
    oy = (R * np.sin(thetas)).astype(np.float32)  # (P,)

    # Bit weights to pack P bits into a code in [0, 2**P)
    weights = (1 << np.arange(P-1, -1, -1, dtype=np.int32)).astype(np.int32)
    code_max = 1 << P

    # Optional uniform-u2 LUT (maps 0..2**P-1 -> 0..P+1)
    if use_uniform_u2:
        codes_lut = np.arange(code_max, dtype=np.int32)
        # bits_lut shape: (2**P, P) with MSB-first bit order
        bits_lut = ((codes_lut[:, None] >> np.arange(P-1, -1, -1)) & 1).astype(np.int32)
        # number of 0->1 / 1->0 transitions in the circular binary pattern
        trans = (bits_lut[:, :-1] ^ bits_lut[:, 1:]).sum(axis=1) + (bits_lut[:, -1] ^ bits_lut[:, 0])
        ones = bits_lut.sum(axis=1)
        u_map = np.empty(code_max, dtype=np.int32)
        u_map[trans <= 2] = ones[trans <= 2]
        u_map[trans >  2] = P + 1
        num_bins = P + 2
    else:
        u_map = None
        num_bins = int(bins)

    # Samplers ---------------------------------------------------------------
    def bilinear_sample(pad_img, y, x):
        """
        Vectorized bilinear interpolation for arbitrary float coordinates.
        pad_img : (Hp, Wp) float32
        y, x    : arrays of same shape (e.g., (P,h,w))
        """
        y0 = np.floor(y).astype(np.int32); x0 = np.floor(x).astype(np.int32)
        y1 = y0 + 1;                        x1 = x0 + 1
        wy = (y - y0).astype(np.float32);   wx = (x - x0).astype(np.float32)
        v00 = pad_img[y0, x0]; v01 = pad_img[y0, x1]
        v10 = pad_img[y1, x0]; v11 = pad_img[y1, x1]
        return (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * ((1.0 - wx) * v10 + wx * v11)

    def nearest_sample(pad_img, y, x):
        """
        Vectorized nearest-neighbor sampling (round to nearest integer indices).
        """
        yi = np.rint(y).astype(np.int32)
        xi = np.rint(x).astype(np.int32)
        return pad_img[yi, xi]

    if sampling not in ('bilinear', 'nearest'):
        raise ValueError("sampling must be 'bilinear' or 'nearest'")
    sample_fn = bilinear_sample if sampling == 'bilinear' else nearest_sample
    # ------------------------------------------------------------------------

    # Pre-build the center grid for a generic (bh, bw) tile
    yy, xx = np.mgrid[0:bh, 0:bw]
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    all_hists = []

    for c in range(C):
        ch = img[..., c]
        # Reflect padding per channel to safely sample neighbors near borders
        p_ch = np.pad(ch, pad, mode='reflect')

        # Iterate through fixed tile starts (same count for any H/W given 'blocks')
        for ys in y_starts:
            for xs in x_starts:
                # Crop the center grid at image borders if the last tile exceeds bounds
                h = min(bh, H - ys)
                w = min(bw, W - xs)
                if h <= 0 or w <= 0:
                    continue

                # Centers (in the padded image coordinate system)
                ci = ys + pad + yy[:h, :w]
                cj = xs + pad + xx[:h, :w]
                center = p_ch[ci.astype(np.int32), cj.astype(np.int32)]  # (h, w)

                # Neighbor coordinates for all P at once -> shapes (P, h, w)
                yk = ci[None, :, :] + oy[:, None, None]
                xk = cj[None, :, :] + ox[:, None, None]
                nb = sample_fn(p_ch, yk, xk)

                # Compare neighbors to center to get P bits per pixel, then pack to codes
                bits = (nb >= center[None, :, :]).astype(np.int32)     # (P, h, w)
                codes = (bits * weights[:, None, None]).sum(axis=0).ravel()  # (h*w,)

                # Histogram per tile
                if use_uniform_u2:
                    # Map codes through u2 LUT, then histogram into P+2 bins
                    mapped = u_map[codes]
                    hist = np.bincount(mapped, minlength=num_bins).astype(np.float32)
                else:
                    if num_bins == code_max:
                        # Exact code histogram (one bin per code)
                        hist = np.bincount(codes, minlength=code_max).astype(np.float32)[:num_bins]
                    else:
                        # Uniform rebin from [0..2**P) -> [0..bins-1]
                        bin_idx = (codes.astype(np.int64) * num_bins) // code_max
                        bin_idx = np.minimum(bin_idx, num_bins - 1).astype(np.int32)
                        hist = np.bincount(bin_idx, minlength=num_bins).astype(np.float32)

                # L1 normalization per tile
                s = hist.sum()
                if s > 0:
                    hist /= (s + 1e-12)

                all_hists.append(hist)

    # Concatenate all tile histograms across channels into a single descriptor
    desc = np.concatenate(all_hists, axis=0)
    # Lightweight debug summary (optional)
    print("OCLBP len:", desc.size, "min/max:", float(desc.min()), float(desc.max()), "L1:", float(desc.sum()))
    return desc

