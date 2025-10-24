import numpy as np
import cv2
# try imports for DCT
_dct_2_available = False
try:
    # scipy >= 1.4 provides scipy.fftpack.dct or scipy.fft.dct; try fftpack first
    from scipy.fftpack import dct as _dct1
    def dct2(block):
        # 2D orthonormal DCT-II using scipy.fftpack.dct
        return _dct1(_dct1(block.T, norm='ortho').T, norm='ortho')
    _dct_2_available = True
except Exception:
    try:
        import cv2
        def dct2(block):
            # cv2.dct expects float32 and returns float32
            b = block.astype(np.float32)
            return cv2.dct(b)
        _dct_2_available = True
    except Exception:
        _dct_2_available = False

if not _dct_2_available:
    raise RuntimeError(
        "No DCT implementation found. Install scipy (`pip install scipy`) or OpenCV (`pip install opencv-python`)."
    )

def zigzag_indices(h, w):
    """Return list of (r,c) indices in zig-zag order for an h x w block."""
    # classical zigzag across diagonals
    idxs = []
    for s in range(h + w - 1):
        if s % 2 == 0:
            # even: go from bottom to top along this anti-diagonal
            r_start = min(s, h - 1)
            c_start = s - r_start
            r, c = r_start, c_start
            while r >= 0 and c < w:
                idxs.append((r, c))
                r -= 1; c += 1
        else:
            # odd: go from top to bottom along this anti-diagonal
            c_start = min(s, w - 1)
            r_start = s - c_start
            r, c = r_start, c_start
            while r < h and c >= 0:
                idxs.append((r, c))
                r += 1; c -= 1
    return idxs

def get_block_coords(H, W, block_h, block_w, step_h, step_w):
    """Return top-left coordinates (y,x) for sliding windows that cover the image.
       Ensure coverage of the right/bottom edges by adding a final window if needed."""
    y_starts = list(range(0, max(1, H - block_h + 1), step_h))
    if y_starts[-1] + block_h < H:
        y_starts.append(H - block_h)
    x_starts = list(range(0, max(1, W - block_w + 1), step_w))
    if x_starts[-1] + block_w < W:
        x_starts.append(W - block_w)
    return y_starts, x_starts

def get_DCT_descriptor_old(image,
                       blocks,
                       step=None,
                       coeffs=16,
                       use_zigzag=True,
                       per_channel=False,
                       normalize_block=True,
                       mode='blocks'):

    """
    Compute a DCT-based descriptor.

    Parameters
    ----------
    image : HxW or HxWxC numpy array (dtype float or uint)
    block_size : (h,w) size of block for block-DCT (default 8x8)
    step : (step_h, step_w) stride for sliding windows. If None, non-overlapping (step = block_size).
           For overlapping use step < block_size (e.g. step=(block_h//2, block_w//2)).
    coeffs : number of DCT coefficients to keep per block (taken in zig-zag or raster order)
    use_zigzag : whether to use zigzag ordering (default True)
    per_channel : if True return list of descriptors per channel; if False return single concatenated descriptor
    normalize_block : if True L2-normalize each block vector
    mode : 'blocks' (default) or 'global'
        - 'blocks': block-wise DCT and concat block descriptors
        - 'global': apply 2D DCT on whole image (per channel) and take top-left patch flattened

    Returns
    -------
    if per_channel: list of 1D numpy arrays [feat_ch0, feat_ch1, ...]
    else: single 1D numpy array (concatenation across channels)
    """
    
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape

    from math import ceil
    bh_grid = max(1, ceil(H / blocks))
    bw_grid = max(1, ceil(W / blocks))

    block_size=(bh_grid, bw_grid)
    


    if mode == 'global':
        # global DCT: compute DCT on full image per channel and take top-left patch flattened
        feats = []
        patch_h = block_size[0]
        patch_w = block_size[1]
        for c in range(C):
            ch = img[..., c].astype(np.float32)
            d = dct2(ch)
            # take top-left patch
            patch = d[:patch_h, :patch_w]
            if use_zigzag:
                zz = zigzag_indices(patch_h, patch_w)
                vec = np.array([patch[r, c] for (r, c) in zz], dtype=np.float32)
            else:
                vec = patch.ravel().astype(np.float32)
            vec = vec[:coeffs].astype(np.float32)
            if normalize_block:
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / (n + 1e-12)
            feats.append(vec)
        if per_channel:
            return feats
        else:
            return np.concatenate(feats, axis=0)

    # === block mode ===
    block_h, block_w = block_size
    if step is None:
        step_h, step_w = block_h, block_w
    else:
        step_h, step_w = step

    y_starts, x_starts = get_block_coords(H, W, block_h, block_w, step_h, step_w)

    # precompute zigzag indices for the block size
    if use_zigzag:
        zz = zigzag_indices(block_h, block_w)
    else:
        zz = [(i, j) for i in range(block_h) for j in range(block_w)]

    # make sure coeffs <= block_h*block_w
    max_coeffs = block_h * block_w
    coeffs = min(coeffs, max_coeffs)

    per_channel_feats = []
    for c in range(C):
        ch = img[..., c].astype(np.float32)
        block_vectors = []
        for ys in y_starts:
            for xs in x_starts:
                block = ch[ys:ys + block_h, xs:xs + block_w]
                if block.shape[0] != block_h or block.shape[1] != block_w:
                    # pad to block size (rare if coordinates were computed properly, but safe)
                    pad_h = block_h - block.shape[0]
                    pad_w = block_w - block.shape[1]
                    block = np.pad(block, ((0, pad_h), (0, pad_w)), mode='reflect')

                d = dct2(block)
                # extract coefficients in ordering
                vec = np.array([d[r, c] for (r, c) in zz], dtype=np.float32)[:coeffs]
                if normalize_block:
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / (n + 1e-12)
                block_vectors.append(vec)

        if block_vectors:
            ch_vec = np.concatenate(block_vectors, axis=0).astype(np.float32)
        else:
            ch_vec = np.zeros(0, dtype=np.float32)
        per_channel_feats.append(ch_vec)

    if per_channel:
        return per_channel_feats
    else:
        return np.concatenate(per_channel_feats, axis=0)


import numpy as np
from math import ceil

def get_DCT_descriptor(image,
                       blocks,
                       coeffs=16,
                       use_zigzag=True,
                       normalize_block=True):
    """
    DCT descriptor using a regular blocks×blocks partition (no overlap),
    always returning a single concatenated 1D vector across all channels.

    - If image size is not divisible by the tile size, the image is reflect-padded
      (bottom/right) so every tile has identical size.
    - Keeps the first `coeffs` per block in zig-zag (or row-major) order.
    - Optionally L2-normalizes each block vector.

    Parameters
    ----------
    image : (H,W) or (H,W,C) ndarray
        Input image (uint8/float/...); converted to float32 internally.
    blocks : int
        Number of tiles per axis. Total tiles = blocks * blocks (e.g., 4 -> 16 tiles).
    coeffs : int, default 16
        Number of DCT coefficients kept per block (after ordering).
    use_zigzag : bool, default True
        If True, read coefficients in zig-zag order (low→high frequencies).
        If False, read in row-major order.
    normalize_block : bool, default True
        If True, L2-normalize each block’s vector.

    Returns
    -------
    desc : (C * blocks*blocks * L,) ndarray
        Single concatenated descriptor over channels and tiles,
        where L = min(coeffs, block_h*block_w).
    """

    # --- ensure 3D and float32 once ---
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]
    img = img.astype(np.float32, copy=False)
    H, W, C = img.shape

    # --- fixed, regular grid (blocks × blocks), no overlap ---
    # tile size via ceil; pad bottom/right so grid fits exactly
    block_h = int(max(1, ceil(H / blocks)))
    block_w = int(max(1, ceil(W / blocks)))

    target_h = block_h * blocks
    target_w = block_w * blocks

    pad_bottom = max(0, target_h - H)
    pad_right  = max(0, target_w - W)

    if pad_bottom or pad_right:
        img = np.pad(img,
                     pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
                     mode='reflect')
        # now shape is exactly (target_h, target_w, C)

    # --- precompute order for this block size ---
    if use_zigzag:
        zz_pairs = zigzag_indices(block_h, block_w)  # iterable of (r,c)
        zz_flat = np.fromiter((r * block_w + c for r, c in zz_pairs), dtype=np.int64)
    else:
        zz_flat = np.arange(block_h * block_w, dtype=np.int64)

    L = min(coeffs, block_h * block_w)

    # --- tile starts (regular grid, no overlap) ---
    y_starts = [i * block_h for i in range(blocks)]
    x_starts = [j * block_w for j in range(blocks)]
    n_blocks = blocks * blocks

    # --- collect features per channel, then concatenate once at the end ---
    channel_vectors = []

    for c in range(C):
        ch = img[..., c]
        # preallocate (n_blocks, L) for speed
        out = np.empty((n_blocks, L), dtype=np.float32)
        bi = 0

        for ys in y_starts:
            y2 = ys + block_h
            for xs in x_starts:
                x2 = xs + block_w
                block = ch[ys:y2, xs:x2]  # exact (block_h, block_w)

                # 2D DCT, then take first L coefficients in chosen order
                d = dct2(block)              # expects float32, returns float32
                flat = d.ravel()
                vec = flat.take(zz_flat[:L])

                if normalize_block:
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / (n + 1e-12)

                out[bi] = vec
                bi += 1

        channel_vectors.append(out.ravel())   # shape: (n_blocks * L,)

    # Always return a single concatenated vector (channels first, then tiles)
    desc = np.concatenate(channel_vectors, axis=0)  # shape: (C * n_blocks * L,)
    return desc
