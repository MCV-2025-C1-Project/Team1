import os
import glob
import csv

import cv2
import numpy as np

import readability 
import distances
import pickle
import matplotlib.pyplot as plt

from average_precision import apk, mapk


def save_histogram_jpg(hist: np.ndarray, out_path: str, bins_per_channel: int = 64):
    """
    Saves a JPG plot of a histogram vector.
    Works for grayscale (256 bins) and concatenated color phists (256*bins_per_channel).
    """
    hist = np.asarray(hist).ravel()

    # Infer number of channels from length and bins_per_channel
    if hist.size % bins_per_channel != 0:
        # Fallback: treat the whole vector as one histogram
        C = 1
        bins_per_channel = hist.size
    else:
        C = hist.size // bins_per_channel

    parts = [hist[i*bins_per_channel:(i+1)*bins_per_channel] for i in range(C)]

    fig, ax = plt.subplots()
    x = np.arange(bins_per_channel)

    if C == 1:
        ax.bar(x, parts[0])
    else:
        # draw side-by-side bars per bin
        width = 0.9 / C
        for ci, h in enumerate(parts):
            # center the grouped bars around each integer x
            offset = (ci - (C - 1) / 2) * width
            ax.bar(x + offset, h, width=width, alpha=0.8, label=f'ch{ci}')
        ax.legend()

    ax.set_xlabel('Bin')
    ax.set_ylabel('Count')
    fig.tight_layout()

    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

class Database:
    """Class to store information of the whole database available to the algorithm."""
    def __init__(self, path: str, debug: bool=False, color_space: readability.COLOR_SPACES='rgb'):
        self.color_space = color_space
        self.debug = debug
        
        self.__load_db(path)
    
    def __len__(self):
        return len(self.images)

    def get_txts(self):
        return len(self.info)
    
    def get_color_space(self):
        return self.color_space
    
    def __load_db(self, db_path: str):
        """Loads the inner database."""
        self.image_paths = []
        self.images = []
        self.info = []
        self.histograms = []
        root_dir = os.path.abspath(os.path.expanduser(db_path))
        pattern = os.path.join(root_dir, '*.jpg')
        for image_path in glob.iglob(pattern, root_dir=root_dir):
            jpg_file = image_path
            image = self.__load_img(jpg_file)
            try:
                txt_file = os.path.splitext(image_path)[0] + '.txt'
                info = self.__parse_txt(txt_file)
                self.info.append(info)
            except:
                pass
            histogram = self.__compute_histogram(image)

            self.images.append(image)
            self.histograms.append(histogram)
            self.image_paths.append(jpg_file)

    def __parse_txt(self, txt_path: str):
        """Parses a txt file."""
        with open(txt_path, 'r', encoding='ISO-8859-1') as f:
            line = f.readline()
            info = line.rstrip().strip('()').replace('\'', '').split(', ')
        return info

    def __load_img(self, img_path: str):
        """Loads an image."""
        if self.color_space == 'rgb':
            cv2_cvt_code = cv2.COLOR_BGR2RGB
        elif self.color_space == 'hsv':
            cv2_cvt_code = cv2.COLOR_BGR2HSV
        elif self.color_space == 'gray_scale':
            cv2_cvt_code = cv2.COLOR_BGR2GRAY
        elif self.color_space == 'lab':
            cv2_cvt_code = cv2.COLOR_BGR2Lab
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2_cvt_code)
        return image 

    def __compute_histogram(self, image: np.ndarray, bins=64):
        """Computes a 256-bin histogram. Works for grayscale (1ch) and color (3ch)."""

        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).ravel()
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            """
            if self.debug:
                save_histogram_jpg(hist, './db.jpg')
            """
            return hist

        hists = [
            cv2.calcHist([image], [i], None, [bins], [0, 256]).ravel()
            for i in range(image.shape[2])
        ]
        hist = np.concatenate(hists, axis=0)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        """
        if self.debug:
            save_histogram_jpg(hist, './db.jpg')
        """
        return hist
    
    def get_top_k_similar_images(self, img_hist, distance_metric, k=1):

        distances_list = []
        for histogram in self.histograms:
            if distance_metric == 'l1':
                distance = distances.l1_distance(img_hist, histogram)
            elif distance_metric == 'x2':
                distance = distances.x2_distance(img_hist,histogram)
            elif distance_metric == 'euclidean':
                distance = distances.euclidean_distance(img_hist,histogram)
            elif distance_metric == 'hist_intersection': 
                distance = distances.hist_intersection(img_hist,histogram)
            elif distance_metric == 'hellinger_kernel':
                distance = distances.hellinger_kernel(img_hist,histogram)
            distances_list.append(distance)

        k_lowest_distance_idxs = sorted(range(len(distances_list)), key=lambda i: (distances_list[i], i))[:k]

        if self.debug:
            k_hist = [self.histograms[i] for i in k_lowest_distance_idxs]
            save_histogram_jpg(k_hist, './db.jpg')
        return k_lowest_distance_idxs
    
    def get_top_k_similar_images1( self, img_hist, distance_metric, k: int = 1, weights=None, ensemble_method: str = "rank"):
        """
        distance_metric: str OR list/tuple of str, e.g. "x2" or ["x2","l1","euclidean"]
        ensemble_method:
            - "rank": sum of ranks per metric (robust to scale)  <-- recommended
            - "score": min-max normalize per metric, then weighted sum of distances
        weights: optional list of floats, same length as metrics
        """

        # Helper to compute distances (convert similarities to distances if needed)
        def _dist_for_metric(m):
            d = np.empty(len(self.histograms), dtype=np.float64)
            for idx, h in enumerate(self.histograms):
                if m == 'l1':
                    d[idx] = distances.l1_distance(img_hist, h)
                elif m == 'x2':
                    d[idx] = distances.x2_distance(img_hist, h)
                elif m == 'euclidean':
                    d[idx] = distances.euclidean_distance(img_hist, h)
                elif m == 'hist_intersection':
                    sim = distances.hist_intersection(img_hist, h)
                    d[idx] = 1.0 - sim
                elif m == 'hellinger_kernel':
                    sim = distances.hellinger_kernel(img_hist, h)
                    d[idx] = np.sqrt(max(0.0, 1.0 - float(sim)))
                else:
                    raise ValueError(f"Unknown metric: {m}")
            return d

        metrics = distance_metric if isinstance(distance_metric, (list, tuple)) else [distance_metric]
        if weights is None:
            weights = [1.0] * len(metrics)
        if len(weights) != len(metrics):
            raise ValueError("weights must have same length as metrics")

        # Compute per-metric distance vectors
        per_metric_d = [ _dist_for_metric(m) for m in metrics ]

        n = len(self.histograms)
        if ensemble_method == "rank":
            agg = np.zeros(n, dtype=np.float64)
            for w, d in zip(weights, per_metric_d):
                order = np.argsort(d)
                ranks = np.empty(n, dtype=np.float64)
                ranks[order] = np.arange(n)
                agg += w * ranks
            k_idx = np.argsort(agg)[:k]  

        elif ensemble_method == "score":
            agg = np.zeros(n, dtype=np.float64)
            for w, d in zip(weights, per_metric_d):
                d = d.astype(np.float64)
                dmin, dmax = d.min(), d.max()
                norm = (d - dmin) / (dmax - dmin + 1e-12)  
                agg += w * norm
            k_idx = np.argsort(agg)[:k]
        else:
            raise ValueError('ensemble_method must be "rank" or "score"')

        if self.debug:
            # Example: show which metrics/weights decided the winners
            print("Ensemble:", list(zip(metrics, weights)))
            print("Top-k indices:", k_idx.tolist())

        return k_idx.tolist()

    def change_color_space(self, color_space: readability.COLOR_SPACES):
        """TODO"""
        if self.color_space == 'rgb':
            cv2_cvt_code = cv2.COLOR_HSV2RGB
        elif self.color_space == 'hsv':
            cv2_cvt_code = cv2.COLOR_RGB2HSV
        elif self.color_space == 'gray_scale':
            cv2_cvt_code = cv2.COLOR_BGR2GRAY

        for idx, image in enumerate(self.images):
            self.images[idx] = cv2.cvtColor(image, cv2_cvt_code)


if __name__ == "__main__":
    # -------- Paths --------
    rel_path_db = r'C:\Users\User\OneDrive\Escritorio\Master\C1\Project\Team1\datasets\BBDD'
    query_glob  = os.path.join(r'C:\Users\User\OneDrive\Escritorio\Master\C1\Project\Team1\datasets\qsd1_w1', '*.jpg')
    gt_path     = r"C:\Users\User\OneDrive\Escritorio\Master\C1\Project\Team1\datasets\qsd1_w1\gt_corresps.pkl"
    csv_path    = "TESTS_v3.csv"
 
    # -------- Grid to search --------:)
    HIST_GRID = {
                # === HSV configurations ===
        "hsv_50_30_30": ("hsv", (50, 30, 30)), 
        "hsv_60_20_20": ("hsv", (60, 20, 20)),   
        "hsv_20_60_20": ("hsv", (20, 60, 20)), 
        "hsv_32x3": ("hsv", (32, 32, 32)),      
        "hsv_90_15_15": ("hsv", (90, 15, 15)),   
        "hsv_72_18_18": ("hsv", (72, 18, 18)),   
        "hsv_45_22_22": ("hsv", (45, 22, 22)),  

        # === Lab configurations ===
        "lab_30x3": ("lab", (30, 30, 30)),
        "lab_40x3": ("lab", (40, 40, 40)),
        "lab_45x3": ("lab", (45, 45, 45)),
        "lab_64x3": ("lab", (64, 64, 64)),
        "lab_16x3": ("lab", (16, 16, 16)),      
        "lab_128x3": ("lab", (128, 128, 128)),  

    }

    SIM_METHODS = [
  
        "canberra",
    
    ]


    """
    HIST_GRID = {
        # === Grayscale configurations ===
        "gray16": ("gray_scale", 16),
        "gray30": ("gray_scale", 30),
        "gray64": ("gray_scale", 64),
        "gray128": ("gray_scale", 128),
        "gray256": ("gray_scale", 256),

        # === RGB configurations (concatenate per-channel histograms) ===
        "rgb16x3": ("rgb", (16, 16, 16)),
        "rgb30x3": ("rgb", (30, 30, 30)),
        "rgb64x3": ("rgb", (64, 64, 64)),
        "rgb128x3": ("rgb", (128, 128, 128)),

        # === HSV configurations ===
        "hsv_50_30_30": ("hsv", (50, 30, 30)), 
        "hsv_60_20_20": ("hsv", (60, 20, 20)),   
        "hsv_20_60_20": ("hsv", (20, 60, 20)), 
        "hsv_32x3": ("hsv", (32, 32, 32)),      
        "hsv_90_15_15": ("hsv", (90, 15, 15)),   
        "hsv_72_18_18": ("hsv", (72, 18, 18)),   
        "hsv_45_22_22": ("hsv", (45, 22, 22)),  

        # === Lab configurations ===
        "lab_30x3": ("lab", (30, 30, 30)),
        "lab_64x3": ("lab", (64, 64, 64)),
        "lab_16x3": ("lab", (16, 16, 16)),      
        "lab_128x3": ("lab", (128, 128, 128)),  
    }

    SIM_METHODS = [
        "l1",
        "x2",
        "euclidean",
        "hist_intersection",   
        "hellinger_kernel",    
        "cosine",
        "chebyshev",
        "canberra",
        "braycurtis",
        "bhattacharyya",      
        "js_divergence"       
    ]

    """
    # --- NEW: preprocessing helpers ---------------------------------------------
    def _apply_on_value_channel(img, space, fn):
        """
        Apply a function `fn(channel) -> channel` to the luminance/value channel
        based on the working color space. Returns an image in the same space.
        """
        if space == "gray_scale":
            return fn(img)

        if space == "lab":
            L, a, b = cv2.split(img)
            L2 = fn(L)
            return cv2.merge([L2, a, b])

        # For HSV/RGB, edit the V channel in HSV space, then convert back
        if space in ("rgb", "hsv"):
            hsv = img if space == "hsv" else cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            v2 = fn(v)
            hsv2 = cv2.merge([h, s, v2])
            if space == "hsv":
                return hsv2
            else:
                return cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

        # Default: unchanged
        return img


    def _gamma_LUT(gamma=1.0):
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)]).astype("uint8")
        return table


    def preprocess_image(img, space, cfg):
        """
        img: numpy array already in the target color space (rgb/hsv/lab/gray_scale).
        cfg: dict like those in PREPROCESS_GRID: {'type', ...params...}.
        Returns an image processed in the SAME color space.
        """
        if cfg is None or cfg.get("type") == "none":
            return img

        t = cfg["type"]

        if t == "clahe":
            clip = float(cfg.get("clipLimit", 2.0))
            tile = tuple(cfg.get("tileGridSize", (8, 8)))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
            return _apply_on_value_channel(img, space, lambda ch: clahe.apply(ch))

        if t == "hist_eq":
            return _apply_on_value_channel(img, space, lambda ch: cv2.equalizeHist(ch))

        if t == "gamma":
            g = float(cfg.get("gamma", 1.2))
            lut = _gamma_LUT(g)
            # Apply gamma on V/L/gray only
            return _apply_on_value_channel(img, space, lambda ch: cv2.LUT(ch, lut))

        if t == "contrast":
            # alpha: contrast (1.0 = same), beta: brightness shift
            alpha = float(cfg.get("alpha", 1.2))
            beta  = float(cfg.get("beta", 0.0))

            def _cb(ch):
                return cv2.convertScaleAbs(ch, alpha=alpha, beta=beta)

            return _apply_on_value_channel(img, space, _cb)

        if t == "gaussian_blur":
            k = int(cfg.get("ksize", 3))
            k = k + (1 - k % 2)  # ensure odd
            sig = float(cfg.get("sigma", 0))
            if img.ndim == 2:
                return cv2.GaussianBlur(img, (k, k), sig)
            else:
                return cv2.GaussianBlur(img, (k, k), sig)

        if t == "median_blur":
            k = int(cfg.get("ksize", 3))
            k = k + (1 - k % 2)
            return cv2.medianBlur(img, k)

        if t == "bilateral":
            d     = int(cfg.get("d", 7))
            sigma = float(cfg.get("sigma", 50))
            return cv2.bilateralFilter(img, d, sigma, sigma)

        if t == "unsharp":
            # Simple unsharp masking in the current space
            k = int(cfg.get("ksize", 3))
            k = k + (1 - k % 2)
            amount = float(cfg.get("amount", 1.0))
            if img.ndim == 2:
                blurred = cv2.GaussianBlur(img, (k, k), 0)
                sharpen = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
                return sharpen
            else:
                blurred = cv2.GaussianBlur(img, (k, k), 0)
                sharpen = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
                return sharpen

        # Fallback: unchanged
        return img


    # --- NEW: preprocessing grid -------------------------------------------------
    PREPROCESS_GRID = {
        "clahe":             {"type": "clahe", "clipLimit": 2.0, "tileGridSize": (8, 8)},

    }


    def compute_hist_1d(image, bins_cfg, preprocess=None):
        """
        image: np.ndarray already in the chosen color space.
        bins_cfg: int or tuple/list of 3 ints (per channel).
        Returns a 1D normalized histogram.
        """
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1): 
            # Grayscale
            bins = bins_cfg if isinstance(bins_cfg, int) else int(bins_cfg)
            h = cv2.calcHist([image], [0], None, [bins], [0, 256]).ravel()
            cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return h
        
        else:
            # Color
            if isinstance(bins_cfg, (list, tuple)):
                ch_bins = list(bins_cfg)
                if len(ch_bins) != image.shape[2]:
                    raise ValueError("bins_cfg must match number of channels.")
            else:
                ch_bins = [int(bins_cfg)] * image.shape[2]

            parts = []
            for i, b in enumerate(ch_bins):
                hi = cv2.calcHist([image], [i], None, [b], [0, 256]).ravel()
                parts.append(hi)
            h = np.concatenate(parts, axis=0)
            cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            return h

    def convert_color(img_bgr, target_space):

        if target_space == "rgb":
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif target_space == "hsv":
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        elif target_space == "gray_scale":
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        elif target_space == "lab":
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
        else:
            raise ValueError("Unknown color space")

    def _normalize_prob(h):
        h = np.asarray(h, dtype=np.float64).ravel()
        h = np.clip(h, 0.0, None)               
        s = h.sum()
        if s <= 0:
            return np.zeros_like(h)          
        return h / s

    def distances_vector(query_hist, db_hists, method):
        """
        Compute distances (lower is better) between `query_hist` and each H in `db_hists`.
        Uses dist.<metric> (or distances.<metric> as a fallback).
        Probability-based metrics are L1-normalized first.
        """
        D = distances # import distances as D !! Vigilar aqui
            
        q = np.asarray(query_hist, dtype=np.float64).ravel()
        d = np.empty(len(db_hists), dtype=np.float64)

        prob_methods = {"hist_intersection", "hellinger_kernel", "bhattacharyya", "js_divergence"}
        use_prob = method in prob_methods  # probability-based metrics
        if use_prob:
            qn = _normalize_prob(q)

        for i, H in enumerate(db_hists):
            H = np.asarray(H, dtype=np.float64).ravel()

            if method == 'l1':
                d[i] = D.l1_distance(q, H)

            elif method == 'x2':
                d[i] = D.x2_distance(q, H)

            elif method == 'euclidean':
                d[i] = D.euclidean_distance(q, H)

            elif method == 'hist_intersection':
                Hn = _normalize_prob(H)
                d[i] = D.hist_intersection(qn, Hn)

            elif method == 'hellinger_kernel':
                Hn = _normalize_prob(H)
                d[i] = D.hellinger_kernel(qn, Hn)

            elif method == 'cosine':
                d[i] = D.cosine_distance(q, H)

            elif method == 'chebyshev':
                d[i] = D.chebyshev_distance(q, H)

            elif method == 'canberra':
                d[i] = D.canberra_distance(q, H)

            elif method == 'braycurtis':
                d[i] = D.braycurtis_distance(q, H)

            elif method == 'bhattacharyya':
                Hn = _normalize_prob(H)
                d[i] = D.bhattacharyya_distance(qn, Hn)

            elif method == 'js_divergence':
                Hn = _normalize_prob(H)
                d[i] = D.js_divergence(qn, Hn) 

            else:
                raise ValueError(f"Unknown similarity method: {method}")

        return d


    def topk_indices(d, k):
        return sorted(range(len(d)), key=lambda i: (d[i], i))[:k]

    # --- CSV header: add Preprocess ---------------------------------------------
    header = ["Histogram_Method", "Preprocess", "Similarity_Method", "MAP@1", "MAP@5"]
    file_exists = os.path.exists(csv_path)
    if not file_exists or os.path.getsize(csv_path) == 0:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    # --- run grid search ---------------------------------------------------------

    with open(gt_path, 'rb') as f:
        gt = pickle.load(f)
        
    for hist_name, (db_space, bins_cfg) in HIST_GRID.items():

        db = Database(rel_path_db, debug=False, color_space=db_space)
        print(f'\n[{hist_name}] Database length: {len(db)} | color_space={db_space}')

        # Iterate over preprocessing options
        for pre_name, pre_cfg in PREPROCESS_GRID.items():

            # --- compute DB histograms with preprocessing -----------------------
            db_hists = []
            for img in db.images:  # img already in db_space
                img_p = preprocess_image(img, db_space, pre_cfg)
                db_hists.append(compute_hist_1d(img_p, bins_cfg))
            db_hists = np.asarray(db_hists, dtype=object)

            # --- compute QUERY histograms with the same preprocessing -----------
            query_hists = []
            q_root = os.path.abspath(os.path.expanduser(rel_path_db))
            for qpath in glob.iglob(query_glob, root_dir=q_root):
                qbgr = cv2.imread(qpath)
                qimg = convert_color(qbgr, db_space)
                qimg = preprocess_image(qimg, db_space, pre_cfg)
                qhist = compute_hist_1d(qimg, bins_cfg)
                query_hists.append(qhist)

            

            assert len(query_hists) == len(gt), "Mismatch between queries and GT length."

            # --- evaluate similarity methods -----------------------------------
            for sim_method in SIM_METHODS:
                preds_k1, preds_k5 = [], []

                for qh in query_hists:
                    dvec = distances_vector(qh, db_hists, sim_method)
                    preds_k1.append(topk_indices(dvec, 1))
                    preds_k5.append(topk_indices(dvec, 5))

                map1 = mapk(gt, preds_k1, k=1)
                map5 = mapk(gt, preds_k5, k=5)

                print(f"{hist_name:>14} | {pre_name:>12} | {sim_method:>18} -> MAP@1: {map1:.4f} | MAP@5: {map5:.4f}")

                with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([hist_name, pre_name, sim_method, f"{map1:.4f}", f"{map5:.4f}"])
