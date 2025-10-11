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

    # paths
    rel_path_db = r'C:\Users\User\OneDrive\Escritorio\Master\C1\Project\Team1\datasets\BBDD'
    query_glob  = os.path.join(r'C:\Users\User\OneDrive\Escritorio\Master\C1\Project\Team1\datasets\qsd1_w1', '*.jpg')
    gt_path     = r"C:\Users\User\OneDrive\Escritorio\Master\C1\Project\Team1\datasets\qsd1_w1\gt_corresps.pkl"

    # funcions utilitzades
    def convert_color_to_lab(img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

    def clahe_on_lab(img_lab, clip=2.0, tile=(8, 8)):
        """Apply CLAHE on L channel; keep a/b unchanged.""" 
        L, a, b = cv2.split(img_lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile))
        L2 = clahe.apply(L)
        return cv2.merge([L2, a, b])

    def compute_hist_1d_color(image, ch_bins):
        """
        image: 3-channel array (Lab here).
        ch_bins: tuple/list of 3 ints, one per channel.
        returns normalized 1D histogram (concatenated).
        """
        parts = []
        for i, b in enumerate(ch_bins):
            hi = cv2.calcHist([image], [i], None, [int(b)], [0, 256]).ravel()
            parts.append(hi)
        h = np.concatenate(parts, axis=0)
        # cv2.normalize(h, h, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        cv2.normalize(h,h, alpha=1, norm_type=cv2.NORM_L1)
        return h

    def _normalize_prob(h):
        h = np.asarray(h, dtype=np.float64).ravel()
        h = np.clip(h, 0.0, None)
        s = h.sum()
        if s <= 0:
            return np.zeros_like(h)
        return h / s

    def distances_vector_minimal(query_hist, db_hists, method):
        """
        Only the two used methods:
          - 'hist_intersection' (on L1-normalized hists)
          - 'canberra'
        """
        D = distances  

        q = np.asarray(query_hist, dtype=np.float64).ravel()
        d = np.empty(len(db_hists), dtype=np.float64)

        if method == "hist_intersection":
            qn = _normalize_prob(q)
            for i, H in enumerate(db_hists):
                Hn = _normalize_prob(H)
                d[i] = D.hist_intersection(qn, Hn)
            return d

        if method == "canberra":
            for i, H in enumerate(db_hists):
               d[i] = D.canberra_distance(q, H)
            return d
        
        if method == "l1":
            for i, H in enumerate(db_hists):
               d[i] = D.l1_distance(q, H)
            return d

        raise ValueError(f"Unsupported method here: {method}")

    def topk_indices(d, k):
        return sorted(range(len(d)), key=lambda i: (d[i], i))[:k]

    # gt
    with open(gt_path, "rb") as f:
        gt = pickle.load(f)

    # setups dels millors
    SETUPS = [
        
        {
            "name": "lab_128x3 + clahe + canberra",
            "bins": (128, 128, 128),
            "sim": "canberra",
        }

    ]

    for setup in SETUPS:
        bins_cfg = setup["bins"]
        sim_method = setup["sim"]

        # DB in Lab, apply CLAHE, compute histograms
        db = Database(rel_path_db, debug=False, color_space="lab")
        db_hists = []
        for img_lab in db.images:
            img_lab_p = clahe_on_lab(img_lab, clip=2.0, tile=(8, 8))
            db_hists.append(compute_hist_1d_color(img_lab_p, bins_cfg))
        db_hists = np.asarray(db_hists, dtype=object)

        # query histograms in Lab with CLAHE
        query_hists = []
        for qpath in glob.glob(query_glob):
            qbgr = cv2.imread(qpath)
            qlab = convert_color_to_lab(qbgr)
            qlab_p = clahe_on_lab(qlab, clip=2.0, tile=(8, 8))
            qhist = compute_hist_1d_color(qlab_p, bins_cfg)
            query_hists.append(qhist)

        assert len(query_hists) == len(gt), "Mismatch between queries and GT length."

        # evaluar els millors metodes  
        preds_k1, preds_k5 = [], []
        for qh in query_hists:
            dvec = distances_vector_minimal(qh, db_hists, sim_method)
            preds_k1.append(topk_indices(dvec, 1))
            preds_k5.append(topk_indices(dvec, 5))

        map1 = mapk(gt, preds_k1, k=1)
        map5 = mapk(gt, preds_k5, k=5)

        print(f"\n=== {setup['name']} ===")
        print(f"MAP@1: {map1:.4f}")
        print(f"MAP@5: {map5:.4f}")
