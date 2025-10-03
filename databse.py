import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

import distances
import readability


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
            histogram = self.__compute_histogram(image, 32)

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
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
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
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
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
    
    def get_top_k_similar_images1( self, img_hist, distance_metric, k: int = 1, weights=None, ensemble_method: str = "score"):
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
                    d[idx] = distances.hist_intersection(img_hist, h)
                elif m == 'hellinger':
                    d[idx] = distances.hellinger_kernel(img_hist, h)
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
    #rel_path = r'c:\Users\maiol\Desktop\Master\C1\Projects\Project1\DATA'
    rel_path = r'C:\Users\maiol\Desktop\Master\C1\Projects\Project1\BBDD'
    db = Database(rel_path, debug= False, color_space='lab')
    print(f'Database length: {len(db)}')

    pattern = os.path.join(r'C:\Users\maiol\Desktop\Master\C1\Projects\Project1\qsd1_w1', '*.jpg')
    root_dir = os.path.abspath(os.path.expanduser(rel_path))
    results = []
    results2 = []
    for image_path in glob.iglob(pattern, root_dir=root_dir):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                hist = cv2.calcHist([image], [0], None, [32], [0, 256]).ravel()
        else:
            hists = [
                cv2.calcHist([image], [i], None, [32], [0, 256]).ravel()
                for i in range(image.shape[2])
            ]
            hist = np.concatenate(hists, axis=0)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        #save_histogram_jpg(hist, './input.jpg')
        k_info = db.get_top_k_similar_images(hist, 'hist_intersection')
        k_info2 = db.get_top_k_similar_images1(hist, ["l1"])
        results.append(k_info)
        results2.append(k_info2)

    with open(r"c:\Users\maiol\Desktop\Master\C1\Projects\Project1\qsd1_w1\gt_corresps.pkl", "rb") as f:   # 'rb' = read binary
        obj = pickle.load(f)
        
    print(results2)
    print(obj)

    matches = [i for i, (x, y) in enumerate(zip(results2, obj)) if x == y]
    count = len(matches)
    print(count, matches)