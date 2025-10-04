import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import metrics.distances as distances
import constants


def save_histogram_jpg(hist: np.ndarray, out_path: str, bins_per_channel: int = 64):
    """
    Save a histogram vector as a JPG plot.

    This function visualizes a histogram (either grayscale or multi-channel)
    and saves it as a `.jpg` image. It supports both single-channel histograms
    (e.g., grayscale with 256 bins) and concatenated multi-channel histograms
    (e.g., RGB histograms with `bins_per_channel * 3` bins).

    Parameters
    ----------
    hist : numpy.ndarray
        The histogram values to plot. Can represent one or multiple channels.
    out_path : str
        Output file path where the histogram image will be saved.
        The function creates any missing directories automatically.
    bins_per_channel : int, optional
        Number of bins per channel (default is 64). Used to infer the number
        of channels when the input histogram is flattened.
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
    """
    Container for the image database used by the retrieval algorithm.

    This class loads images (``*.jpg``), optional sidecar metadata (``*.txt``),
    and pre-computes per-image histograms for fast similarity search. It also
    exposes utilities to query the database and to compute top-k matches for a
    given query histogram.

    Parameters
    ----------
    path : str
        Root directory of the database. All ``*.jpg`` files in this directory
        are considered database entries.
    debug : bool, optional
        If True, prints additional information (e.g., ensemble details during
        retrieval). Default is ``False``.
    color_space : {"rgb", "hsv", "gray_scale", "lab"}, optional
        Color space used when loading images. Default is ``"rgb"``.
        - ``"lab"`` applies CLAHE on the L channel before merging (contrast enhancement).

    Attributes
    ----------
    color_space : str
        The color space used to load images.
    debug : bool
        Debug flag.
    image_paths : list of str
        Absolute paths to all ``*.jpg`` images found under ``path``.
    images : list of numpy.ndarray
        Loaded images converted to the requested color space.
    info : list of list[str]
        Parsed per-image metadata read from matching ``.txt`` files, if present.
        Each entry corresponds to one image; missing or invalid TXT files are skipped.
    histograms : list of numpy.ndarray
        L1-normalized histograms (concatenated per-channel for color images).
    """
    def __init__(self, path: str, bins: int, debug: bool=False, color_space: constants.COLOR_SPACES='rgb'):
        self.color_space = color_space
        self.debug = debug
        self.bins = bins
        
        self.__load_db(path)
    
    def __len__(self):
        """
        Return the number of images in the database.

        Returns
        -------
        int
            Total number of images indexed.
        """
        return len(self.images)

    def get_txts(self):
        """
        Return the number of TXT metadata entries parsed.

        Returns
        -------
        int
            Count of successfully parsed TXT files. This may be smaller than
            ``len(self)`` if some images have no TXT sidecar.
        """
        return len(self.info)
    
    def get_color_space(self):
        """
        Get the color space used by the database.

        Returns
        -------
        str
            One of ``{"rgb", "hsv", "gray_scale", "lab"}``.
        """
        return self.color_space
    
    def __load_db(self, db_path: str):
        """
        Load images, optional TXT metadata, and compute histograms.

        Parameters
        ----------
        db_path : str
            Path to the root directory containing ``*.jpg`` images.
        """
        self.image_paths = []
        self.images_raw = []
        self.images = []
        self.info = []
        self.histograms = []
        root_dir = os.path.abspath(os.path.expanduser(db_path))
        pattern = os.path.join(root_dir, '*.jpg')
        for image_path in sorted(glob.glob(pattern, root_dir=root_dir)):
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
        """
        Parse a TXT sidecar file into a list of strings.

        Parameters
        ----------
        txt_path : str
            Path to a ``.txt`` file associated with an image.

        Returns
        -------
        list of str
            Parsed tokens from the first line of the file.
        """
        with open(txt_path, 'r', encoding='ISO-8859-1') as f:
            line = f.readline()
            info = line.rstrip().strip('()').replace('\'', '').split(', ')
        return info

    def __load_img(self, img_path: str):
        """
        Load an image with OpenCV and convert it to the configured color space.

        Parameters
        ----------
        img_path : str
            Path to a JPEG image.

        Returns
        -------
        numpy.ndarray
            The loaded image in the requested color space.
        """
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, constants.CV2_CVT_COLORS[self.color_space])

        if self.color_space == 'lab':
            l, a, b = cv2.split(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            image = cv2.merge((l_eq, a, b))
        
        return image 

    def __compute_histogram(self, image: np.ndarray):
        """
        Compute an L1-normalized histogram for a grayscale or color image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image. Can be 2D (grayscale) or 3D with a channel dimension.
        bins : int, optional
            Number of bins per channel. Default is ``64``.

        Returns
        -------
        numpy.ndarray
            Flattened histogram. For grayscale, shape is ``(bins,)``.
            For multi-channel images, shape is ``(bins * C,)`` where ``C`` is
            the number of channels in ``image``.
        """

        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            hist = cv2.calcHist([image], [0], None, [self.bins], [0, 256]).ravel()
            cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
            """
            if self.debug:
                save_histogram_jpg(hist, './db.jpg')
            """
            return hist

        hists = [
            cv2.calcHist([image], [i], None, [self.bins], [0, 256]).ravel()
            for i in range(image.shape[2])
        ]
        hist = np.concatenate(hists, axis=0)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return hist
    
    def change_color(self, color_space):
        self.color_space = color_space
        for idx, image in enumerate(self.images_raw):
            self.images[idx] = cv2.cvtColor(image, constants.CV2_CVT_COLORS[self.color_space])
    
    def get_top_k_similar_images(self, img_hist, distance_metric, k: int = 1, weights=None, ensemble_method: str = "score"):
        """
        Retrieve indices of the top-k most similar database images to a query histogram.

        Parameters
        ----------
        img_hist : array_like
            L1-normalized query histogram (shape ``(bins,)`` for grayscale or
            ``(bins * C,)`` for color).
        distance_metric : str or sequence of str
            One or more metric names from:
            ``{"l1", "x2", "euclidean", "hist_intersection", "hellinger"}``.
        k : int, optional
            Number of nearest neighbors to return. Default is ``1``.
        weights : sequence of float, optional
            Per-metric weights. Must match the length of ``distance_metric``.
            If ``None``, all metrics are weighted equally.
        ensemble_method : {"rank", "score"}, optional
            Method to combine multiple metrics:
            - ``"rank"`` : Sum of per-metric ranks (more robust to scale).
            - ``"score"`` : Min–max normalize each distance vector, then weighted sum.

        Returns
        -------
        list of int
            Indices of the top-k most similar images (ascending by aggregate distance).
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
                #hist_intersection here returns a *distance* (``1 - similarity``)
                elif m == 'hist_intersection':
                    d[idx] = distances.hist_intersection(img_hist, h)
                elif m == 'hellinger':
                    d[idx] = distances.hellinger_kernel(img_hist, h)
                elif m == 'canberra':
                    d[idx] = distances.canberra_distance(img_hist, h)
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
