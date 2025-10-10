import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import constants
import metrics.distances as distances
from operations import histograms, preprocessing


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
    and prepares structures for later histogram computation (performed when
    change_color is first invoked). It also exposes utilities to query the database
    and to compute top-k matches for a given query histogram.

    Parameters
    ----------
    path : str
        Root directory of the database. All ``*.jpg`` files in this directory
        are considered database entries.
    bins : int
        Number of histogram bins per channel.
    color_space : {"rgb", "hsv", "gray_scale", "lab"}, optional
        Color space used when processing images. Default ``"rgb"``.
    preprocess : str or None, optional
        Preprocessing operation key applied after color conversion (e.g. 'clahe').
        If None, no preprocessing is applied.
    debug : bool, optional
        If True, prints additional information (e.g., ensemble details during
        retrieval). Default is ``False``.

    Attributes
    ----------
    color_space : str
        Current color space applied to processed images.
    preprocess : str or None
        Current preprocessing method tag.
    bins : int
        Number of bins per channel used for histogram computation.
    image_paths : list[str]
        Absolute paths to all discovered ``*.jpg`` images.
    images_raw : list[numpy.ndarray]
        Original BGR images as read by OpenCV (unconverted, unprocessed).
    images : list[numpy.ndarray or None]
        Processed images in the active color space.
    info : list[list[str]]
        Parsed metadata tokens from matching ``.txt`` files (if present).
    histograms : list[numpy.ndarray or None]
        L1-normalized histograms (computed after change_color).
    debug : bool
        Debug flag.
    """
    def __init__(self, path: str, color_space: constants.COLOR_SPACES='rgb', preprocess=None, hist_dims: int = 1, bins: int = 64, hierarchy: int = 1, debug: bool=False):
        self.color_space = color_space
        self.preprocess = preprocess
        self.hist_dims = hist_dims
        self.bins = bins
        self.hierarchy = hierarchy
        self.debug = debug
        
        self.__load_db(path)
        self.change_color(color_space, preprocess)
    
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
        Load raw images and optional TXT metadata into memory.

        This does NOT compute histograms nor apply color conversion; those steps
        occur later in change_color.

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

            self.image_paths.append(jpg_file)
            self.images_raw.append(image)
            self.images.append(None)
            self.histograms.append(None)

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
        Load an image with OpenCV (BGR order) without color conversion.

        Color space conversion and preprocessing are deferred to change_color.

        Parameters
        ----------
        img_path : str
            Path to a JPEG image.

        Returns
        -------
        numpy.ndarray
            Loaded BGR image (uint8).
        """
        image = cv2.imread(img_path)
        
        return image 

    def __compute_histogram(self, image: np.ndarray):
        """
        Compute an L1-normalized histogram for a grayscale or color image.

        Uses the instance-wide self.bins value for each channel.

        Parameters
        ----------
        image : numpy.ndarray
            2D (grayscale) or 3D (H, W, C) uint8 image.

        Returns
        -------
        numpy.ndarray
            Flattened histogram. For grayscale, shape is ``(bins,)``.
            For multi-channel images, shape is ``(bins * C,)`` where ``C`` is
            the number of channels in ``image``.
        """
        if self.hist_dims == 1:
            hist = histograms.gen_1d_hist(image, self.hierarchy, self.bins)
        elif self.hist_dims == 2:
            hist = histograms.gen_2d_hist(image, self.hierarchy, self.bins)
        elif self.hist_dims == 3:
            hist = histograms.gen_3d_hist(image, self.hierarchy, self.bins)
        for idx, histogram in enumerate(hist):
            cv2.normalize(histogram, hist[idx], alpha=1.0, norm_type=cv2.NORM_L1)
        return hist
    
    def __preprocess_image(self, image):
        """
        Apply the configured preprocessing operation to an image.

        This internal helper dispatches to the corresponding function in
        operations.preprocessing based on the value of self.preprocess. If no
        preprocessing method is configured (None or unknown key), the input
        image is returned unchanged.

        Parameters
        ----------
        image : numpy.ndarray
            Image already converted to the current database color space
            (see self.color_space). Expected dtype uint8.

        Returns
        -------
        numpy.ndarray
            Preprocessed (or original) image.

        Notes
        -----
        Supported values for self.preprocess:
        - 'clahe'
        - 'hist_eq'
        - 'gamma'
        - 'contrast'
        - 'gaussian_blur'
        - 'median_blur'
        - 'bilateral'
        - 'unsharp'
        Any other value results in a no-op.
        """

        if self.preprocess == 'clahe':
            preprocessed_image = preprocessing.clahe_preprocessing(image, self.color_space)
        elif self.preprocess == 'hist_eq':
            preprocessed_image = preprocessing.hist_eq(image, self.color_space)
        elif self.preprocess == 'gamma':
            preprocessed_image = preprocessing.gamma(image)
        elif self.preprocess == 'contrast':
            preprocessed_image = preprocessing.contrast(image)
        elif self.preprocess == 'gaussian_blur':
            preprocessed_image = preprocessing.gaussian_blur(image)
        elif self.preprocess == 'median_blur':
            preprocessed_image = preprocessing.median_blur(image)
        elif self.preprocess == 'bilateral':
            preprocessed_image = preprocessing.bilateral(image)
        elif self.preprocess == 'unsharp':
            preprocessed_image = preprocessing.unsharp(image)
        else:
            preprocessed_image = image

        return preprocessed_image
    
    def change_color(self, color_space, preprocess=None):
        """
        Recompute database images and histograms in a new color space (and optional preprocessing).

        Converts every raw image to the specified color_space, then applies the
        selected preprocessing pipeline, and finally recomputes (and overwrites)
        the per-image histograms.

        Parameters
        ----------
        color_space : {"rgb", "hsv", "gray_scale", "lab"}
            Target color space for subsequent processing and retrieval.
        preprocess : str or None, optional
            Preprocessing operation key (see __preprocess_image for supported
            values). If None, no preprocessing is applied.

        Returns
        -------
        None

        Notes
        -----
        This operation is O(N) over the number of images and invalidates
        previous histograms.
        """
        self.color_space = color_space
        self.preprocess = preprocess
        for idx, image in enumerate(self.images_raw):
            processed_image = cv2.cvtColor(image, constants.CV2_CVT_COLORS[self.color_space])
            processed_image = self.__preprocess_image(processed_image)
            
            self.images[idx] = processed_image
            self.histograms[idx] = self.__compute_histogram(processed_image)

    def change_hist(self, hierarchy, hist_dims, bins):
        """
        Update the number of histogram bins and recompute all histograms.

        Parameters
        ----------
        bins : int
            New number of bins per channel used for histogram computation.

        Returns
        -------
        None

        Notes
        -----
        Only histograms are recomputed; images (and preprocessing) are left untouched.
        """
        self.hierarchy = hierarchy
        self.hist_dims = hist_dims
        self.bins = bins
        for idx, image in enumerate(self.images):
            self.histograms[idx] = self.__compute_histogram(image)

    def get_top_k_similar_images(self, img_hist, distance_metric, k: int = 1, weights=None, ensemble_method: str = "score"):
        """
        Retrieve indices of the top-k most similar database images to a query histogram.

        Parameters
        ----------
        img_hist : numpy.ndarray
            Query histogram (shape ``(bins,)`` for grayscale or
            ``(bins * C,)`` for color).
        distance_metric : str or sequence of str
            One or more metric names from:
            ``{"l1", "x2", "euclidean", "hist_intersection", "hellinger", "canberra"}``.
        k : int, optional
            Number of nearest neighbors to return. Default is ``1``.
        weights : sequence of float, optional
            Per-metric weights. Must match the length of ``distance_metric``.
            If ``None``, all metrics are weighted equally.
        ensemble_method : {"rank", "score"}, optional
            Method to combine multiple metrics:
            - ``"rank"`` : Sum of per-metric ranks (more robust to scale).
            - ``"score"`` : Min-max normalize each distance vector, then weighted sum.

        Returns
        -------
        list[int]
            Indices of the top-k most similar images (ascending by aggregate distance).
        """

        # Helper to compute distances (convert similarities to distances if needed)
        def _dist_for_metric(m):
            d = np.empty(len(self.histograms), dtype=np.float64)
            for idx, h in enumerate(self.histograms):
                hier_dist = []
                for level in range(self.hierarchy):
                    if m == 'l1':
                        hier_dist.append(distances.l1_distance(img_hist[level], h[level]))
                    elif m == 'x2':
                        hier_dist.append(distances.x2_distance(img_hist[level], h[level]))
                    elif m == 'euclidean':
                        hier_dist.append(distances.euclidean_distance(img_hist[level], h[level]))
                    #hist_intersection here returns a *distance* (``1 - similarity``)
                    elif m == 'hist_intersection':
                        hier_dist.append(distances.hist_intersection(img_hist[level], h[level]))
                    elif m == 'hellinger':
                        hier_dist.append(distances.hellinger_kernel(img_hist[level], h[level]))
                    elif m == 'canberra':
                        hier_dist.append(distances.canberra_distance(img_hist[level], h[level]))
                    else:
                        raise ValueError(f"Unknown metric: {m}")
                d[idx] = sum(hier_dist)
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
