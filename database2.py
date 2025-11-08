import glob
import os

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from keypoints_descriptors import generate_descriptor


class Database:
    def __init__(self, path: str):
        # In-memory storage for grayscale images and their descriptors
        self.images = []
        self.descriptors = []

        # Active descriptor family and its parameters (set via change_params)
        self.kp_descriptor = None
        self.parameters = {}

        # Load database images immediately, then precompute descriptors
        self.load_db(path)
        #self.process()

    def load_db(self, path: str):
        """Load all .jpg images from `path` as grayscale, preprocessed and size-capped."""
        self.images = []
        pattern = os.path.join(path, '*.jpg')
        file_list = sorted(glob.glob(pattern))
        for f in file_list:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = self._limit_size(img, 512)          # <- size reduction here
            self.images.append(img)

    @staticmethod
    def _limit_size(img: np.ndarray, max_side: int = 512) -> np.ndarray:
        """Light blur + downscale so the longest side is <= max_side."""
        h, w = img.shape[:2]
        ms = max(h, w)
        if ms > max_side:
            scale = max_side / ms
            # blur before strong downscale to reduce aliasing
            img = cv2.GaussianBlur(img, (3, 3), 1)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img
    
    def change_params(self, kp_descriptor: str | None = None, parameters: dict | None = None, autoprocess: bool = False):
        """
        Update the active keypoint/descriptor family and its parameters.
        If `autoprocess` is True, recompute descriptors for the whole DB.
        """
        if kp_descriptor is not None:
            self.kp_descriptor = kp_descriptor
        if parameters is not None:
            self.parameters = parameters
        if autoprocess:
            self.process()

    def process(self):
        """Regenerate descriptors for the entire database with current parameters."""
        self.__generate_descriptors()

    def __generate_descriptors(self):
        """
        For each database image, compute descriptors with the selected method.
        We store ONLY the descriptors (not the DB keypoints), because matching
        uses the query keypoints to build centroids.
        """
        self.descriptors = []
        for img in self.images:
            kp, desc = generate_descriptor(img, self.kp_descriptor, **self.parameters)
            self.descriptors.append(desc)

    def get_similar(self, kp, desc) -> list[list[int]]:
        from tqdm import tqdm

        # --- Guard against blank/degenerate queries ---
        if kp is None or len(kp) == 0 or desc is None or len(desc) == 0:
            return [[-1]]

        # 1) Matcher selection
        if self.kp_descriptor == 'sift':
            bf = cv2.BFMatcher.create(cv2.NORM_L2)
        elif self.kp_descriptor == 'orb':
            W = self.parameters.get('WTA_K', 2)
            norm = cv2.NORM_HAMMING if W == 2 else cv2.NORM_HAMMING2
            bf = cv2.BFMatcher.create(norm)
        elif self.kp_descriptor == 'akaze':
            bf = cv2.BFMatcher.create(cv2.NORM_HAMMING)

        # 2) Collect candidates (per-DB centroid)
        entries = []  # [(db_idx, num_good, cx, cy)]
        ratio = 0.75
        min_good = 10
        min_frac_good = 0.1

        # Progress bar around the slowest part: matching vs all DB descriptors
        for db_idx, db_desc in enumerate(self.descriptors):
            if db_desc is None or len(db_desc) == 0:
                continue

            matches = bf.knnMatch(desc, db_desc, k=2)

            good_fwd = []
            for pair in matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio * n.distance:
                    good_fwd.append(m)
            
            matches_bwd = bf.knnMatch(db_desc, desc, k=2)

            good_bwd = []
            for pair in matches_bwd:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio * n.distance:
                    good_bwd.append(m)
            
            good = []
            bwd_map = {m.queryIdx: m.trainIdx for m in good_bwd}
            for fwd_match in good_fwd:
                q_idx = fwd_match.queryIdx
                t_idx = fwd_match.trainIdx

                if t_idx in bwd_map and bwd_map[t_idx] == q_idx:
                    good.append(fwd_match)

            # gate by absolute and fractional thresholds
            required = max(min_good, int(np.ceil(min_frac_good * max(1, len(matches)))))
            if len(good) < min_good:
                continue

            entries.append((db_idx, len(good)))

        if entries is None or len(entries) == 0:
            return [-1]

        # 3) Sort by number of good matches (desc)
        entries.sort(key=lambda e: e[1], reverse=True)

        final_ids = [idx for idx, _ in entries]

        return final_ids

    def get_similar_simple(self, kp, desc,
                        ratio=0.75,       # Lowe ratio for SIFT/ORB
                        top_k=20,         # how many best distances per DB to aggregate
                        min_good=10,      # minimum good matches to consider a DB a candidate
                        dominance=1.4,    # how many times top1 count must exceed top2 to be "clearly best"
                        mean_gap=0.05):   # relative gap (12%) on mean distance to call it "clearly better"
        """
        Very simple retrieval:
        1) BFMatcher + Lowe ratio
        2) For each DB: collect 'good' matches, take the top_k smallest distances, compute their mean
        3) Rank by (good_count desc, mean_dist asc)
        4) Decide: [id] or [id1, id2] or [-1]
        Returns a FLAT list of ints, e.g. [104] or [104, 251] or [-1]
        """


        
        # --- guard
        if kp is None or len(kp) == 0 or desc is None or len(desc) == 0:
            return [-1]

        # matcher
        if self.kp_descriptor in ('sift', 'color_sift'):
            bf = cv2.BFMatcher.create(cv2.NORM_L2)
        elif self.kp_descriptor == 'orb':
            W = self.parameters.get('WTA_K', 2)
            norm = cv2.NORM_HAMMING if W == 2 else cv2.NORM_HAMMING2
            bf = cv2.BFMatcher.create(norm)
        else:
            return [-1]

        # collect simple stats per DB
        stats = []  # (db_idx, good_count, mean_bestK)
        for db_idx, db_desc in enumerate(self.descriptors):
            if db_desc is None or len(db_desc) == 0:
                continue

            # knn matches against this DB
            m2 = bf.knnMatch(desc, db_desc, k=2)
            good = []
            for pair in m2:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m.distance)

            if len(good) < min_good:
                continue

            # take K best distances (smaller is better)
            good = np.sort(np.array(good, dtype=np.float32))
            mean_best = float(good[:top_k].mean()) if len(good) > 0 else 1e9
            stats.append((db_idx, len(good), mean_best))

        if not stats:
            return [-1]

        # sort: more good matches first, then smaller mean distance
        stats.sort(key=lambda t: (-t[1], t[2]))

        # decision rules
        if len(stats) == 1:
            return [stats[0][0]]

        # unpack top entries
        id1, c1, m1 = stats[0]
        id2, c2, m2 = stats[1]
        c3 = stats[2][1] if len(stats) > 2 else 0
        m3 = stats[2][2] if len(stats) > 2 else m2 * (1.0 / (1.0 - mean_gap))  # safe default

        # 1) single clear winner?
        #    - top1 has many more matches than top2, OR
        #    - top1's mean distance is clearly smaller (by mean_gap)
        if (c1 >= dominance * max(1, c2)) and (m1 <= (1.0 - mean_gap) * m2):
            return [id1]

        # 2) two stand out vs the rest?
        #    - both have enough matches
        #    - and either their means are clearly better than #3, or counts stand out vs #3
        two_good_enough = (c1 >= min_good and c2 >= min_good)
        two_better_than_rest = (m1 <= (1.0 - mean_gap) * m3) or (m2 <= (1.0 - mean_gap) * m3) \
                            or (c1 >= dominance * max(1, c3)) or (c2 >= dominance * max(1, c3))
        if two_good_enough and two_better_than_rest:
            # return left->right order is not relevant here, so keep best first
            return [id1, id2]

        # 3) otherwise: everything looks similar → no confident decision
        return [-1]
