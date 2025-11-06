import glob
import os

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from keypoints_descriptors import generate_descriptor


class Database:
    def __init__(self, path: str):
        self.images = []
        self.descriptors = []

        # Create parameters
        self.kp_descriptor = None
        self.parameters = {}

        self.load_db(path)
        self.process()
        pass

    def load_db(self, path: str):
        self.images = []
        pattern = os.path.join(path, '*.jpg')
        file_list = sorted(glob.glob(pattern))
        for file in file_list:
            f = os.path.join(path, file)
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.images.append(img)
    
    def change_params(self, kp_descriptor: str | None = None, parameters: dict = None, autoprocess: bool = False):
        if kp_descriptor: self.kp_descriptor = kp_descriptor
        if parameters: self.parameters = parameters

        if autoprocess: self.process()

    def process(self):
        self.__generate_descriptors()

    def __generate_descriptors(self):
        self.descriptors = []
        for img in self.images:
            self.descriptors.append(generate_descriptor(img, self.kp_descriptor, **self.parameters))
    
    def get_similar(self, kp, desc) -> list[int]:
        if self.kp_descriptor in ('sift', 'color_sift'):
            bf = cv2.BFMatcher.create(cv2.NORM_L2)
        elif self.kp_descriptor == 'orb':
            if 'WTA_K' in self.parameters.keys() and self.parameters['WTA_K'] > 2:
                bf = cv2.BFMatcher.create(cv2.NORM_HAMMING)
            else:
                bf = cv2.BFMatcher.create(cv2.NORM_HAMMING2)
        
        match_list = []
        centroids = []
        for idx, db_desc in enumerate(self.descriptors):
            matches = bf.knnMatch(desc, db_desc)
            
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance: good.append(m)

                    
            points = []
            if len(good) == 0 or len(good) < len(matches) * 0.75: continue
            
            for m in good:
                idx = m.queryIdx
                (x, y) = kp[idx].pt
                points.append([x, y])
            points = np.array(points)
            xcentroid = np.mean(points[..., 0])
            ycentroid = np.mean(points[..., 1])

            match_list.append((idx, good))
            centroids.append(np.array([xcentroid, ycentroid]))

        if len(match_list) == 0:
            return [[-1]]
        
        match_list = sorted(match_list, lambda x: len(x[1]), reverse=True)

        # Cluster
        centroids = np.stack(centroids, axis=0)
        db = DBSCAN(eps=50, min_samples=15).fit(centroids)
        num_clusters = len(set(db.labels_) - {-1})
        final_matches = [[] for _ in num_clusters]
        for i, (idx, m) in enumerate(match_list):
            if db.labels_[i] != -1:
                final_matches[db.labels_[i]].append(idx)

        return final_matches
