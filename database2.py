import os
import cv2
import glob
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
    
    def get_similar(self, desc) -> list[int]:
        
        if self.kp_descriptor in ('sift', 'color_sift'):
            bf = cv2.BFMatcher.create(cv2.NORM_L2)
        elif self.kp_descriptor == 'orb':
            if 'WTA_K' in self.parameters.keys() and self.parameters['WTA_K'] > 2:
                bf = cv2.BFMatcher.create(cv2.NORM_HAMMING)
            else:
                bf = cv2.BFMatcher.create(cv2.NORM_HAMMING2)
        
        match_list = []
        for idx, db_desc in enumerate(self.descriptors):
            matches = bf.knnMatch(desc, db_desc)
            
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance: good.append([m])

        if len(good) > len(matches) * 0.75:
            match_list.append((idx, good))        

        if len(match_list) == 0:
            return [-1]

        match_list = sorted(match_list, lambda x: len(x[1]), reverse=True)
        match_list = [m[0] for m in match_list]


        return match_list
