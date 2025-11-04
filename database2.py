import os
import cv2
import glob
import numpy as np

import descriptors

class Database:
    def __init__(self, path: str):
        self.images = []
        self.descriptors = []

        # Create parameters

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
    
    def change_params(self, autoprocess: bool = False):
        

        if autoprocess: self.process()
        pass

    def process(self):

        pass

    def __generate_descriptors(self):
        self.descriptors = []
        for img in self.images:
            # TODO: Call descriptors generation
            pass

    def get_similar(self):
        pass