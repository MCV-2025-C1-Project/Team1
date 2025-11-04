import os
import cv2
import glob
import numpy as np

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
            generate_descriptor(img, self.kp_descriptor, **self.parameters)
            pass

    def get_similar(self):
        pass