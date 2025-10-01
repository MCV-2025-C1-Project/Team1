import time

import os
import glob

import cv2

from typing import Literal

COLOR_SPACES = Literal['rgb', 'hsv']

class Database:
    """Class to store information of the whole database available to the algorithm."""
    def __init__(self, path: str, color_space: COLOR_SPACES='rgb'):
        self.color_space = 'rgb'
        
        self.load_db(path)
    
    def load_db(self, db_path: str):
        """Loads the consultation database."""
        self.images = []
        self.info = []
        root_dir = os.path.abspath(db_path)
        pattern = os.path.join(root_dir, '*.jpg')
        for image_path in glob.iglob(pattern, root_dir=root_dir):
            jpg_file = image_path
            txt_file = os.path.splitext(image_path)[0] + '.txt'

            self.images.append(self.load_img(jpg_file))
            self.info.append(self.parse_txt(txt_file))

    def parse_txt(self, txt_path: str):
        """Parses a txt file."""
        with open(txt_path, 'r', encoding='ISO-8859-1') as f:
            line = f.readline()
            info = line.rstrip().strip('()').replace('\'', '').split(', ')
        return info

    def load_img(self, img_path: str):
        """Loads an image."""
        if self.color_space == 'rgb':
            cv2_cvt_code = cv2.COLOR_BGR2RGB
        elif self.color_space == 'hsv':
            cv2_cvt_code = cv2.COLOR_BGR2HSV
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2_cvt_code)
        return image 

    def change_color_space(self, color_space: COLOR_SPACES):
        """TODO"""
        if self.color_space == 'rgb':
            cv2_cvt_code = cv2.COLOR_HSV2RGB
        elif self.color_space == 'hsv':
            cv2_cvt_code = cv2.COLOR_RGB2HSV

        for idx, image in enumerate(self.images):
            self.images[idx] = cv2.cvtColor(image, cv2_cvt_code)


if __name__ == "__main__":
    start_time = time.time()
    rel_path = '../Datasets/BBDD'
    db = Database(rel_path, color_space='rgb')
    print(f"Elapsed time {time.time() - start_time}")

    abs_path = '/home/adriangt2001/MCVC/C1/Project/Datasets/BBDD'
    db = Database(abs_path, color_space='hsv')
    
    compr_abs_path = '~/MCVC/C1/Project/Datasets/BBDD'
    db = Database(compr_abs_path, color_space='hsv')
    
