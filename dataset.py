import os
import glob

import cv2
import numpy as np

import readability

class Dataset:
    """Class for the query dataset."""
    def __init__(self, path: str, color_space: readability.COLOR_SPACES='rgb'):
        self.color_space = color_space
        self.__load_dataset(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx]
    
    def __load_dataset(self, data_path: str):
        """Load the query dataset."""
        self.images = []
        self.histograms = []
        root_dir = os.path.abspath(os.path.expanduser(data_path))
        pattern = os.path.join(root_dir, '*.jpg')
        for image_path in glob.iglob(pattern, root_dir=root_dir):
            image = self.__load_img(image_path)
            histogram = self.__compute_histogram(image)

            self.images.append(image)
            self.histograms.append(histogram)

    def __load_img(self, img_path: str):
        """Loads an image."""
        if self.color_space == 'rgb':
            cv2_cvt_code = cv2.COLOR_BGR2RGB
        elif self.color_space == 'hsv':
            cv2_cvt_code = cv2.COLOR_BGR2HSV
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2_cvt_code)
        return image
    
    def __compute_histogram(self, image: np.ndarray):
        """Computes the histogram of a given image."""
        H, W, C = image.shape
        image_size = H * W
        histogram = [cv2.calcHist(image, [i], None, [image_size], [0, 256]) for i in range(C)]
        histogram = np.concat(histogram, axis=0)
        return histogram


if __name__ == '__main__':
    rel_path = '../Datasets/qsd1_w1'
    db = Dataset(rel_path, color_space='rgb')
    print(f'Dataset length: {len(db)}')

    abs_path = '/home/adriangt2001/MCVC/C1/Project/Datasets/qsd1_w1'
    db = Dataset(abs_path, color_space='hsv')
    print(f'Dataset length: {len(db)}')
    
    compr_abs_path = '~/MCVC/C1/Project/Datasets/qsd1_w1'
    db = Dataset(compr_abs_path, color_space='hsv')
    print(f'Dataset length: {len(db)}')
