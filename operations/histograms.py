
import cv2
import numpy as np


def gen_1d_hist(image, hierarchy_levels: int, bins: int):
    H, W, C = image.shape
    hists = []

    for level in range(hierarchy_levels):
        blocks = 2 ** level
        divided_image = np.reshape(image, [-1, H // blocks, W // blocks, C])
        block_hists = []
        for image_block in divided_image:
            hist = [cv2.calcHist(image_block, [i], None, [bins], [0, 256]).ravel() for i in range(C)]
            hist = np.concatenate(hist)
            block_hists.append(hist)
        block_hists = np.stack(block_hists)
        hists.append(block_hists)
    
    return hists

def gen_2d_hist(image, hierarchy_levels: int, bins: int):
    H, W, C = image.shape
    hists = []

    for level in range(hierarchy_levels):
        blocks = 2 ** level
        divided_image = np.reshape(image, [-1, H // blocks, W // blocks, C])
        block_hists = []
        for image_block in divided_image:
            hist = [cv2.calcHist([image_block[..., c1], image_block[..., c2]], [0, 1], None, [bins, bins], [0, 256, 0, 256]) for c1 in range(C) for c2 in range(c1+1, C)]
            hist = np.stack(hist)
            block_hists.append(hist)
        block_hists = np.stack(block_hists)
        hists.append(block_hists)
    return hists



def gen_3d_hist(image, hierarchy_levels: int, bins: int):
    H, W, C = image.shape
    hists = []

    for level in range(hierarchy_levels):
        blocks = 2 ** level
        divided_image = np.reshape(image, [-1, H // blocks, W // blocks, C])
        block_hists = []
        for image_block in divided_image:
            hist = cv2.calcHist([image_block], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            block_hists.append(hist)
        block_hists = np.stack(block_hists)
        hists.append(block_hists)
    return hists


if __name__ == '__main__':
    image = cv2.imread('datasets/BBDD/bbdd_00000.jpg')
    hierarchy_levels = 3
    bins = 32
    hist = gen_1d_hist(image, hierarchy_levels, bins)
    print('Yay1!')
    hist = gen_2d_hist(image, hierarchy_levels, bins)
    print('Yay1!')
    hist = gen_3d_hist(image, hierarchy_levels, bins)
    print('Yay1!')