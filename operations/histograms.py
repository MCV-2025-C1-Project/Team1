
import cv2
import numpy as np

def divide_image(image, blocks: int):
    H, W, C = image.shape
    
    height_padding = blocks - (H % blocks)
    width_padding = blocks - (W % blocks)

    new_image = cv2.copyMakeBorder(image, 0, height_padding, 0, width_padding, cv2.BORDER_REFLECT)

    H, W, C = new_image.shape
    new_image = np.reshape(new_image, [blocks, H // blocks, blocks, W // blocks, C]).transpose([0, 2, 1, 3, 4]).reshape(-1, H // blocks, W // blocks, C)
    return new_image

def gen_hist_old(image, hierarchy_levels: int, bins: int, hist_dim: int, hierarchy: bool):
    H, W, C = image.shape
    hists = []

    if hierarchy:
        for level in range(hierarchy_levels):
            blocks = 2 ** level
            divided_image = divide_image(image, blocks)
            block_hists = []
            for image_block in divided_image:
                if hist_dim == 1:
                    hist = [cv2.calcHist([image_block], [i], None, [bins], [0, 256]).ravel() for i in range(C)]
                elif hist_dim == 2:
                    hist = [cv2.calcHist([image_block[..., c1], image_block[..., c2]], [0, 1], None, [bins, bins], [0, 256, 0, 256]) for c1 in range(C) for c2 in range(c1+1, C)]
                elif hist_dim == 3:
                    hist = [cv2.calcHist([image_block], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])]
                hist = np.stack(hist)
                block_hists.append(hist)
            block_hists = np.concatenate(block_hists)
            hists.append(block_hists)
    else:
        blocks = 2 ** (hierarchy_levels - 1)
        divided_image = divide_image(image, blocks)
        block_hists = []
        for image_block in divided_image:
            if hist_dim == 1:
                hist = [cv2.calcHist([image_block], [i], None, [bins], [0, 256]).ravel() for i in range(C)]
            elif hist_dim == 2:
                hist = [cv2.calcHist([image_block[..., c1], image_block[..., c2]], [0, 1], None, [bins, bins], [0, 256, 0, 256]) for c1 in range(C) for c2 in range(c1+1, C)]
            elif hist_dim == 3:
                hist = [cv2.calcHist([image_block], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])]
            hist = np.stack(hist)
            block_hists.append(hist)
        block_hists = np.concatenate(block_hists)
        hists.append(block_hists)

    hists = np.concatenate(hists)
    cv2.normalize(hists, hists, alpha=1.0, norm_type=cv2.NORM_L1)   
    return hists

def gen_hist(image, bins, num_windows, num_dimensions):
    H, W, C = image.shape
    windowed_image = divide_image(image, num_windows)
    hists = []

    for image_window in windowed_image:
        if num_dimensions == 1:
            hist = [cv2.calcHist([image_window], [i], None, [bins], [0, 256]).ravel() for i in range(C)]
        elif num_dimensions == 2:
            hist = [cv2.calcHist([image_window[..., c1], image_window[..., c2]], [0, 1], None, [bins, bins], [0, 256, 0, 256]) for c1 in range(C) for c2 in range(c1+1, C)]
        elif num_dimensions == 3:
            hist = [cv2.calcHist([image_window], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])]
        hist = np.stack(hist, axis=0).ravel()
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        hists.append(hist)
    hists = np.stack(hists, axis=0).ravel()
    return hists

def gen_hist_hierarchical(image, bins, num_windows, num_dimensions):
    H, W, C = image.shape
    windowed_image = divide_image(image, num_windows)
    hists = []

    for image_window in windowed_image:
        if num_dimensions == 1:
            hist = [cv2.calcHist([image_window], [i], None, [bins], [0, 256]).ravel() for i in range(C)]
        elif num_dimensions == 2:
            hist = [cv2.calcHist([image_window[..., c1], image_window[..., c2]], [0, 1], None, [bins, bins], [0, 256, 0, 256]) for c1 in range(C) for c2 in range(c1+1, C)]
        elif num_dimensions == 3:
            hist = [cv2.calcHist([image_window], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])]
        hist = np.stack(hist, axis=0).ravel()
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
        hists.append(hist)
    hists = np.stack(hists, axis=0).ravel()
    return hists

if __name__ == '__main__':
    import itertools
    image = cv2.imread('datasets/BBDD/bbdd_00000.jpg')
    bins = 32
    num_windows_list = [1, 2, 3, 4, 5]
    num_dimensions_list = [1, 2, 3]

    for num_windows, num_dimensions in itertools.product(num_windows_list, num_dimensions_list):
        hist = gen_hist(image, bins, num_windows, num_dimensions)
        print('Yay!')