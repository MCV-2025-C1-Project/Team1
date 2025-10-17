import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import descriptors.preprocessing as preprocessing
import background.w2_mask as w2_mask

dir_path = 'datasets/qsd2_w1'
use_micro = True
debug = False

recalls_by_error = []
precisions_by_error = []
f1_by_error = []
preprocess_list = ['clahe', 'hist_eq']#, 'gamma', 'contrast', 'gaussian_blur', 'median_blur', 'bilateral','unsharp']
color_space = 'lab'

TP_total = FP_total = FN_total = 0

precision_total_list = []
recall_total_list = []
for preprocess in preprocess_list:
    precision_list = []
    recall_list = []
    f1_list = []
    for image_path in glob.iglob(os.path.join(dir_path, "*.jpg")):
        base, _ = os.path.splitext(image_path)
        image_raw = cv2.imread(image_path)
        mask_groundtruth = cv2.imread(base + '.png', cv2.IMREAD_GRAYSCALE)
        
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2Lab)
        if preprocess == 'clahe':
            image = preprocessing.clahe_preprocessing(image, color_space)
        elif preprocess == 'hist_eq':
            image = preprocessing.hist_eq(image, color_space)
        elif preprocess == 'gamma':
            image = preprocessing.gamma(image)
        elif preprocess == 'contrast':
            image = preprocessing.contrast(image)
        elif preprocess == 'gaussian_blur':
            image = preprocessing.gaussian_blur(image)
        elif preprocess == 'median_blur':
            image = preprocessing.median_blur(image)
        elif preprocess == 'bilateral':
            image = preprocessing.bilateral(image)
        elif preprocess == 'unsharp':
            image = preprocessing.unsharp(image)

        mask = w2_mask.get_mask(image, 'wiwi')

        t, l, b, r = w2_mask.largest_axis_aligned_rectangle(mask)
        
        result = np.zeros_like(mask)
        result[t:b+1, l:r+1] = 255
        
        image_name = os.path.basename(base)
        folder = 'results'

        cv2.imwrite(os.path.join(folder, f"{image_name}_img.jpg"), image_raw)
        cv2.imwrite(os.path.join(folder, f"{image_name}_output_mask.png"), result)
        cv2.imwrite(os.path.join(folder, f"{image_name}_annotation.png"), mask_groundtruth)

        mask_groundtruth = (mask_groundtruth > 127).astype(np.uint8)
        result = (result > 127).astype(np.uint8)

        TP = np.sum((mask_groundtruth == 1) & (result == 1))
        FP = np.sum((mask_groundtruth == 0) & (result == 1))
        FN = np.sum((mask_groundtruth == 1) & (result == 0))

        precision_i = TP / (TP + FP + 1e-8)
        recall_i = TP / (TP + FN + 1e-8)
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i + 1e-8)

        print(f'Image: {image_path}')
        print(f'Precision: {precision_i}')
        print(f'Recall: {recall_i}')
        print(f'F1: {f1_i}')

        if precision_i < 0.5:
            print(f'{image_path} has less than 50 precision: {precision_i}') 
        if recall_i < 0.5:
            print(f'{image_path} has less than 50 recall: {recall_i}')
        if f1_i < 0.5:
            print(f'{image_path} has less than 50 f1: {f1_i}') 
        precision_list.append(precision_i)
        recall_list.append(recall_i)
        f1_list.append(f1_i)

        TP_total += TP
        FP_total += FP
        FN_total += FN
    
    if use_micro:
        precision = TP_total / (TP_total + FP_total + 1e-8)
        recall = TP_total / (TP_total + FN_total + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        precision = float(np.mean(precision_list))
        recall = float(np.mean(recall_list))
        f1 = float(np.mean(f1_list))
    
    precision_total_list.append(precision)
    recall_total_list.append(recall)

    print(f"\nTotal\n")
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')

fig, axs = plt.subplots(nrows=1, ncols=2, sharex='row')
axs[0] = plt.bar(preprocess_list, precision_total_list)
axs[1] = plt.bar(preprocess_list, precision_total_list)

fig.tight_layout()
fig.savefig('results/mask_by_preprocess.png')