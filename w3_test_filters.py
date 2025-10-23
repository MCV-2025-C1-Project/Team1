import argparse
import glob
import os
import textwrap
import numpy as np
import pywt

import cv2
import yaml
from tqdm import tqdm

import descriptors.filters as filters
import metrics.image_quality as quality

import pandas as pd


def parse_yaml(config):
    if not config: return None

    yaml_args = {}
    with open(config, 'r') as f:
        contents: dict = yaml.safe_load(f)
        for _, params in contents.items():
            if params:
                for key, value in params.items():
                    yaml_args[key] = value
    
    return yaml_args

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image retrieval evaluation.",
        epilog=textwrap.dedent('''\
            Example usage:
              python week2.py --k 1 5 10 \\
                                 --database_path ./data/db \\
                                 --query_path ./data/queries \\
                                 --metrics hist_intersection \\
                                 --color_space lab \\
                                 --bins 64 \\
                                 --val True \\

            Notes:
              - You can specify multiple K values using space-separated integers.
              - Metrics can include: hist_intersection, euclidean, etc. Check metrics/distances.py file to see all the distance metrics.
              - Both absolute and relative paths should work.
              - For validation, the gt is expected to be in the same dir as the queries in a pickle format.
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--config', type=str,
                        help='Path to a YAML config file (loads presets).')
    
    parser.add_argument('--data_path', type=str,
                        help='Path to the database directory (images).')
    parser.add_argument('--groundtruth_path', type=str,
                        help='Path to the query image directory.')
    
    # Median
    parser.add_argument('--median_kernel_size', nargs='+', type=int)

    # Gaussian
    parser.add_argument('--gaussian_kernel_size', nargs='+', type=int)
    parser.add_argument('--gaussian_sigma', nargs='+', type=float)

    # Bilateral
    parser.add_argument('--bilateral_kernel_size', nargs='+', type=int)
    parser.add_argument('--bilateral_sigma', nargs='+', type=float)

    # Nlm
    parser.add_argument('--nlm_h', nargs='+', type=int)
    parser.add_argument('--nlm_hcolor', nargs='+', type=int)
    parser.add_argument('--nlm_template_window_size', nargs='+', type=int)
    parser.add_argument('--nlm_search_window_size', nargs='+', type=int)

    # Wavelets
    parser.add_argument('--wavelet', nargs='+', type=str)
    parser.add_argument('--wavelet_threshold', nargs='+', type=float)
    parser.add_argument('--wavelet_mode', nargs='+', type=str)
    
    tmp_args = parser.parse_args()
    yaml_args = parse_yaml(tmp_args.config)
    if yaml_args:
        parser.set_defaults(**yaml_args)

    args = parser.parse_args()

    return args

def denoise_with_metrics(dataset_list, gt_list, mode, params):
    mse = 0
    psnr = 0
    ssim = 0
    for input_image, gt_image in tqdm(zip(dataset_list, gt_list), desc=f'Checking mode {mode}', unit='Images', total=len(dataset_list)):
        output_image = filters.denoise_image(input_image, mode, **params)
        metrics_tuple = quality.compute_metrics(output_image, gt_image)
        mse += metrics_tuple[0]
        psnr += metrics_tuple[1]
        ssim += metrics_tuple[2]

    mse /= len(dataset_list)
    psnr /= len(dataset_list)
    ssim /= len(dataset_list)

    return mse, psnr, ssim

def main():
    args = parse_args()

    data_path = args.data_path
    gt_path = args.groundtruth_path

    # Median
    median_kernel_size = args.median_kernel_size

    # Gaussian
    gaussian_kernel_size = args.gaussian_kernel_size
    gaussian_sigma = args.gaussian_sigma

    # Bilateral
    bilateral_kernel_size = args.bilateral_kernel_size
    bilateral_sigma = args.bilateral_sigma

    # Nlm
    nlm_h = args.nlm_h
    nlm_hcolor = args.nlm_hcolor
    nlm_template_window_size = args.nlm_template_window_size
    nlm_search_window_size = args.nlm_search_window_size

    # Wavelets
    wavelet_families = args.wavelet
    wavelet_threshold = args.wavelet_threshold
    wavelet_mode = args.wavelet_mode

    # Load dataset
    dataset_abs_path = os.path.abspath(data_path)
    dataset_pattern = os.path.join(dataset_abs_path, '*.jpg')
    dataset_list = []

    for image_path in tqdm(sorted(glob.glob(dataset_pattern, root_dir=dataset_abs_path)), desc=f'Loading images...', unit='Images'):
        image = cv2.imread(image_path)
        dataset_list.append(image)

    # Load groundtruth
    gt_abs_path = os.path.abspath(gt_path)
    gt_pattern = os.path.join(gt_abs_path, '*.jpg')
    gt_list = []

    for image_path in tqdm(sorted(glob.glob(gt_pattern, root_dir=gt_abs_path)), desc=f'Loading images...', unit='Images', total=len(dataset_list)):
        image = cv2.imread(image_path)
        gt_list.append(image)
    
    # Get metrics results
    results_mse = []
    results_psnr = []
    results_ssim = []

    for mode in filters.METHODS:
        if mode == 'median':
            best_mse = (None, np.inf, 0, 0)
            best_psnr = (None, np.inf, 0, 0)
            best_ssim = (None, np.inf, 0, 0)

            for ksize in median_kernel_size:
                params = {'kernel_size': ksize}

                mse, psnr, ssim = denoise_with_metrics(dataset_list, gt_list, mode, params)

                if mse < best_mse[1]:
                    best_mse = (params, mse, psnr, ssim)
                if psnr > best_psnr[2]:
                    best_psnr = (params, mse, psnr, ssim)
                if ssim > best_ssim[3]:
                    best_ssim = (params, mse, psnr, ssim)

        elif mode == 'gaussian':
            best_mse = (None, np.inf, 0, 0)
            best_psnr = (None, np.inf, 0, 0)
            best_ssim = (None, np.inf, 0, 0)

            for ksize in gaussian_kernel_size:
                for sigma in gaussian_sigma:
                    params = {'kernel_size': ksize, 'sigma': sigma}

                    mse, psnr, ssim = denoise_with_metrics(dataset_list, gt_list, mode, params)

                    if mse < best_mse[1]:
                        best_mse = (params, mse, psnr, ssim)
                    if psnr > best_psnr[2]:
                        best_psnr = (params, mse, psnr, ssim)
                    if ssim > best_ssim[3]:
                        best_ssim = (params, mse, psnr, ssim)
        
        elif mode == 'bilateral':
            best_mse = (None, np.inf, 0, 0)
            best_psnr = (None, np.inf, 0, 0)
            best_ssim = (None, np.inf, 0, 0)

            for ksize in bilateral_kernel_size:
                for sigma in bilateral_sigma:
                    params = {'kernel_size': ksize, 'sigma': sigma}

                    mse, psnr, ssim = denoise_with_metrics(dataset_list, gt_list, mode, params)

                    if mse < best_mse[1]:
                        best_mse = (params, mse, psnr, ssim)
                    if psnr > best_psnr[2]:
                        best_psnr = (params, mse, psnr, ssim)
                    if ssim > best_ssim[3]:
                        best_ssim = (params, mse, psnr, ssim)
        
        elif mode == 'nlm':
            best_mse = (None, np.inf, 0, 0)
            best_psnr = (None, np.inf, 0, 0)
            best_ssim = (None, np.inf, 0, 0)

            for h in nlm_h:
                for hcolor in nlm_hcolor:
                    for template_window_size in nlm_template_window_size:
                        for search_window_size in nlm_search_window_size:
                            params = {'h': h, 'hcolor': hcolor, 'template_window_size': template_window_size, 'search_window_size': search_window_size}
                    
                            mse, psnr, ssim = denoise_with_metrics(dataset_list, gt_list, mode, params)

                            if mse < best_mse[1]:
                                best_mse = (params, mse, psnr, ssim)
                            if psnr > best_psnr[2]:
                                best_psnr = (params, mse, psnr, ssim)
                            if ssim > best_ssim[3]:
                                best_ssim = (params, mse, psnr, ssim)

        elif mode == 'wavelets':
            best_mse = (None, np.inf, 0, 0)
            best_psnr = (None, np.inf, 0, 0)
            best_ssim = (None, np.inf, 0, 0)

            for wavelet_family in wavelet_families:
                for wavelet in pywt.wavelist(family=wavelet_family, kind='discrete'):
                    for threshold in wavelet_threshold:
                        for threshold_mode in wavelet_mode:
                            params = {'wavelet': wavelet, 'threshold': threshold, 'threshold_mode': threshold_mode}

                            mse, psnr, ssim = denoise_with_metrics(dataset_list, gt_list, mode, params)

                            if mse < best_mse[1]:
                                best_mse = (params, mse, psnr, ssim)
                            if psnr > best_psnr[2]:
                                best_psnr = (params, mse, psnr, ssim)
                            if ssim > best_ssim[3]:
                                best_ssim = (params, mse, psnr, ssim)
        
        results_mse.append({
            **best_mse,
            'mse': best_mse[1],
            'psnr': best_mse[2],
            'ssim': best_mse[3],
        })
        results_psnr.append({
            **best_psnr,
            'mse': best_psnr[1],
            'psnr': best_psnr[2],
            'ssim': best_psnr[3],
        })
        results_ssim.append({
            **best_ssim,
            'mse': best_ssim[1],
            'psnr': best_ssim[2],
            'ssim': best_ssim[3],
        })
    
    results_mse = pd.DataFrame(results_mse)
    results_psnr = pd.DataFrame(results_psnr)
    results_ssim = pd.DataFrame(results_ssim)
    results = pd.concat([results_mse, results_psnr, results_ssim])
    results.to_csv('results/w3/results_filters.csv')

if __name__ == '__main__':
    main()