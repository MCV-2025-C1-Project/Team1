
import argparse
import glob
import itertools
import os
import pickle
import time

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import constants
import database
import operations.mask as mask
from metrics import average_precision
from operations import histograms, preprocessing


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
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        help='Path to a YAML config file (loads presets).')
    
    parser.add_argument('--database_path', type=str,
                        help='Path to the database directory (images).')
    parser.add_argument('--query_path', type=str,
                        help='Path to the query image directory.')
    
    parser.add_argument('--color_space_list', nargs='+', type=str, default=['lab'],
                        help='Color space(s) to use. Options: gray_scale, rgb, hsv, lab, ycbcr. (Default: [lab])')
    parser.add_argument('--preprocesses_list', nargs='+', type=str, default=[None],
                        help="""Preprocessing methods applied to both DB and queries. Options: clahe, hist_eq, gamma, contrast, gaussian_blur, median_blur, bilateral, unsharp, None. (Default: [None])\n
                                If you want to add an option of no preprocessing to a list, add either an invalid argument (such as None) to the list or an empty string
                                in case of using a yaml file. Example terminal: [None, l1]. Example yaml: ['', euclidean]""")
    parser.add_argument('--masking', type=bool, default=False,
                        help='Whether to remove the background by getting a mask.')
    
    parser.add_argument('--bins_list', type=int, nargs='+', default=[64],
                        help='Number of histogram bins per channel. (Default: [64])')
    parser.add_argument('--blocks_list', nargs='+', type=int, default=[1],
                        help="Number of blocks to divide the image into")
    parser.add_argument('--hist_dims_list', nargs='+', type=int, default=[1],
                        help='Number of dimensions of the histogram')
    
    parser.add_argument('--distances_list', nargs='+', type=str, default=['hist_intersection'],
                        help='Distance / similarity metrics. Options: euclidean, l1, x2, hist_intersection, hellinger, canberra. (Default: [hist_intersection])')
    
    parser.add_argument('--k_list', nargs='+', type=int, default=[1],
                        help='List of K values for top-K retrieval (e.g. --k 1 5 10)')
    parser.add_argument('--val', type=bool, default=False,
                        help='Whether to run validation (expects gt_corresps.pkl in query directory). Options: True / False. (Default: False)')
    parser.add_argument('--pickle_filename', type=str, default=None,
                        help='Boolean to determine whether to save results in a pickle.')
    
    tmp_args = parser.parse_args()
    yaml_args = parse_yaml(tmp_args.config)
    if yaml_args:
        parser.set_defaults(**yaml_args)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    db_path = args.database_path
    query_path = args.query_path

    color_spaces_list = args.color_space_list
    preprocesses_list = args.preprocesses_list
    masking = args.masking

    bins_list = args.bins_list
    blocks_list = args.blocks_list
    hist_dims_list = args.hist_dims_list

    distances_list = args.distances_list

    k_list = args.k_list
    val = args.val
    output_pickle = args.pickle_filename

    color_space = color_spaces_list[0]
    preprocess = preprocesses_list[0]

    if val:
        best_config = [[] for _ in range(len(k_list))]
        best_result = [[] for _ in range(len(k_list))]
        best_mapk = [0 for _ in range(len(k_list))]
        grid_search_df = pd.DataFrame(columns=['color_space', 'preprocess', 'bins', 'blocks', 'hist_dim', 'distances', *[f'mapk{k}' for k in k_list]])
        
        with open(os.path.join(query_path, 'gt_corresps.pkl'), 'rb') as f:
            groundtruth = pickle.load(f)
    
    db = database.Database(db_path, color_space=color_space, preprocess=preprocess)

    query_abs_path = os.path.abspath(query_path)
    query_pattern = os.path.join(query_abs_path, '*.jpg')
    query_list = []

    for image_path in sorted(glob.glob(query_pattern, root_dir=query_abs_path)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, constants.CV2_CVT_COLORS[color_spaces_list[0]])
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
        
        if masking:
            mask_frame = mask.get_mask(image, 'none')
            top, left, bottom, right = mask.largest_axis_aligned_rectangle(mask_frame)
            image = image[top:bottom+1, left:right+1, ...]
        query_list.append(image)
    
    num_tests = len(bins_list) * len(blocks_list) * len(hist_dims_list) * len(distances_list)
    with tqdm(total=num_tests) as pbar:
        for bins, blocks, hist_dims in itertools.product(bins_list, blocks_list, hist_dims_list):
            if (hist_dims == 2 and (bins > 64 or blocks > 16)) or (hist_dims == 3 and (bins > 32 or blocks > 8)):
                pbar.set_postfix({
                        "color_space": color_space,
                        "preprocess": preprocess,
                        "bins": bins,
                        "blocks": blocks,
                        "hist_dims": hist_dims
                        })
                pbar.update(len(distances_list))
                continue

            db.change_hist(bins, blocks, hist_dims)
            for distance in distances_list:
                results = [[] for _ in k_list]
                for idx, query in enumerate(query_list):
                    start_time = time.time()
                    hist = histograms.gen_hist(query, bins, blocks, hist_dims)
                    similar_images = db.get_top_k_similar_images(hist, distance)

                    for i, k in enumerate(k_list):
                        results[i].append(similar_images[:k])
                    end_time = time.time()
                    print(f"Elapsed time for image {end_time - start_time}")

                if val:
                    mapk_list = []
                    for idx, (result, k) in enumerate(zip(results, k_list)):
                        mapk = average_precision.mapk(groundtruth, result, k=k)
                        mapk_list.append(mapk)

                        if mapk > best_mapk[idx]:
                            best_config[idx] = [color_space, preprocess, bins, blocks, hist_dims, distance]
                            best_result[idx] = result
                            best_mapk[idx] = mapk
                    
                    grid_search_df.loc[len(grid_search_df)] = [color_space, preprocess, bins, blocks, hist_dims, distance, *mapk_list]
                    pbar.set_postfix({
                        "color_space": color_space,
                        "preprocess": preprocess,
                        "bins": bins,
                        "blocks": blocks,
                        "hist_dims": hist_dims,
                        "distance": distance,
                        **{f"MAP@{k}": mapk_list[idx] for idx, k in enumerate(k_list)}
                        })

                pbar.update(1)

    if val:
        os.makedirs('./results', exist_ok=True)
        grid_search_df.to_csv('results/grid_search_results_w2.csv')
        for config, results, mapk, k in zip(best_config, best_result, best_mapk, k_list):
            print(f"\nFor K = {k}\n")
            print(f"Best results: {results}")
            print(f"Best config: {config}")
            print(f"Best mapk: {mapk:.4f}")

    if output_pickle:
        for result in results:
            with open(output_pickle, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
                    
