import argparse
import glob
import os
import pickle
import textwrap

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import constants
import metrics.average_precision as average_precision
from database import Database
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
    """
    Parse the command-line arguments for the image retrieval evaluation program.

    This function defines and parses command-line arguments required to run an
    image retrieval evaluation. It uses the `argparse` module to specify input
    parameters such as retrieval depths (K values), dataset paths, similarity
    metrics, and validation mode.

    The help message (`-h` or `--help`) displays detailed usage instructions,
    examples, and notes about valid options and expected directory structures.
    """
    parser = argparse.ArgumentParser(
        description="Run image retrieval evaluation.",
        epilog=textwrap.dedent('''\
            Example usage:
              python week1.py --k 1 5 10 \\
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
    
    parser.add_argument('--database_path', type=str,
                        help='Path to the database directory (images).')
    parser.add_argument('--query_path', type=str,
                        help='Path to the query image directory.')
    
    parser.add_argument('--k', nargs='+', type=int,
                        help='List of K values for top-K retrieval (e.g. --k 1 5 10)')
    parser.add_argument('--color_space', nargs='+', type=str, default=['lab'],
                        help='Color space(s) to use. Options: gray_scale, rgb, hsv, lab, ycbcr. (Default: [lab])')
    parser.add_argument('--bins', type=int, nargs='+', default=[64],
                        help='Number of histogram bins per channel. (Default: [64])')
    parser.add_argument('--distances', nargs='+', type=str, default=['hist_intersection'],
                        help='Distance / similarity metrics. Options: euclidean, l1, x2, hist_intersection, hellinger, canberra. (Default: [hist_intersection])')
    parser.add_argument('--preprocessings', nargs='+', type=str, default=[None],
                        help="""Preprocessing methods applied to both DB and queries. Options: clahe, hist_eq, gamma, contrast, gaussian_blur, median_blur, bilateral, unsharp, None. (Default: [None])\n
                                If you want to add an option of no preprocessing to a list, add either an invalid argument (such as None) to the list or an empty string
                                in case of using a yaml file. Example terminal: [None, l1]. Example yaml: ['', euclidean]""")
    parser.add_argument('--hier_levels', nargs='+', type=int, default=[1],
                        help='Number of divisions for the histogram')
    parser.add_argument('--hist_dims', nargs='+', type=int, default=[1],
                        help='Number of dimensions of the histogram')
    parser.add_argument('--val', type=bool, default=False,
                        help='Whether to run validation (expects gt_corresps.pkl in query directory). Options: True / False. (Default: False)')
    parser.add_argument('--hierarchy', nargs='+', type=bool, default=[False],
                        help='Whether to hierarchize the histogram or not')
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

    # Load arguments into variables
    database_path = args.database_path
    query_path = args.query_path
    k_list = args.k
    color_spaces = args.color_space if isinstance(args.color_space, list) else [args.color_space]
    bins = args.bins if isinstance(args.bins, list) else [args.bins]
    distances = args.distances
    preprocessings = args.preprocessings
    hier_levels = args.hier_levels
    hist_dims = args.hist_dims
    hierarchy = args.hierarchy
    val = args.val
    output_pickle = args.pickle_filename

    # Prepare data structures and load groundtruth for grid search in case of validation
    if val:
        best_config = [[] for _ in range(len(k_list))]
        best_result = [[] for _ in range(len(k_list))]
        best_mapk = [0 for _ in range(len(k_list))]
        grid_search_df = pd.DataFrame(columns=['color_space', 'preprocess', 'bins', 'distances', *[f'mapk{k}' for k in k_list]])
        
        with open(os.path.join(query_path, 'gt_corresps.pkl'), 'rb') as f:
            groundtruth = pickle.load(f)
    
    # Load data
    db = Database(database_path)

    query_abs_path = os.path.abspath(os.path.expanduser(query_path))
    query_pattern = os.path.join(query_abs_path, '*.jpg')
    query_images = []

    for image_path in sorted(glob.glob(query_pattern, root_dir=query_abs_path)):
        image = cv2.imread(image_path)
        query_images.append(image)
    
    # Loop through all arguments (GridSearch)
    num_tests = len(color_spaces) * len(preprocessings) * len(hierarchy) * len(hier_levels) * len(hist_dims) * len(bins) * len(distances) * len(query_images)
    with tqdm(total=num_tests) as pbar:
        for cs in color_spaces:
            if cs not in list(constants.CV2_CVT_COLORS.keys()): continue
            for preprocess in preprocessings:
                db.change_color(cs, preprocess)
                for is_hierarchy in hierarchy:
                    for level in hier_levels:
                        for hist_dim in hist_dims:
                            for single_bin in bins:
                                db.change_hist(level, hist_dim, single_bin)
                                for dist in distances:
                                    results = [[] for _ in k_list]
                                    for image in query_images:
                                        image = cv2.cvtColor(image, constants.CV2_CVT_COLORS[cs])

                                        if preprocess == 'clahe':
                                            image = preprocessing.clahe_preprocessing(image, cs)
                                        elif preprocess == 'hist_eq':
                                            image = preprocessing.hist_eq(image, cs)
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

                                        hist = histograms.gen_hist(image, level, single_bin, hist_dim, is_hierarchy)                         

                                        for i, k in enumerate(k_list):
                                            k_info = db.get_top_k_similar_images(hist, dist, k = k)
                                            results[i].append(k_info)
                                    
                                    if val:
                                        mapk_list = []
                                        for idx, (result, k) in enumerate(zip(results, k_list)):
                                            mapk = average_precision.mapk(groundtruth, result, k=k)
                                            mapk_list.append(mapk)
                                            # print(f"MAP@{k}: {mapk:.4f}")

                                            if mapk > best_mapk[idx]:
                                                best_config[idx] = [cs, preprocess, single_bin, dist]
                                                best_result[idx] = result
                                                best_mapk[idx] = mapk

                                        grid_search_df.loc[len(grid_search_df)] = [cs, preprocess, single_bin, dist, *mapk_list]
                                    
                                        pbar.set_postfix({f"BestMAP@{k}": best_mapk[idx] for idx, k in enumerate(k_list)})
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
                    

