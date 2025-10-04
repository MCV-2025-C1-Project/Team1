import argparse
import glob
import os
import pickle
import textwrap

import cv2
import numpy as np
import pandas as pd
import yaml

import constants
import metrics.average_precision as average_precision
from database import Database


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
              python myscript.py --k 1 5 10 \\
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
                        help='Config .yaml to have a preset of arguments.')

    parser.add_argument('--database_path', type=str,
                        help='Path to the database directory (Images ).')
    parser.add_argument('--query_path', type=str,
                        help='Path to the query image directory.')
    parser.add_argument('--k', nargs='+', type=int,
                        help='List of K values for top-K retrieval (e.g. --k 1 5 10)')
    parser.add_argument('--color_space', nargs='+', type=str, default='lab',
                        help='Name of the color space: rgb, hsv, gray_scale, lab')
    parser.add_argument('--bins', type=int, nargs='+', default=64,
                        help='Number of bins for the histogram per channel')
    parser.add_argument('--distances', nargs='+', type=str, default=['hist_intersection'],
                        help='List of similarity metrics to use (default: hist_intersection).')
    parser.add_argument('--val', type=bool, default=False,
                        help='Boolean to determine whether to do validation')

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
    color_spaces = args.color_space if isinstance(args.color_space, list) else [args.color_space]
    bins = args.bins if isinstance(args.bins, list) else [args.bins]
    distances = args.distances
    k_list = args.k
    val = args.val

    # Prepare data structures and load groundtruth for grid search in case of validation
    if val:
        best_config = [[] for _ in range(len(k_list))]
        best_result = [[] for _ in range(len(k_list))]
        best_mapk = [0 for _ in range(len(k_list))]
        grid_search_df = pd.DataFrame(columns=['color_space', 'bins', 'distances', *[f'mapk{k}' for k in k_list]])
        
        with open(os.path.join(query_path, 'gt_corresps.pkl'), 'rb') as f:
            groundtruth = pickle.load(f)
    
    # Load data
    db = Database(database_path, bins=bins[0], color_space=color_spaces[0])

    query_abs_path = os.path.abspath(os.path.expanduser(query_path))
    query_pattern = os.path.join(query_abs_path, '*.jpg')
    query_images = []

    for image_path in sorted(glob.glob(query_pattern, root_dir=query_abs_path)):
        image = cv2.imread(image_path)
        query_images.append(image)
    
    # Loop through all arguments (GridSearch)
    for cs in color_spaces:
        db.change_color(cs)
        for single_bin in bins:
            db.change_bins(single_bin)
            for dist in distances:
                results = [[] for _ in k_list]
                for image in query_images:
                    image = cv2.cvtColor(image, constants.CV2_CVT_COLORS[cs])

                    if cs == 'lab_processed':
                        l, a, b = cv2.split(image)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l_eq = clahe.apply(l)
                        image = cv2.merge((l_eq, a, b))

                    # Compute query histogram (image descriptor)
                    if image.ndim == 2:
                        hist = cv2.calcHist([image], [0], None, [single_bin], [0, 256]).ravel()
                    else:
                        H, W, C = image.shape
                        hists = [cv2.calcHist([image], [i], None, [single_bin], [0, 256]).ravel() for i in range(C)]
                        hist = np.concatenate(hists, axis=0)
                    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

                    for i, k in enumerate(k_list):
                        k_info = db.get_top_k_similar_images(hist, dist, k = k)
                        results[i].append(k_info)
                
                if val:
                    mapk_list = []
                    for idx, (result, k) in enumerate(zip(results, k_list)):
                        mapk = average_precision.mapk(groundtruth, result, k = k)
                        mapk_list.append(mapk)
                        print(f"MAP@{k}: {mapk:.4f}")

                        if mapk > best_mapk[idx]:
                            best_config[idx] = [cs, single_bin, dist]
                            best_result[idx] = result
                            best_mapk[idx] = mapk

                grid_search_df.loc[len(grid_search_df)] = [cs, single_bin, dist, *mapk_list]
                for result, k in zip(results, k_list):
                    print(f"\nFor k = {k}, the most similar images from the dataset to the queries are:\n")
                    print(result)
        
    if val:
        grid_search_df.to_csv('grid_search_results.csv')
        for config, results, mapk, k in zip(best_config, best_result, best_mapk, k_list):
            print(f"\nFor K = {k}\n")
            print(f"Best results: {results}")
            print(f"Best config: {config}")
            print(f"Best mapk: {mapk:.4f}")

                

if __name__ == "__main__":
    main()
                    

