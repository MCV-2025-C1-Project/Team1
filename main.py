import argparse
import glob
import os
import pickle
import textwrap

import cv2
import numpy as np

import metrics.average_precision as average_precision
from databse import Database


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

    parser.add_argument('--k', nargs='+', type=int, required=True,
                        help='List of K values for top-K retrieval (e.g. --k 1 5 10)')
    parser.add_argument('--database_path', type=str, required=True,
                        help='Path to the database directory (Images ).')
    parser.add_argument('--query_path', type=str, required=True,
                        help='Path to the query image directory (can include wildcards).')
    parser.add_argument('--metrics', nargs='+', type=str, default=['hist_intersection'],
                        help='List of similarity metrics to use (default: hist_intersection).')
    parser.add_argument('--color_space', type=str, default='lab',
                        help='Name of the color space: rgb, hsv, gray_scale, lab')
    parser.add_argument('--bins', type=int, default=64,
                        help='Number of bins for the histogram per channel')
    parser.add_argument('--val', type=bool, default=False,
                        help='Boolean to determine whether to do validation')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    k_list = args.k

    k_list = args.k
    database_path = args.database_path
    query_path = args.query_path
    metrics = args.metrics
    val = args.val
    color_space = args.color_space
    bins = args.bins

    # Create Database object, this includes processing all images in database_path
    db = Database(database_path, bins = bins, debug = False, color_space='lab')

    root_dir = os.path.abspath(os.path.expanduser(database_path))

    results = [[] for _ in k_list]

    for image_path in glob.iglob(query_path, root_dir=root_dir):

        #Process query image
        image = cv2.imread(image_path)

        if color_space == 'rgb':
            cv2_cvt_code = cv2.COLOR_BGR2RGB           
            image = cv2.cvtColor(image, cv2_cvt_code)

        elif color_space == 'hsv':
            cv2_cvt_code = cv2.COLOR_BGR2HSV      
            image = cv2.cvtColor(image, cv2_cvt_code)

        elif color_space == 'gray_scale':
            cv2_cvt_code = cv2.COLOR_BGR2GRAY
            image = cv2.cvtColor(image, cv2_cvt_code)

        elif color_space == 'lab':
            cv2_cvt_code = cv2.COLOR_BGR2Lab
            image = cv2.cvtColor(image, cv2_cvt_code)
            l, a, b = cv2.split(image)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_eq = clahe.apply(l)
            image = cv2.merge((l_eq, a, b))

        #Calc histograms for gray_scale
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                hist = cv2.calcHist([image], [0], None, [bins], [0, 256]).ravel()
        #Calc histograms for 3 channel images
        else:
            hists = [
                cv2.calcHist([image], [i], None, [bins], [0, 256]).ravel()
                for i in range(image.shape[2])
            ]
            hist = np.concatenate(hists, axis=0)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

        #Compute distances given k
        for i, k in enumerate(k_list):
             
            k_info = db.get_top_k_similar_images(hist, metrics, k = k)
            results[i].append(k_info)

    # Compute MAP@k given a gt
    if val:
        directory = os.path.dirname(query_path)
        pickle_gt = os.path.join(directory, r'gt_corresps.pkl')
        with open(pickle_gt, "rb") as f:
            obj = pickle.load(f)

        for result, k in zip(results, k_list):
            map1 = average_precision.mapk(obj, result, k = k)
        
            print(f"MAP@{k}: {map1:.4f}")

    for result, k in zip(results, k_list):
        print(f'\nFor k = {k}, the most similar images from the dataset to de queries are:\n')
        print(result)
if __name__ == "__main__":
    main()