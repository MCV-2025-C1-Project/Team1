
import argparse
import glob
import itertools
import os
import pickle

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import database2
import mask_creation_w3_main
from keypoints_descriptors import generate_descriptor
from metrics import average_precision


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        return None

def check_args(args: argparse.Namespace):
    if args.mode == 'search':
        if not args.output_csv: raise ValueError(f"For mode {args.mode} argument --output_csv must be specified.")
    elif args.mode == 'eval':
        if not args.output_pkl: raise ValueError(f"For mode {args.mode} argument --output_pkl must be specified.")
    else:
        raise ValueError(f"{args.mode} not valid for --mode argument. Choose between 'search' and 'eval'.")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    
    parser.add_argument('--database', type=str)
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--kp_descriptor', nargs='+', type=str)
    parser.add_argument('--n_features', nargs='+', type=int)
    parser.add_argument('--edge_threshold', nargs='+', type=int)

    # SIFT
    parser.add_argument('--n_octave_layers', nargs='+', type=int)
    parser.add_argument('--contrast_threshold', nargs='+', type=float)
    parser.add_argument('--sigma', nargs='+', type=float)

    # ORB
    parser.add_argument('--scale_factor', nargs='+', type=float)
    parser.add_argument('--n_levels', nargs='+', type=int)
    parser.add_argument('--first_level', nargs='+', type=int)
    parser.add_argument('--WTA_K', nargs='+', type=int)
    parser.add_argument('--score_type', nargs='+', type=int)
    parser.add_argument('--patch_size', nargs='+', type=int)
    parser.add_argument('--fast_threshold', nargs='+', type=int)

    # Color SIFT
    parser.add_argument('--sift_mode', nargs='+', type=str)
    parser.add_argument('--use_rootsift', nargs='+', type=str2bool)

    parser.add_argument('-k', nargs='+', type=int)

    parser.add_argument('--mode', type=str, choices=['search', 'eval'])

    parser.add_argument('--output_pkl', type=str)
    parser.add_argument('--output_csv', type=str)

    return parser.parse_args()

def parse_config_file(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config: return args
    
    cfg = yaml.safe_load(args.config)
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value
    return cfg

def load_dataset(path: str) -> list[np.ndarray]:
    images = []
    pattern = os.path.join(path, '*.jpg')
    file_list = sorted(glob.glob(pattern))
    for file in file_list:
        f = os.path.join(path, file)
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3, None)
        images.append(img)
    return images

def mask_dataset(path: str, qs: list[np.ndarray]) -> list[np.ndarray]:
    qs_masked = []
    mask_list, quads_list = mask_creation_w3_main.process_dataset(path)
    for idx, (extracted_masks, extracted_quads) in enumerate(zip(mask_list, quads_list)):
        subimages = []
        for mask, quads in zip(extracted_masks, extracted_quads):
            image = qs[idx]

            quads = np.stack(quads, axis=0)
            xmin, ymin = quads.min(axis=0).astype(int)
            xmax, ymax = quads.max(axis=0).astype(int) + 1

            image = image[ymin:ymax, xmin:xmax]
            mask = mask[ymin:ymax, xmin:xmax]

            masked_image = np.where(mask[..., np.newaxis], image, np.zeros_like(image))

            newH, newW = ymax - ymin, xmax - xmin
            new_quads = quads - np.array([[xmin, ymin]])

            M, mask = cv2.findHomography(new_quads, np.array([[0, 0], [newW - 1, 0], [newW - 1, newH - 1], [0, newH - 1]]))
            transformed_image = cv2.warpPerspective(masked_image, M, (newW, newH))
            subimages.append(transformed_image)
        qs_masked.append(subimages)
    return qs_masked

def find_match(qs: list[np.ndarray], db: database2.Database, cfg: itertools.product[tuple], k_list: list[int]) -> list:
    db.change_params(cfg['kp_descriptor'], cfg, autoprocess=True)
    results = [[] for _ in k_list]
    for img in qs:
        kp, desc = generate_descriptor(img, cfg[0])
        matches = db.get_similar(kp, desc)
        
        for idx, k in enumerate(k_list):
            for match in matches:
                results[idx].append(match[:k])
    
    return results

def dataset_mapk(
    results: list, groundtruth: list,
    best_mapk: list, best_config: list, best_result: list,
    cfg: dict, k_list: list[int]
) -> tuple[list]:
    # Compute MAP@K
    mapk_list = []
    for idx, (r, k) in enumerate(zip(results, k_list)):
        mapk = average_precision.mapk(groundtruth, r, k=k, multi=True)
        mapk_list.append(mapk)
        if mapk >= best_mapk[idx]:
            best_config[idx] = cfg
            best_result[idx] = r
            best_mapk[idx] = mapk
    return mapk_list, best_config, best_result, best_mapk

def generate_combos(args: argparse.Namespace) -> list:
    sift_params = {
        'kp_descriptor': ['sift'],
        'n_features': args.n_features,
        'edge_threshold': args.edge_threshold,
        'n_octave_layers': args.n_octave_layers,
        'contrast_threshold': args.contrast_threshold,
        'sigma': args.sigma
    }
    orb_params = {
        'kp_descriptor': ['orb'],
        'n_features': args.n_features,
        'edge_threshold': args.edge_threshold,
        'scale_factor': args.scale_factor,
        'n_levels': args.n_levels,
        'WTA_K': args.WTA_K,
        'score_type': args.score_type,
        'patch_size': args.patch_size,
        'fast_threshold': args.fast_threshold
    }
    csift_params = {
        'kp_descriptor': ['color_sift'],
        'n_features': args.n_features,
        'edge_threshold': args.edge_threshold,
        'n_octave_layers': args.n_octave_layers,
        'contrast_threshold': args.contrast_threshold,
        'sigma': args.sigma,
        'sift_mode': args.sift_mode,
        'use_rootsift': args.use_rootsift
    }

    def product_dict(**kwargs):
        keys = kwargs.keys()
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(keys, instance))
    
    combo_sift = list(product_dict(**sift_params))
    combo_orb = list(product_dict(**orb_params))
    combo_csift = list(product_dict(**csift_params))

    combos = combo_sift + combo_orb + combo_csift

    return combos

def main():
    args = parse_config_file(parse_args())
    check_args(args)

    db = database2.Database(args.database)
    qs = load_dataset(args.dataset)
    qs = mask_dataset(args.dataset, qs)

    if args.mode == 'search':
        best_config = [[] for _ in range(len(args.k))]
        best_result = [[] for _ in range(len(args.k))]
        best_mapk = [0 for _ in range(len(args.k))]
        with open(os.path.join(args.dataset, 'gt_corresps.pkl'), 'rb') as f:
            groundtruth = pickle.load(f)

        combos = generate_combos(args)

        grid_search_df = pd.DataFrame(columns=list(combos[0].keys()) + [f'mapk{k}' for k in args.k])
        
        for cfg in tqdm(combos):
            results = find_match(qs, db, cfg, args.k)

            mapk_list, best_config, best_result, best_mapk = dataset_mapk(results, groundtruth,
                                                                          best_mapk, best_config, best_result,
                                                                          cfg, args.k)
            
            row = {
                **cfg,
                **{f'mapk{k}': mapk_list[i] for i, k in enumerate(args.k)}
            }
            grid_search_df.loc[len(grid_search_df)] = row

        grid_search_df.to_csv(args.output_csv)
        for config, results, mapk, k in zip(best_config, best_result, best_mapk, args.k):
            print(f"\nFor K = {k}\n")
            print(f"Best results: {results}")
            print(f"Best config: {config}")
            print(f"Best mapk: {mapk:.4f}")

    elif args.mode == 'eval':
        combos = generate_combos(args)
        cfg = combos[0]
        results = find_match(qs, db, cfg, args.k)

        for result in results:
            with open(args.output_pkl, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
