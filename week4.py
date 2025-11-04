
import argparse
import glob
import itertools
import os

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import database2
from keypoints_descriptors import generate_descriptor

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        return None

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

    parser.add_argument('--output_file', type=str)

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
        images.append(img)
    return images

def find_match(qs: list[np.ndarray], db: database2.Database, cfg: itertools.product[tuple]):
    db.change_params(cfg['kp_descriptor'], cfg, autoprocess=True)
    for img in qs:
        kp, desc = generate_descriptor(img, cfg[0])
        matches = db.get_similar()
    pass

def main():
    args = parse_config_file(parse_args())

    db = database2.Database(args.database)
    qs = load_dataset(args.dataset)


    if args.mode == 'search':
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
        
        # Prepare SIFT parameters
        combo_sift = list(product_dict(**sift_params))

        # Prepare ORB parameters
        combo_orb = list(product_dict(**orb_params))

        # Prepare Color SIFT parameters
        combo_csift = list(product_dict(**csift_params))

        combos = combo_sift + combo_orb + combo_csift

        for cfg in tqdm(combos):
            find_match(qs, db, cfg)
    
        # Compute MAP@K
    elif args.mode == 'eval':
        
        pass

    pass

if __name__ == '__main__':
    main()
