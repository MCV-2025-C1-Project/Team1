
import database
import numpy as np
import glob
import cv2
import os
import argparse
import yaml
import itertools
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str)
    
    parser.add_argument('--database', type=str)
    parser.add_argument('--dataset', type=str)

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

def find_match(qs: list[np.ndarray], db: database.Database, cfg: itertools.product[tuple]):
    pass

def main():
    args = parse_config_file(parse_args())

    db = database.Database(args.database)
    qs = load_dataset(args.dataset)

    if args.mode == 'search':
        search_record = []
        combos = itertools.product([])

        for cfg in tqdm(combos):
            find_match(qs, db, cfg)
            pass
    
        # Compute MAP@K
    elif args.mode == 'eval':
        
        pass

    pass

if __name__ == '__main__':
    main()
