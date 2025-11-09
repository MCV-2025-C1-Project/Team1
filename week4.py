
import argparse
import glob
import itertools
import os
import pickle

import cv2
import imageio
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import database2
import mask_creation_w3_main
from background.group2.bckg_rmv import remove_background_morphological_gradient
from background.group2.descriptors import preprocess_image
from background.group2.image_split import split_images
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

def parse_config_file(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    
    with open(args.config, "r") as f:
        file_cfg = yaml.safe_load(f) or {}

    # fusiona: CLI pisa a YAML
    merged = {**file_cfg, **{k: v for k, v in vars(args).items() if v is not None}}
    return argparse.Namespace(**merged)

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

    # AZAKE
    #parser.add_argument('--descriptor_type', nargs='+', type=int)
    parser.add_argument('--descriptor_size', nargs='+', type=int)
    parser.add_argument('--descriptor_channels', nargs='+', type=int)
    parser.add_argument('--threshold', nargs='+', type=float)
    parser.add_argument('--n_octaves', nargs='+', type=int)
    #parser.add_argument('--diffusivity', nargs='+', type=int)

    # Matchers
    parser.add_argument('--lowe_ratio', nargs='+', type=float)
    parser.add_argument('--min_good', nargs='+', type=int)
    parser.add_argument('--match_distance', nargs='+', type=str)
    parser.add_argument('--max_size', nargs='+', type=int)

    parser.add_argument('-k', nargs='+', type=int)

    parser.add_argument('--mode', type=str, choices=['search', 'eval'])

    parser.add_argument('--output_pkl', type=str)
    parser.add_argument('--output_csv', type=str)

    return parser.parse_args()

"""def parse_config_file(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config: return args
    
    cfg = yaml.safe_load(args.config)
    for key, value in vars(args).items():
        if value is not None:
            cfg[key] = value
    return cfg
"""
"""def load_dataset(path: str) -> list[np.ndarray]:
    images = []
    pattern = os.path.join(path, '*.jpg')
    file_list = sorted(glob.glob(pattern))
    for file in file_list:
        f = os.path.join(path, file)
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3, None)
        images.append(img)
    return images"""

def _limit_size(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = img.shape[:2]
    ms = max(h, w)
    if max_side > 0 and ms > max_side:
        scale = max_side / ms
        img = cv2.GaussianBlur(img, (3, 3), 1)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def apply_mask(img_bgr: np.ndarray, mask: np.ndarray, transparent: bool = True) -> np.ndarray:
    """
    Keep only the white region(s) of `mask` in `img_bgr`.
    If `transparent=True`, return BGRA with outside set to alpha=0.
    Otherwise return BGR with outside set to black.
    """
    # ensure mask is single-channel uint8 the same size as the image
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    bin_mask = (mask > 127).astype(np.uint8)  # 0/1

    if transparent:
        # BGRA output with alpha = mask
        out = np.dstack([img_bgr, (bin_mask * 255).astype(np.uint8)])
    else:
        # black outside
        out = img_bgr.copy()
        out[bin_mask == 0] = 0
        # Apply homography
    return out

def split_by_components(img_bgr: np.ndarray, mask: np.ndarray, min_area: int = 5000) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Split an image into separate paintings using connected components on the mask.
    Returns a list of (cropped_img, cropped_mask) for each component (largest first).
    """
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape[:2] != img_bgr.shape[:2]:
        mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    bin_mask = (mask > 127).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    comps = []
    coords = []
    for i in range(1, num):  # skip background
        x, y, w, h, area = stats[i, 0], stats[i, 1], stats[i, 2], stats[i, 3], stats[i, 4]
        if area < min_area:
            continue
        crop_img  = img_bgr[y:y+h, x:x+w]
        crop_mask = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
        comps.append((crop_img, crop_mask))
        coords.append(np.array([x, y]))

    coords_array = np.stack(coords, axis=0)
    diff_x = coords_array[..., 0].max(axis=0) - coords_array[..., 0].min(axis=0)
    diff_y = coords_array[..., 1].max(axis=0) - coords_array[..., 1].min(axis=0)
    
    if (diff_x > diff_y):
        comps, coords = zip(*sorted(zip(comps, coords), key=lambda p: p[1][0]))
    else:
        comps, coords = zip(*sorted(zip(comps, coords), key=lambda p: p[1][1]))
    return comps

def load_dataset(path: str, masks: list, max_size: int) -> list[np.ndarray]:
    images = []
    file_list = sorted(glob.glob(os.path.join(path, '*.jpg')))
    for f, mask in zip(file_list, masks):
        img = cv2.imread(f)
        paintings = []
        parts = split_by_components(img, mask)
        for i, (crop_img, crop_mask) in enumerate(parts, 1):
            crop = apply_mask(crop_img, crop_mask, transparent=False)
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = cv2.medianBlur(crop, 3)
            crop = _limit_size(crop, max_size)
            paintings.append(crop)
        images.append(paintings)
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

"""def find_match(qs: list[np.ndarray], db: database2.Database, cfg: itertools.product[tuple], k_list: list[int]) -> list:
    db.change_params(cfg['kp_descriptor'], cfg, autoprocess=True)
    results = [[] for _ in k_list]
    for img in qs:
        kp, desc = generate_descriptor(img, cfg[0])
        matches = db.get_similar(kp, desc)
        
        for idx, k in enumerate(k_list):
            for match in matches:
                results[idx].append(match[:k])
    
    return results"""

def find_match(qs: list[list[np.ndarray]], db: database2.Database, cfg: dict[str, any], k_list: list[int], show=False):
    db.change_match_params(lowe_ratio=cfg['lowe_ratio'], min_good=cfg['min_good'], match_distance=cfg['match_distance'])

    results = [[] for _ in k_list]
    for q in tqdm(qs, desc='Matching query set', leave=False):
        parts = q
        per_query = [[] for _ in k_list]
        for img in parts:
            desc = img
            if desc is None or len(desc) == 0:
                ranked = []
            else:
                ranked = db.get_similar(desc)
            
            for k_idx, k in enumerate(k_list):
                per_query[k_idx].append(ranked[:k])
        
        for k_idx in range(len(k_list)):
            results[k_idx].append(per_query[k_idx])
    return results

def dataset_mapk(
    results: list, groundtruth: list,
    best_mapk: list, best_config: list, best_result: list,
    cfg: dict, k_list: list[int]
) -> tuple[list]:
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
    kd = set(args.kp_descriptor) if isinstance(args.kp_descriptor, list) else {args.kp_descriptor}
    combos = []
    arg_keys = []

    def product_dict(**kwargs):
        keys = list(kwargs.keys())
        vals = [kwargs[k] for k in keys]
        for inst in itertools.product(*vals):
            yield dict(zip(keys, inst))

    if 'sift' in kd:
        sift_params = {
            'kp_descriptor': ['sift'],
            'n_features': args.n_features,
            'edge_threshold': args.edge_threshold,
            'n_octave_layers': args.n_octave_layers,
            'contrast_threshold': args.contrast_threshold,
            'sigma': args.sigma,
            'lowe_ratio': args.lowe_ratio,
            'min_good': args.min_good,
            'match_distance': ['l2'],
            'max_size': args.max_size
        }
        combos += list(product_dict(**sift_params))
        arg_keys += list(sift_params.keys())

    if 'orb' in kd:
        orb_params = {
            'kp_descriptor': ['orb'],
            'n_features': args.n_features,
            'edge_threshold': args.edge_threshold,
            'scale_factor': args.scale_factor,
            'n_levels': args.n_levels,
            'WTA_K': args.WTA_K,
            'patch_size': args.patch_size,
            'fast_threshold': args.fast_threshold,
            'lowe_ratio': args.lowe_ratio,
            'min_good': args.min_good,
            'match_distance': ['hamming'],
            'max_size': args.max_size
        }
        combos += list(product_dict(**orb_params))
        arg_keys += list(orb_params.keys())

    if 'akaze' in kd:
        akaze_params = {
            'kp_descriptor': ['akaze'],
            #'descriptor_type': args.descriptor_type,
            'descriptor_size': args.descriptor_size,
            'descriptor_channels': args.descriptor_channels,
            'threshold': args.threshold,
            'n_octaves': args.n_octaves,
            'n_octave_layers': args.n_octave_layers,
            'lowe_ratio': args.lowe_ratio,
            'min_good': args.min_good,
            'match_distance': ['hamming'],
            #'diffusivity': args.diffusivity,
            'max_size': args.max_size
        }
        combos += list(product_dict(**akaze_params))
        arg_keys += list(akaze_params.keys())
    
    arg_keys = list(set(arg_keys))
    return combos, arg_keys

def extract_final_mask(res):
    import numpy as np
    if isinstance(res, dict):
        for key in ('fm', 'final_mask', 'mask', 'pred_mask'):
            if key in res:
                return res[key]
    if isinstance(res, (list, tuple)):
        candidates = [-1, 7, 3, 1]
        for k in candidates:
            if -len(res) <= k < len(res):
                x = res[k]
                if isinstance(x, np.ndarray) and x.ndim == 2:
                    return x
        arrays2d = [x for x in res if isinstance(x, np.ndarray) and x.ndim == 2]
        if arrays2d:
            return arrays2d[-1]
    raise ValueError("No se pudo extraer la máscara final de 'res'.")

def get_masks(dataset_path):
    masks = []
    image_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.jpg')])
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(dataset_path, image_file)
        im = imageio.imread(image_path)  # RGB
        im = _limit_size(im, max_side=512)

        _, splitted = split_images(preprocess_image(im))   # (flag, parts or original)

        # Normalize to list of parts
        parts = list(splitted) if isinstance(splitted, (list, tuple)) else [splitted]

        # Optional: better labels if vertical split is later detected
        part_labels = ["izquierda", "derecha"][:len(parts)]

        # 3️⃣ Procesar cada subimagen individualmente
        results = []  # ✅ keep only one initialization
        for i, part in enumerate(parts):
            result = remove_background_morphological_gradient(part)
            results.append(result)

        # 4️⃣ Combinar máscaras (respetando orientación)
        if len(results) == 1:
            fm = extract_final_mask(results[0])
            pred_bool = (fm > 0) if fm.dtype != bool else fm
        else:
            # 1) Resize each predicted mask back to its part size
            resized_masks = []
            for part, r in zip(parts, results):
                fm = extract_final_mask(r)
                fm_bool = (fm > 0) if fm.dtype != bool else fm
                ph, pw = part.shape[:2]
                fm_res = cv2.resize(fm_bool.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST).astype(bool)
                resized_masks.append(fm_res)

            # 2) Decide whether the split is horizontal or vertical by part shapes
            h0, w0 = parts[0].shape[:2]
            h1, w1 = parts[1].shape[:2]

            if h0 == h1:
                pred_bool = np.hstack(resized_masks)   # horizontal (left|right)
            elif w0 == w1:
                pred_bool = np.vstack(resized_masks)   # vertical (top over bottom)
                part_labels = ["arriba", "abajo"]      # ✅ optional relabel
            else:
                # Rare fallback: pad along height and hstack
                max_h = max(m.shape[0] for m in resized_masks)
                padded = [np.pad(m, ((0, max_h - m.shape[0]), (0, 0)), constant_values=False) if m.shape[0] < max_h else m
                        for m in resized_masks]
                pred_bool = np.hstack(padded)

        base = os.path.splitext(image_file)[0]

        # Save mask
        mask_uint8 = pred_bool.astype(np.uint8) * 255
        os.makedirs("outputs_mask", exist_ok=True)
        save_mask_path = os.path.join("outputs_mask", f"{base}_mask.png")
        imageio.imwrite(save_mask_path, mask_uint8)
        print(f"✅ Saved binary mask: {save_mask_path}")
        masks.append(mask_uint8)
    return masks

def main():
    args = parse_config_file(parse_args())
    check_args(args)

    if args.mode == 'eval':
        masks = get_masks(args.dataset)
        # masks = []
        # image_files = sorted([f for f in os.listdir('outputs_mask') if f.endswith('.png')])
        # for f in image_files:
        #     path = os.path.join('outputs_mask', f)
        #     mask = cv2.imread(path)
        #     masks.append(mask)
        pass
    else:
        masks = []
        image_files = sorted([f for f in os.listdir('outputs_mask') if f.endswith('.png')])
        for f in image_files:
            path = os.path.join('outputs_mask', f)
            mask = cv2.imread(path)
            masks.append(mask)

    db = database2.Database(args.database, 512)
    qs = load_dataset(args.dataset, masks, 512)

    if args.mode == 'search':
        best_config = [[] for _ in range(len(args.k))]
        best_result = [[] for _ in range(len(args.k))]
        best_mapk = [0 for _ in range(len(args.k))]
        with open(os.path.join(args.dataset, 'gt_corresps.pkl'), 'rb') as f:
            groundtruth = pickle.load(f)

        combos, arg_keys = generate_combos(args)

        grid_search_df = pd.DataFrame(columns=arg_keys + [f'mapk{k}' for k in args.k])
        
        db.change_desc_params(combos[0]['kp_descriptor'], combos[0], autoprocess=True)

        q_desc = []
        for q in qs:
            parts = q
            subq_kp = []
            for img in parts:
                kp, desc = generate_descriptor(img, combos[0]['kp_descriptor'], **combos[0])
                subq_kp.append(desc)
            q_desc.append(subq_kp)
        last_desc = combos[0]['kp_descriptor']

        for cfg in tqdm(combos, desc="Grid Search combos"):
            masks = []
            image_files = sorted([f for f in os.listdir('outputs_mask') if f.endswith('.png')])
            for f in image_files:
                path = os.path.join('outputs_mask', f)
                mask = cv2.imread(path)
                masks.append(mask)
            
            db = database2.Database(args.database, cfg['max_size'])
            qs = load_dataset(args.dataset, masks, cfg['max_size'])
            
            db.change_desc_params(cfg['kp_descriptor'], cfg, autoprocess=True)
            q_desc = []
            for q in qs:
                parts = q
                subq_kp = []
                for img in parts:
                    kp, desc = generate_descriptor(img, cfg['kp_descriptor'], **cfg)
                    subq_kp.append(desc)
                q_desc.append(subq_kp)
            last_desc = cfg['kp_descriptor']

            results = find_match(q_desc, db, cfg, args.k, show=True)

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
            print(f"Groundtruth: {groundtruth}")
            print(f"Best results: {results}")
            print(f"Best config: {config}")
            print(f"Best mapk: {mapk:.4f}")

    elif args.mode == 'eval':
        combos, _ = generate_combos(args)
        cfg = combos[0]
        db.change_desc_params(cfg['kp_descriptor'], cfg, autoprocess=True)
        db.change_match_params(lowe_ratio=cfg['lowe_ratio'], min_good=cfg['min_good'], match_distance=cfg['match_distance'])
        q_desc = []
        for q in qs:
            parts = q
            subq_kp = []
            for img in parts:
                kp, desc = generate_descriptor(img, cfg['kp_descriptor'], **cfg)
                subq_kp.append(desc)
            q_desc.append(subq_kp)
        results = find_match(q_desc, db, cfg, args.k)

        for result in results:
            with open(args.output_pkl, 'wb') as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
