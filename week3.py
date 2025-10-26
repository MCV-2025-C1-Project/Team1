import argparse
import glob
import itertools
import os
import pickle
import textwrap
import time

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

import constants
import database
import background.w2_mask as w2_mask
from descriptors import histograms, preprocessing, LBP, DCT, filters, wavelets
from metrics import average_precision
import mask_creation_w3_main


def parse_yaml(config):
    if not config:
        return None

    yaml_args = {}
    with open(config, 'r') as f:
        contents = yaml.safe_load(f) or {}
        for section_key, params in contents.items():
            if isinstance(params, dict):
                # merge all keys inside this section
                for key, value in params.items():
                    yaml_args[key] = value
            else:
                # scalar at top level (e.g., pickle_filename)
                yaml_args[section_key] = params

    return yaml_args



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run image retrieval evaluation.",
        epilog=textwrap.dedent('''\
            Example usage:
              python week3.py --k 1 5 10 \\
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
    
    parser.add_argument('--color_space_list', nargs='+', type=str, default=['lab'],
                        help='Color space(s) to use. Options: gray_scale, rgb, hsv, lab, ycbcr. (Default: [lab])')
    parser.add_argument('--preprocesses_list', nargs='+', type=str, default=[None],
                        help="""Preprocessing methods applied to both DB and queries. Options: clahe, hist_eq, gamma, contrast, gaussian_blur, median_blur, bilateral, unsharp, None. (Default: [None])\n
                                If you want to add an option of no preprocessing to a list, add either an invalid argument (such as None) to the list or an empty string
                                in case of using a yaml file. Example terminal: [None, l1]. Example yaml: ['', euclidean]""")
    parser.add_argument('--masking', type=bool, default=False,
                        help='Whether to remove the background by getting a mask.')
    
    parser.add_argument('--descriptors_list', type=str, nargs='+', default=['hist'],
                        help='Image descriptor. Options: hist, LBP, Multiscale_LBP, OCLBP, DCT, wavelet. (Default: [hist])')
    
    parser.add_argument('--bins_list', type=int, nargs='+', default=[64],
                        help='Number of histogram bins per channel. (Default: [64])')
    parser.add_argument('--blocks_list', nargs='+', type=int, default=[1],
                        help="Number of blocks to divide the image into")
    parser.add_argument('--hist_dims_list', nargs='+', type=int, default=[1],
                        help='Number of dimensions of the histogram')
    parser.add_argument('--LBP_scales_list', type=eval, nargs='+', default=[(8,1.0)],
                        help=(
                            "LBP scale(s). For LBP or OCLBP pass a single tuple like (P,R). "
                            "For Multiscale LBP pass a list of tuples like [(P1,R1),(P2,R2),...]. "
                            "Each tile will produce as many histograms as there are scales, and they will be concatenated"
                            "P = number of neighbors, R = radius."
                        ))
    
    parser.add_argument('--OCLBP_uniform_u2', type=bool, default=False,
                        help=("if True use uniform-u2 mapping -> bins = P + 2 per scale"
                              "else: bins. Options: True / False. (Default: False)"))
    parser.add_argument('--DCT_coeffs_list',  nargs='+', type=int, default=[16],
                        help='Number of DCT coefficients to keep per block (taken in zig-zag)')
    
    parser.add_argument('--wavelets_list',  nargs='+', type=str, default=["bior1.1"],
                        help='Wavelet')
    
    parser.add_argument('--distances_list', nargs='+', type=str, default=['hist_intersection'],
                        help='Distance / similarity metrics. Options: euclidean, l1, x2, hist_intersection, hellinger, canberra. (Default: [hist_intersection])')
    
    parser.add_argument('--k_list', nargs='+', type=int, default=[1],
                        help='List of K values for top-K retrieval (e.g. --k 1 5 10)')
    parser.add_argument('--val', type=bool, default=False,
                        help='Whether to run validation (expects gt_corresps.pkl in query directory). Options: True / False. (Default: False)')
    parser.add_argument('--pickle_filename', type=str, default=None,
                        help='If set, pickle ALL combos (params + results) into this file.')
    
    tmp_args = parser.parse_args()
    yaml_args = parse_yaml(tmp_args.config)
    if yaml_args:
        parser.set_defaults(**yaml_args)

    args = parser.parse_args()

    return args

def process_queries_for_combo(query_list_raw, color_space, preprocess, masking):
    """Return queries converted to color_space + preprocess (+mask if requested)."""
    qlist = []
    for query_set in query_list_raw:
        sub_qlist = []
        for img_bgr in query_set:
            q = cv2.cvtColor(img_bgr, constants.CV2_CVT_COLORS[color_space])
            if preprocess == 'clahe':
                q = preprocessing.clahe_preprocessing(q, color_space)
            elif preprocess == 'hist_eq':
                q = preprocessing.hist_eq(q, color_space)
            elif preprocess == 'gamma':
                q = preprocessing.gamma(q)
            elif preprocess == 'contrast':
                q = preprocessing.contrast(q)
            elif preprocess == 'gaussian_blur':
                q = preprocessing.gaussian_blur(q)
            elif preprocess == 'median_blur':
                q = preprocessing.median_blur(q)
            elif preprocess == 'bilateral':
                q = preprocessing.bilateral(q)
            elif preprocess == 'unsharp':
                q = preprocessing.unsharp(q)

            sub_qlist.append(q)
        qlist.append(sub_qlist)
    return qlist


def main():
    args = parse_args()

    db_path = args.database_path
    query_path = args.query_path

    color_spaces_list = args.color_space_list
    preprocesses_list = args.preprocesses_list
    masking = args.masking

    descriptors_list = args.descriptors_list
    bins_list = args.bins_list
    blocks_list = args.blocks_list

    hist_dims_list = args.hist_dims_list
    LBP_scales_list = args.LBP_scales_list

    uniform_u2 = args.OCLBP_uniform_u2
    DCT_coeffs_list = args.DCT_coeffs_list
    wavelets_list = args.wavelets_list

    distances_list = args.distances_list

    k_list = args.k_list
    val = args.val
    output_pickle = args.pickle_filename

    # Initial DB color/preprocess (will be changed later as needed)
    init_color_space = color_spaces_list[0]
    init_preprocess = preprocesses_list[0]

    if val:
        best_config = [[] for _ in range(len(k_list))]
        best_result = [[] for _ in range(len(k_list))]
        best_mapk = [0 for _ in range(len(k_list))]
        grid_search_df = pd.DataFrame(columns=['descriptor', 'color_space', 'preprocess',
                                               'bins', 'blocks', 'hist_dim', 'distances',
                                                'lbp_scales', 'dct_coeffs', 'wavelet',
                                                *[f'mapk{k}' for k in k_list]
                                            ])

        with open(os.path.join(query_path, 'gt_corresps.pkl'), 'rb') as f:
            groundtruth = pickle.load(f)

    # Fast, cheap init so we see the first tqdm quickly
    db = database.Database(
        db_path,
        color_space=init_color_space,
        preprocess=init_preprocess,
        bins=8,                 # small, cheap
        num_blocks=1,           # single tile
        hist_dims=1,
        descriptor="hist",      # CHEAP placeholder
        scales=LBP_scales_list[0],
        dct_coeffs=DCT_coeffs_list[0],
        oclbp_uniform_u2=uniform_u2,
        wavelet=wavelets_list[0]
    )

     
    # Load queries RAW BGR; process per combo later
    query_abs_path = os.path.abspath(query_path)
    query_pattern = os.path.join(query_abs_path, '*.jpg')
    query_list_raw = []
    for image_path in sorted(glob.glob(query_pattern, root_dir=query_abs_path)):
        img_bgr = cv2.imread(image_path)
        query_list_raw.append(filters.denoise_image(img_bgr, 'median', kernel_size=3))

    # Extract and apply masks
    masked_query_list_raw = []
    masks_list, quads_list = mask_creation_w3_main.process_dataset(query_path)
    for idx, (extracted_masks, extracted_quads) in enumerate(zip(masks_list, quads_list)):
        subimages = []
        for mask, quads in zip(extracted_masks, extracted_quads):
            image = query_list_raw[idx]

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
        masked_query_list_raw.append(subimages)
    
    query_list_raw = masked_query_list_raw

    # Collect every run (params + results) to pickle together
    all_runs = []

    # Descriptor-aware grid search
    for descriptor_ in descriptors_list:
        if descriptor_ == "hist":
            combos = itertools.product(color_spaces_list, preprocesses_list, bins_list, blocks_list, hist_dims_list)
            total = (len(color_spaces_list)*len(preprocesses_list)*len(bins_list)*len(blocks_list)*len(hist_dims_list))
        elif descriptor_ == "LBP":
            combos = itertools.product(color_spaces_list, preprocesses_list, bins_list, blocks_list)
            total = (len(color_spaces_list)*len(preprocesses_list)*len(bins_list)*len(blocks_list))
        elif descriptor_ == "Multiscale_LBP":
            combos = itertools.product(color_spaces_list, preprocesses_list, bins_list, blocks_list, LBP_scales_list)
            total = (len(color_spaces_list)*len(preprocesses_list)*len(bins_list)*len(blocks_list)*len(LBP_scales_list))
        elif descriptor_ == "OCLBP":
            combos = itertools.product(color_spaces_list, preprocesses_list, bins_list, blocks_list, LBP_scales_list)
            total = (len(color_spaces_list)*len(preprocesses_list)*len(bins_list)*len(blocks_list)*len(LBP_scales_list))
        elif descriptor_ == "DCT":
            combos = itertools.product(color_spaces_list, preprocesses_list, blocks_list, DCT_coeffs_list)
            total = (len(color_spaces_list)*len(preprocesses_list)*len(blocks_list)*len(DCT_coeffs_list))
        elif descriptor_ == "wavelet":
            combos = itertools.product(color_spaces_list, preprocesses_list, bins_list, blocks_list, hist_dims_list, wavelets_list)
            total = (len(color_spaces_list)*len(preprocesses_list)*len(bins_list)*len(blocks_list)*len(hist_dims_list)*len(wavelets_list))
        else:
            raise ValueError(f"Unknown descriptor: {descriptor_}")

        with tqdm(total=total) as pbar:
            current_cs = None
            current_pp = None
            current_queries = None

            for combo in combos:
                # Defaults so logging works even when fields are N/A
                bins = None
                blocks = None
                hist_dims = None

                if descriptor_ == "hist":
                    color_space, preprocess, bins, blocks, hist_dims = combo
                elif descriptor_ == "LBP":
                    color_space, preprocess, bins, blocks = combo
                elif descriptor_ == "Multiscale_LBP":
                    color_space, preprocess, bins, blocks, scales_ = combo
                    # Ensure list-of-tuples
                    scales_ = scales_ if (isinstance(scales_, list) and len(scales_) and isinstance(scales_[0], (list, tuple))) else [scales_]
                elif descriptor_ == "OCLBP":
                    color_space, preprocess, bins, blocks, pr_tuple = combo
                    P_, R_ = pr_tuple
                elif descriptor_ == "DCT":
                    color_space, preprocess, blocks, coeffs_ = combo
                elif descriptor_ == "wavelet":
                    color_space, preprocess, bins, blocks, hist_dims, wavelet = combo

                # Optional skip for heavy hist configs
                if descriptor_ == "hist":
                    if (hist_dims == 2 and (bins > 64 or blocks > 16)) or (hist_dims == 3 and (bins > 32 or blocks > 8)):
                        pbar.set_postfix({"desc": descriptor_, "cs": color_space, "pp": preprocess, "bins": bins, "blocks": blocks, "hD": hist_dims, "skip": True})
                        pbar.update(1)
                        continue

                # Recompute DB color/preprocess and queries if needed
                if (color_space != current_cs) or (preprocess != current_pp):
                    db.change_color(color_space, preprocess)
                    current_queries = process_queries_for_combo(query_list_raw, color_space, preprocess, masking)
                    current_cs, current_pp = color_space, preprocess

                # Recompute DB feature bank for this combo
                if descriptor_ == "hist":
                    db.change_hist(bins, blocks, hist_dims, descriptor_)
                elif descriptor_ == "LBP":
                    db.change_hist(bins, blocks, 1, descriptor_)
                elif descriptor_ == "Multiscale_LBP":
                    db.change_hist(bins, blocks, 1, descriptor_, scale=scales_)
                elif descriptor_ == "OCLBP":
                    db.change_hist(bins, blocks, 1, descriptor_, scale=(P_, R_))
                elif descriptor_ == "DCT":
                    db.change_hist(1, blocks, 1, descriptor_, coeffs=coeffs_)  # bins irrelevant for DCT
                elif descriptor_ == "wavelet":
                    db.change_hist(bins, blocks, hist_dims, descriptor_, wavelet=wavelet)


                # Retrieve for all queries
                for distance in distances_list:
                    results = [[] for _ in k_list]
                    for q_list in current_queries:
                        subresults = [[] for _ in k_list]
                        for q in q_list:
                            start_time = time.time()

                            if descriptor_ == "hist":
                                desc = histograms.gen_hist(q, bins, blocks, hist_dims)
                            elif descriptor_ == "LBP":
                                desc = LBP.get_LBP_hist(q, bins, blocks)
                            elif descriptor_ == "Multiscale_LBP":
                                desc = LBP.get_Multiscale_LBP_hist(q, bins, blocks, scales_)
                            elif descriptor_ == "OCLBP":
                                desc = LBP.get_OCLBP_hist(q, bins, blocks, P_, R_, use_uniform_u2=uniform_u2)
                            elif descriptor_ == "DCT":
                                desc = DCT.get_DCT_descriptor(q, blocks, coeffs=coeffs_)
                            elif descriptor_ == "wavelet":
                                desc = wavelets.wavelets_descriptor(q, wavelet=wavelet, bins=bins, num_windows=blocks, num_dimensions=hist_dims)                    
                            
                            order = db.get_top_k_similar_images(desc, distance)
                            for i, k in enumerate(k_list):
                                subresults[i].append(order[:k])
                        
                        for i in range(len(k_list)):
                            results[i].append(subresults[i])

                    # Evaluate & bookkeeping
                    if val:
                        mapk_list = []
                        for i_k, (result_k, k_val) in enumerate(zip(results, k_list)):
                            mapk = average_precision.mapk(groundtruth, result_k, k=k_val)
                            mapk_list.append(mapk)

                            if mapk >= best_mapk[i_k]:
                                extra = None
                                if descriptor_ == "DCT":
                                    extra = coeffs_
                                elif descriptor_ == "wavelet":
                                    extra = wavelet
                                elif descriptor_ == "OCLBP":
                                    extra = (P_, R_)
                                elif descriptor_ == "Multiscale_LBP":
                                    extra = scales_

                                best_config[i_k] = [descriptor_, color_space, preprocess, bins, blocks, hist_dims, distance, extra]
                                best_result[i_k] = results[i_k]
                                best_mapk[i_k] = mapk

                        # Row for CSV
                        row = {
                            'descriptor': descriptor_,
                            'color_space': color_space,
                            'preprocess': preprocess,
                            'bins': bins,
                            'blocks': blocks,
                            'hist_dim': hist_dims,
                            'distances': distance,
                            # new fields (None when not applicable)
                            'lbp_scales': (
                                scales_ if descriptor_ == "Multiscale_LBP"
                                else ((P_, R_) if descriptor_ == "OCLBP" else None)
                            ),
                            'dct_coeffs': (coeffs_ if descriptor_ == "DCT" else None),
                            'wavelet': (wavelet if descriptor_ == "wavelet" else None),
                            **{f'mapk{k}': mapk_list[i] for i, k in enumerate(k_list)}
                        }

                        grid_search_df.loc[len(grid_search_df)] = row

                        tqdm_post = {
                            "desc": descriptor_, "cs": color_space, "pp": preprocess,
                            "bins": bins, "blocks": blocks, "hD": hist_dims, "dist": distance
                        }
                        for i, k in enumerate(k_list):
                            tqdm_post[f"MAP@{k}"] = f"{mapk_list[i]:.3f}"
                        pbar.set_postfix(tqdm_post)

                    # Store this combo for unified pickle
                    run_record = {
                                "descriptor": descriptor_,
                                "color_space": color_space,
                                "preprocess": preprocess,
                                "distance": distance,
                                "bins": bins,
                                "blocks": blocks,
                                "hist_dim": hist_dims,
                                "coeffs": coeffs_ if descriptor_ == "DCT" else None,
                                "scales": ([(P_, R_)] if descriptor_ == "OCLBP" else (scales_ if descriptor_ == "Multiscale_LBP" else None)),
                                "wavelet": wavelet if descriptor_ == "wavelet" else None,   # ← add this
                                "k_list": k_list,
                                "results": results,
                            }

                    if val:
                        run_record["mapk_list"] = mapk_list
                    all_runs.append(run_record)

                pbar.update(1)

    if val:
        os.makedirs('./results', exist_ok=True)
        grid_search_df.to_csv('results/grid_search_results_w3_masks_clean_lab.csv', index=False)
        for config, results, mapk, k in zip(best_config, best_result, best_mapk, k_list):
            print(f"\nFor K = {k}\n")
            print(f"Best results: {results}")
            print(f"Best config: {config}")
            print(f"Best mapk: {mapk:.4f}")

    # Pickle ALL combos together if requested
    if output_pickle:
        with open(output_pickle, "wb") as f:
            pickle.dump(all_runs, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
