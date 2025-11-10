import os
import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern
from scipy.fftpack import dct
from .noise_filter import fourier_noise_score, remove_noise_median, remove_noise_nlmeans, remove_noise_bilateral
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# CONFIG
IMG_FOLDER_NOISY = "../Data/Week3/qsd1_w3/"
IMG_FOLDER_GT = "../Data/Week3/BBDD/"
#IMG_FOLDER_GT = "../Data/Week3/qsd1_w3/non_augmented"
GT_CORRESPS_PATH = "../Data/Week3/qsd1_w3/gt_corresps.pkl"
THRESHOLD = 40
SHOW_EXAMPLES = False  # Set to True to visualize noisy vs denoised images

# Noise preprocessing

def preprocess_image(img, show_examples=SHOW_EXAMPLES):
    """Detect noise and denoise if necessary."""
    noise_score = fourier_noise_score(img, radius_ratio=0.75)
    denoised_img = img.copy()

    if noise_score > THRESHOLD:
        denoised_img = remove_noise_median(img, ksize=3)
        #denoised_img = remove_noise_nlmeans(denoised_img, h=5, templateWindowSize=3, searchWindowSize=21)
        denoised_img = remove_noise_bilateral(denoised_img, d=21, sigmaColor=25, sigmaSpace=50)

        if show_examples:
            fig, axes = plt.subplots(1, 2, figsize=(10,5))
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"Original (score={noise_score:.2f})")
            axes[0].axis('off')
            axes[1].imshow(cv2.cvtColor(denoised_img, cv2.COLOR_BGR2RGB))
            axes[1].set_title("Denoised")
            axes[1].axis('off')
            plt.show()
    return denoised_img


# Descriptors

def compute_lbp_descriptor(img, num_points=8, radius=1, grid_x=4, grid_y=4, multiscale=False):
    """Compute (possibly multiscale) block-based LBP descriptor."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptors = []

    scales = [radius] if not multiscale else [1, 2, 3]

    for r in scales:
        lbp = local_binary_pattern(gray, num_points, r, method='uniform')
        h, w = lbp.shape
        block_h, block_w = h // grid_y, w // grid_x

        histograms = []
        for i in range(grid_y):
            for j in range(grid_x):
                block = lbp[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                hist, _ = np.histogram(block.ravel(),
                                       bins=np.arange(0, num_points + 3),
                                       range=(0, num_points + 2))
                hist = hist.astype("float") / (hist.sum() + 1e-8)
                histograms.extend(hist)
        descriptors.extend(histograms)

    return np.array(descriptors)

def compute_dct_descriptor(img, num_coeff=64):
    """Compute 2D DCT descriptor (zig-zag scan)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')
    dct_vector = dct_coeffs.flatten()[:num_coeff]
    return dct_vector


# Descriptor extraction

def extract_descriptors(folder, preprocess=False, multiscale_lbp=True):
    descriptors = []
    img_names = sorted([f for f in os.listdir(folder) if f.endswith('.jpg')])

    for name in img_names:
        print(f"\nProcessing {name} ...")
        img_path = os.path.join(folder, name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping {name}: could not read image.")
            continue
        if preprocess:
            img = preprocess_image(img)

        lbp_desc = compute_lbp_descriptor(img, multiscale=multiscale_lbp)
        dct_desc = compute_dct_descriptor(img)
        combined_desc = np.concatenate([lbp_desc, dct_desc])

        descriptors.append(combined_desc)

    return np.array(descriptors), img_names

def extract_descriptor(img, multiscale_lbp=True):
    #img = cv2.imread(img)

    # Compute descriptors
    lbp_desc = compute_lbp_descriptor(img, multiscale=multiscale_lbp)
    dct_desc = compute_dct_descriptor(img)
    combined_desc = np.concatenate([lbp_desc, dct_desc])

    return combined_desc



# Evaluations

def compute_map_at_k(descriptors_query, descriptors_gt, gt_corresps, k=5):
    """Compute mean Average Precision at K using GT correspondences."""
    sims = cosine_similarity(descriptors_query, descriptors_gt)
    map_scores = []

    if isinstance(gt_corresps, list):
        gt_corresps_dict = {i: gt_corresps[i] for i in range(len(gt_corresps))}
    else:
        gt_corresps_dict = gt_corresps

    for i in range(len(descriptors_query)):
        ranked_indices = np.argsort(-sims[i])[:k]  # descending order
        gt_indices = gt_corresps_dict.get(i, [])
        num_relevant = len(gt_indices)
        num_correct = 0
        precision_at_i = []

        for rank, idx in enumerate(ranked_indices, start=1):
            if idx in gt_indices:
                num_correct += 1
                precision_at_i.append(num_correct / rank)

        ap = np.sum(precision_at_i) / num_relevant if num_relevant > 0 else 0
        map_scores.append(ap)

    return np.mean(map_scores)


# Main

def main():

    desc_query, query_names = extract_descriptors(IMG_FOLDER_NOISY, preprocess=False)
    desc_gt, gt_names = extract_descriptors(IMG_FOLDER_GT, preprocess=False)

    os.makedirs("results", exist_ok=True)
    with open("results/descriptors_task1_2.pkl", "wb") as f:
        pickle.dump({
            "query_descriptors": desc_query,
            "gt_descriptors": desc_gt,
            "query_names": query_names,
            "gt_names": gt_names
        }, f)
    print("\n Descriptors saved to results/descriptors.pkl")

    gt_corresps = {i: [i] for i in range(len(desc_query))}

    # Charge pkl file for gt correspondances with the BBDD folder
    with open(GT_CORRESPS_PATH, "rb") as f:
        gt_corresps = pickle.load(f)
        
    map1 = compute_map_at_k(desc_query, desc_gt, gt_corresps, k=1)
    map5 = compute_map_at_k(desc_query, desc_gt, gt_corresps, k=5)

    print(f"\n Evaluation Results:")
    print(f"mAP@1 = {map1:.4f}")
    print(f"mAP@5 = {map5:.4f}")

if __name__ == "__main__":
    main()
