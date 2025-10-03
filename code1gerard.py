#CANVIAR A IMPORTAR BÉ DATASET ns com va a github !!

import os
import cv2
import numpy as np
import pickle
import re

# Correct paths
path_bbdd = "../BBDD/BBDD"   
path_qsd1_w1 = "../qsd1_w1/qsd1_w1"  

# Collect all image files (sorted by name)
bbdd_files = sorted([
    os.path.join(path_bbdd, f)
    for f in os.listdir(path_bbdd)
    if f.lower().endswith(('.jpg', '.jpeg'))
])

qsd1_w1_files = sorted([
    os.path.join(path_qsd1_w1, f)
    for f in os.listdir(path_qsd1_w1)
    if f.lower().endswith(('.jpg', '.jpeg'))
])

print(bbdd_files[:3])
print(qsd1_w1_files[:3])


print(f"Loaded {len(bbdd_files)} images from BBDD")
print(f"Loaded {len(qsd1_w1_files)} images from qsd1_w1")


pkl_path = "../qsd1_w1/qsd1_w1/gt_corresps.pkl"
with open(pkl_path, "rb") as f:
    gt_corresps = pickle.load(f)


def compute_1d_channel_histograms(img, colorspace="RGB", bins=256, normalize=True):
    """
    Compute concatenated 1D histograms for each channel of an image.

    Parameters
    ----------
    img : ndarray (BGR by default from cv2.imread)
    colorspace : str, one of {"RGB", "HSV", "Lab", "YCbCr", "GRAY"}
    bins : int, number of bins per channel (default 256)
    normalize : bool, if True normalize each channel histogram so total = 1

    Returns
    -------
    hist : 1D numpy array of shape (channels * bins,)
    """

    # Convert to chosen color space
    if colorspace.upper() == "RGB":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif colorspace.upper() == "HSV":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif colorspace.upper() == "LAB":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    elif colorspace.upper() == "YCRCB":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif colorspace.upper() == "GRAY":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Color space '{colorspace}' not supported.")

    # Split into channels
    if colorspace.upper() == "GRAY":
        channels = [img_cs]
    else:
        channels = cv2.split(img_cs)

    hist_list = []
    for ch in channels:
        h = cv2.calcHist([ch], [0], None, [bins], [0, 256])
        h = h.flatten().astype("float32")
        if normalize:
            h /= (h.sum() + 1e-8)
        hist_list.append(h)

    # Concatenate all 1D histograms
    return np.concatenate(hist_list)



def hist_distance(h1: np.ndarray, h2: np.ndarray, metric: str = "chi2", eps: float = 1e-12) -> float:
    """
    Compute a distance between two 1D histograms using several metrics.

    Parameters
    ----------
    h1, h2 : np.ndarray
        1D histograms of the same length (already normalized).
        Can be grayscale (256 bins) or concatenated color channels (3*256 = 768).
    metric : str
        One of {"l2","l1","chi2","intersection","hellinger","cosine","bhattacharyya"}.
    eps : float
        Small constant to avoid division by zero and log(0).

    Returns
    -------
    float
        Distance value (smaller means more similar).
    """
    m = metric.strip().lower()
    #h1 = np.asarray(h1, dtype=np.float64).ravel()
    #h2 = np.asarray(h2, dtype=np.float64).ravel()

    assert h1.shape == h2.shape, "Histograms must have the same length."

    if m == "l2":
        return float(np.linalg.norm(h1 - h2, ord=2))

    if m == "l1":
        return float(np.linalg.norm(h1 - h2, ord=1))

    if m == "chi2":
        denom = h1 + h2 + eps
        return float(0.5 * np.sum(((h1 - h2) ** 2) / denom))

    if m == "intersection":
        # inter is the sum of the minimums
        inter = float(np.sum(np.minimum(h1, h2)))
        total = float(h1.sum())  # will be 1 (grayscale) or 3 (color)
        return float(total - inter)

    if m == "hellinger":
        return float(np.linalg.norm(np.sqrt(h1 + eps) - np.sqrt(h2 + eps)) / np.sqrt(2.0))

    if m == "cosine":
        num = float(np.dot(h1, h2))
        den = float(np.linalg.norm(h1) * np.linalg.norm(h2)) + eps
        cos_sim = num / den
        return float(1.0 - cos_sim)

    if m == "bhattacharyya":
        bc = float(np.sum(np.sqrt((h1 + eps) * (h2 + eps))))
        return float(-np.log(bc + eps))

    raise ValueError(f"Unsupported metric: {metric}")


color_spaces = ["RGB", "HSV", "Lab", "YCrCb", "GRAY"]
metrics = ["l2", "l1", "chi2",  "intersection", "hellinger", "cosine", "bhattacharyya"]

# Precompute histograms for BBDD
bbdd_hists = {cs: [] for cs in color_spaces}
for cs in color_spaces:
    for path in bbdd_files:
        img = cv2.imread(path)
        hist = compute_1d_channel_histograms(img, colorspace=cs, bins=256, normalize=True)
        bbdd_hists[cs].append(hist)
    bbdd_hists[cs] = np.stack(bbdd_hists[cs])  # shape (N_bbdd, bins)
    
# Precompute histograms for queries
qsd_hists = {cs: [] for cs in color_spaces}
for cs in color_spaces:
    for path in qsd1_w1_files:
        img = cv2.imread(path)
        hist = compute_1d_channel_histograms(img, colorspace=cs, bins=256, normalize=True)
        qsd_hists[cs].append(hist)
    qsd_hists[cs] = np.stack(qsd_hists[cs])  # shape (N_queries, bins)


results = {}  # (cs, metric) -> list of predicted indices

for cs in color_spaces:
    for m in metrics:
        preds = []
        for q_hist in qsd_hists[cs]:
            dists = [hist_distance(q_hist, b_hist, metric=m) for b_hist in bbdd_hists[cs]]
            best_idx = int(np.argmin(dists))  # nearest neighbor
            preds.append(best_idx)  # index in bbdd_files
        results[(cs, m)] = preds



def id_from_path(p):
    # extrae el número final del nombre: bbdd_00120.jpg -> 120
    fname = os.path.basename(p)
    base = os.path.splitext(fname)[0]
    nums = re.findall(r"\d+", base)
    return int(nums[-1]) if nums else -1

def preds_to_ids(pred_indices, bbdd_paths):
    return [id_from_path(bbdd_paths[i]) for i in pred_indices]



# EVALUACION SPACECOLOR+DISTANCIA SIMPLE PRINT
"""
# --- evaluación ---
for (cs, m), pred_indices in results.items():
    pred_ids = preds_to_ids(pred_indices, bbdd_files)

    correct = sum(int(p in gt) for p, gt in zip(pred_ids, gt_corresps))
    total = len(gt_corresps)
    errors = total - correct

    acc = 100.0 * correct / total
    err = 100.0 * errors / total

    print(f"{cs}+{m}: {correct} aciertos, {errors} errores "
          f"-> {acc:.1f}% correct, {err:.1f}% wrong")

"""

#HACER GRAFICAS EN FUNCION DE BITS

import matplotlib.pyplot as plt

# === choose the sweep of bins ===
# Simple linear schedule requested: 256, 246, 236, ... down to 16
#bin_values = list(range(256, 15, -10))   # stop at >=16
# If you want a more "balanced" schedule, consider something like:
bin_values = [256, 192, 128, 96, 64, 32, 16]

# Use your full set or a subset (can be slow with all combos)
color_spaces = ["RGB", "HSV", "Lab", "YCrCb", "GRAY"]
metrics = ["l2", "l1", "chi2", "intersection", "hellinger", "cosine", "bhattacharyya"]

def evaluate_nn_accuracy(bins, cs, metric):
    """
    Compute accuracy (precision@1) for a given number of bins, color space, and metric.
    """
    # Precompute BBDD histograms for this setup
    bbdd_h = []
    for path in bbdd_files:
        img = cv2.imread(path)
        hist = compute_1d_channel_histograms(img, colorspace=cs, bins=bins, normalize=True)
        bbdd_h.append(hist)
    bbdd_h = np.stack(bbdd_h)

    # Precompute QUERY histograms for this setup
    qsd_h = []
    for path in qsd1_w1_files:
        img = cv2.imread(path)
        hist = compute_1d_channel_histograms(img, colorspace=cs, bins=bins, normalize=True)
        qsd_h.append(hist)
    qsd_h = np.stack(qsd_h)

    # NN predictions (indices in bbdd_files)
    preds_idx = []
    for q_hist in qsd_h:
        dists = [hist_distance(q_hist, b_hist, metric=metric) for b_hist in bbdd_h]
        best_idx = int(np.argmin(dists))
        preds_idx.append(best_idx)

    # Convert indices -> IDs to compare with GT
    def id_from_path(p):
        fname = os.path.basename(p)
        base = os.path.splitext(fname)[0]
        nums = re.findall(r"\d+", base)
        return int(nums[-1]) if nums else -1

    pred_ids = [id_from_path(bbdd_files[i]) for i in preds_idx]

    # Precision@1 = Recall@1 = accuracy (1 GT per query)
    correct = sum(int(p in gt) for p, gt in zip(pred_ids, gt_corresps))
    total = len(gt_corresps)
    acc = 100.0 * correct / total
    return acc

# Run the sweep and store accuracies:
acc_curves = {}  # (cs, metric) -> list of accuracies matching bin_values order

for cs in color_spaces:
    for m in metrics:
        acc_list = []
        for b in bin_values:
            acc = evaluate_nn_accuracy(b, cs, m)
            acc_list.append(acc)
        acc_curves[(cs, m)] = acc_list
        print(f"Done: {cs}+{m}")

# === Plot ===
plt.figure(figsize=(10, 7))
for (cs, m), accs in acc_curves.items():
    label = f"{cs}+{m}"
    plt.plot(bin_values, accs, label=label)

plt.gca().invert_xaxis()  # optional: show 256 -> 16 from left to right
plt.xlabel("Bins per channel")
plt.ylabel("% correct (Precision@1 = Recall@1)")
plt.ylim(0, 100)  # <-- force Y axis between 0 and 100
plt.title("Accuracy vs. Number of Bins (one line per colorspace+distance)")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)  # tweak as you like
plt.tight_layout()
plt.show()





"""
def precision_recall(preds, gts, n_bbdd):
    # preds: list of predicted indices (len = N_queries)
    # gts: list of lists with ground-truth indices
    y_true, y_score = [], []
    for pred, gt in zip(preds, gts):
        for i in range(n_bbdd):
            y_true.append(1 if i in gt else 0)
            y_score.append(1 if i == pred else 0)


    from sklearn.metrics import precision_recall_curve, average_precision_score


    precision, recall, _ = precision_recall_curve(y_true, y_score)
    

    ap = average_precision_score(y_true, y_score)
    return precision, recall, ap

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

plt.figure(figsize=(8,6))

for (cs, m), preds in results.items():
    precision, recall, ap = precision_recall(preds, gt_corresps, len(bbdd_files))
    print("Pred index:", preds[0], "=> file", bbdd_files[preds[0]])
    print("GT IDs:", gt_corresps[0])
    plt.plot(recall, precision, label=f"{cs}+{m} (AP={ap:.2f})")
    # Mark the operating point (since NN gives a single choice)
    plt.scatter(recall[-1], precision[-1], marker="o")




plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curves for color spaces + distances")
plt.legend()
plt.grid(True)
plt.show()

"""