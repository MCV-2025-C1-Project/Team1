import numpy as np

def apk(actual, predicted, k=10):
    """Average precision at k for a single painting."""
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            print("actual, predicted, score:", actual, predicted, score)
    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10, multi=False):
    """
    Computes mean average precision at k for:
      - actual: list of lists (ground truth idx per image)
      - predicted: list of lists of lists (predicted idx per painting in each image)
    """
    if multi:
        all_apks = []

        for actual_img, predicted_img in zip(actual, predicted):
            print("actual_img, predicted_img:", actual_img, predicted_img)
            # For each painting in the image
            for gt_idx, pred_list in zip(actual_img, predicted_img):
                print("gt_idx, pred_list", gt_idx, pred_list)
                all_apks.append(apk([gt_idx], pred_list, k))  # wrap gt_idx as list

        return np.mean(all_apks) if all_apks else 0.0
    else:
        return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    

def mapk_new(actual, predicted, k=10, multi=False):
    """
    Computes mean average precision at k for:
      - actual: list of lists (ground truth idx per image)
      - predicted: list of lists of lists (predicted idx per painting in each image)
    """
    if multi:
        all_apks = []

        for actual_img, predicted_img in zip(actual, predicted):
            # For each painting in the image
            all_apks.append(apk([actual_img], predicted_img, k))  # wrap gt_idx as list

        return np.mean(all_apks) if all_apks else 0.0
    else:
        return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
    
