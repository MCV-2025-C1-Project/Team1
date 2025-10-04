import numpy as np

def euclidean_distance(prediction, groundtruth):
    """
    Compute the Euclidean (L2) distance between two vectors.

    Parameters
    ----------
    prediction : array_like
        Predicted feature vector or histogram.
    groundtruth : array_like
        Ground-truth feature vector or histogram.

    Returns
    -------
    float
        The Euclidean distance between `prediction` and `groundtruth`.
    """
    distance = groundtruth - prediction
    distance = distance * distance
    distance = np.sum(distance)
    distance = np.sqrt(distance)
    return distance

def l1_distance(prediction, groundtruth):
    """
    Compute the L1 (Manhattan) distance between two vectors.

    Parameters
    ----------
    prediction : array_like
        Predicted feature vector or histogram.
    groundtruth : array_like
        Ground-truth feature vector or histogram.

    Returns
    -------
    float
        The L1 distance between `prediction` and `groundtruth`.
    """
    distance = groundtruth - prediction
    distance = np.abs(distance)
    distance = np.sum(distance)
    return distance

def x2_distance(prediction, groundtruth):
    """
    Compute the Chi-squared (χ²) distance between two histograms.

    Parameters
    ----------
    prediction : array_like
        Predicted histogram, converted internally to `float64`.
    groundtruth : array_like
        Ground-truth histogram, converted internally to `float64`.

    Returns
    -------
    float
        The Chi-squared distance between `prediction` and `groundtruth`.
    """
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("Histograms must have the same length")

    num = (a - b) ** 2
    den = a + b
    term = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return float(term.sum())

def hist_intersection(prediction, groundtruth):
    """
    Compute the histogram intersection distance between two histograms.

    Parameters
    ----------
    prediction : array_like
        Predicted histogram.
    groundtruth : array_like
        Ground-truth histogram.

    Returns
    -------
    float
        The histogram intersection *distance* (1 - similarity) between
        `prediction` and `groundtruth`. The value ranges between 0 and 1.
    """
    distance = np.min([groundtruth, prediction], axis=0)
    distance = np.sum(distance)
    return 1 - distance

def hellinger_kernel(prediction, groundtruth):
    """
    Compute the Hellinger distance between two normalized histograms.

    Parameters
    ----------
    prediction : array_like
        Predicted histogram (should be normalized to sum to 1).
    groundtruth : array_like
        Ground-truth histogram (should be normalized to sum to 1).

    Returns
    -------
    float
        The Hellinger distance between `prediction` and `groundtruth`.
        The value lies in the range [0, 1].
    """
    distance = groundtruth * prediction
    distance = np.sqrt(distance)
    distance = np.sum(distance)
    return np.sqrt(max(0.0, 1.0 - distance))

def canberra_distance(prediction, groundtruth):
    """
    Compute the Canberra distance between two vectors.

    The Canberra distance is a weighted version of the L1 distance that
    normalizes each absolute difference by the sum of the absolute values
    of the corresponding elements. It is particularly well-suited for
    comparing sparse vectors or histograms containing zeros.

    Parameters
    ----------
    prediction : array_like
        Predicted feature vector or histogram.
    groundtruth : array_like
        Ground-truth feature vector or histogram.

    Returns
    -------
    float
        The Canberra distance between `prediction` and `groundtruth`.
    """
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    num = np.abs(a - b)
    den = np.abs(a) + np.abs(b) + 1e-12
    return float(np.sum(num / den))

if __name__ == "__main__":
    generator = np.random.default_rng()
    prediction = generator.standard_normal(size=[512, 1])
    groundtruth = generator.standard_normal(size=[512, 1])

    dist = euclidean_distance(prediction, groundtruth)
    print(f"Euclidean distance: {dist}")

    dist = l1_distance(prediction, groundtruth)
    print(f"L1 distance: {dist}")

    dist = x2_distance(prediction, groundtruth)
    print(f"X2 distance: {dist}")

    dist = hist_intersection(prediction, groundtruth)
    print(f"Histogram intersection: {dist}")

    dist = hellinger_kernel(prediction, groundtruth)
    print(f"Hellinger kernel: {dist}")