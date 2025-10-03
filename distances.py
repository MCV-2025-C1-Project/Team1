import numpy as np

# ----------------------------
# Basic metric helpers
# ----------------------------

def _normalize_prob(h):
    """L1-normalize a nonnegative histogram to sum to 1 (adds small epsilon)."""
    h = np.asarray(h, dtype=np.float64).ravel()
    h = np.maximum(h, 0.0)
    s = h.sum()
    if s <= 0.0:
        # fallback: uniform if vector is all zeros or negative
        h = np.ones_like(h, dtype=np.float64)
        s = h.sum()
    return h / (s + 1e-12)

def _kl_divergence(p, q):
    """
    KL(p || q) with standard safeguards.
    Interprets 0*log(0/x) = 0 by convention.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))

# ----------------------------
# Distances / similarities (lower is better)
# ----------------------------

def euclidean_distance(prediction, groundtruth):
    """L2 distance."""
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    return float(np.linalg.norm(b - a))

def l1_distance(prediction, groundtruth):
    """L1 (Manhattan) distance."""
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    return float(np.sum(np.abs(b - a)))

def x2_distance(prediction, groundtruth):
    """
    Chi-square distance for histograms (not necessarily probabilities):
    sum( (a-b)^2 / (a+b) ), with safe divide when a+b == 0.
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
    Histogram intersection distance = 1 - sum(min(p, q)).
    Assumes p and q are probability histograms (sum to 1).
    """
    p = np.asarray(prediction, dtype=np.float64).ravel()
    q = np.asarray(groundtruth, dtype=np.float64).ravel()
    inter = np.sum(np.minimum(p, q))
    return float(1.0 - inter)

def hellinger_kernel(prediction, groundtruth):
    """
    Hellinger distance using the Bhattacharyya kernel form:
    H(p, q) = sqrt( max(0, 1 - sum(sqrt(p*q))) ), for probabilities.
    """
    p = np.asarray(prediction, dtype=np.float64).ravel()
    q = np.asarray(groundtruth, dtype=np.float64).ravel()
    bc = float(np.sum(np.sqrt(p * q)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))

def cosine_distance(prediction, groundtruth):
    """Cosine distance = 1 - cosine similarity."""
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    an = np.linalg.norm(a) + 1e-12
    bn = np.linalg.norm(b) + 1e-12
    cos_sim = float(np.dot(a, b) / (an * bn))
    return float(1.0 - cos_sim)

def chebyshev_distance(prediction, groundtruth):
    """Chebyshev (L∞) distance."""
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    return float(np.max(np.abs(a - b)))

def canberra_distance(prediction, groundtruth):
    """Canberra distance (well-defined for zeros)."""
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    num = np.abs(a - b)
    den = np.abs(a) + np.abs(b) + 1e-12
    return float(np.sum(num / den))

def braycurtis_distance(prediction, groundtruth):
    """Bray–Curtis distance in [0, 1]."""
    a = np.asarray(prediction, dtype=np.float64).ravel()
    b = np.asarray(groundtruth, dtype=np.float64).ravel()
    num = np.sum(np.abs(a - b))
    den = np.sum(np.abs(a) + np.abs(b)) + 1e-12
    return float(num / den)

def bhattacharyya_distance(prediction, groundtruth):
    """
    Bhattacharyya-based distance for probabilities:
    d = sqrt( max(0, 1 - BC) ), where BC = sum(sqrt(p*q)).
    """
    p = np.asarray(prediction, dtype=np.float64).ravel()
    q = np.asarray(groundtruth, dtype=np.float64).ravel()
    bc = float(np.sum(np.sqrt(p * q)))
    return float(np.sqrt(max(0.0, 1.0 - bc)))

def js_divergence(prediction, groundtruth):
    """
    Jensen–Shannon divergence (symmetric, finite).
    Returns JS in nats, bounded in [0, ln(2)] for probabilities.
    """
    p = np.asarray(prediction, dtype=np.float64).ravel()
    q = np.asarray(groundtruth, dtype=np.float64).ravel()
    m = 0.5 * (p + q)
    return float(0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m))

# ----------------------------
# Demo
# ----------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # arbitrary real vectors for general metrics
    a = rng.standard_normal(size=[512, 1])
    b = rng.standard_normal(size=[512, 1])

    print(f"Euclidean distance: {euclidean_distance(a, b)}")
    print(f"L1 distance: {l1_distance(a, b)}")
    print(f"X2 distance: {x2_distance(a, b)}")
    print(f"Cosine distance: {cosine_distance(a, b)}")
    print(f"Chebyshev distance: {chebyshev_distance(a, b)}")
    print(f"Canberra distance: {canberra_distance(a, b)}")
    print(f"Bray–Curtis distance: {braycurtis_distance(a, b)}")

    # nonnegative histograms -> probabilities for prob-based metrics
    p = np.abs(rng.standard_normal(size=[512, 1])); p = _normalize_prob(p)
    q = np.abs(rng.standard_normal(size=[512, 1])); q = _normalize_prob(q)

    print(f"Histogram intersection: {hist_intersection(p, q)}")
    print(f"Hellinger kernel: {hellinger_kernel(p, q)}")
    print(f"Bhattacharyya distance: {bhattacharyya_distance(p, q)}")
    print(f"JS divergence: {js_divergence(p, q)}")
