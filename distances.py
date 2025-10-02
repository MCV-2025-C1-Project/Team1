import numpy as np

def euclidean_distance(prediction, groundtruth):
    distance = groundtruth - prediction
    distance = distance * distance
    distance = np.sum(distance)
    distance = np.sqrt(distance)
    return distance

def l1_distance(prediction, groundtruth):
    distance = groundtruth - prediction
    distance = np.abs(distance)
    distance = np.sum(distance)
    return distance

def x2_distance(prediction, groundtruth):
    numerator = groundtruth - prediction
    numerator = numerator * numerator
    denominator = groundtruth + prediction
    distance = numerator / denominator
    distance = np.sum(distance)
    return distance

def hist_intersection(prediction, groundtruth):
    distance = np.min([groundtruth, prediction], axis=0)
    distance = np.sum(distance)
    return distance

def hellinger_kernel(prediction, groundtruth):
    distance = groundtruth * prediction
    distance = np.sqrt(distance)
    distance = np.sum(distance)
    return distance

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
