import itertools

import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance


def js_divergence(x, y):
    m = 0.5 * (x + y)
    return 0.5 * entropy(x, m) + 0.5 * entropy(y, m)


def compare(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    metric="emd",
):
    metric = metric.lower()

    if metric == "emd":
        rdist = wasserstein_distance(x1, y1)
        ddist = wasserstein_distance(x2, y2)

    elif metric == "chebyshev":
        rdist = np.linalg.norm(x1 - y1, ord=np.inf)
        ddist = np.linalg.norm(x2 - y2, ord=np.inf)

    elif metric == "median":
        rdist = np.median(np.abs(x1 - y1))
        ddist = np.median(np.abs(x2 - y2))

    elif metric == "mean":
        rdist = np.mean(np.abs(x1 - y1))
        ddist = np.mean(np.abs(x2 - y2))

    elif metric == "frobenius":
        rdist = np.linalg.norm(x1 - y1)
        ddist = np.linalg.norm(x2 - y2)

    elif metric == "js":
        rdist = js_divergence(x1, y1)
        ddist = js_divergence(x2, y2)

    else:
        raise ValueError(f"metric {metric} not found")

    return rdist + ddist


def compare_matrices(
    x1: np.ndarray,
    x2: np.ndarray,
    y1: np.ndarray,
    y2: np.ndarray,
    metric="emd",
):
    metric = metric.lower()
    M, N = len(x1), len(y1)

    if metric in ["emd", "js"]:
        dist = [
            compare(x1[i], x2[i], y1[j], y2[j], metric=metric)
            for i, j in itertools.product(range(M), range(N))
        ]
        return np.array(dist).reshape(M, N)

    diff1 = x1.reshape(M, np.newaxis, -1) - y1.reshape(N, -1, np.newaxis)
    diff2 = x2.reshape(M, np.newaxis, -1) - y2.reshape(N, -1, np.newaxis)

    if metric == "chebyshev":
        rdist = np.linalg.norm(diff1, ord=np.inf, axis=-1)
        ddist = np.linalg.norm(diff2, ord=np.inf, axis=-1)

    elif metric == "median":
        rdist = np.median(np.abs(diff1), axis=-1)
        ddist = np.median(np.abs(diff2), axis=-1)

    elif metric == "mean":
        rdist = np.mean(np.abs(diff1), axis=-1)
        ddist = np.mean(np.abs(diff2), axis=-1)

    elif metric == "frobenius":
        rdist = np.linalg.norm(diff1, axis=-1)
        ddist = np.linalg.norm(diff2, axis=-1)

    return rdist + ddist
