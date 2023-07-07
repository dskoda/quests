import numpy as np
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.stats import epps_singleton_2samp
from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance


def kl_divergence(x, y):
    return entropy(x, y)


def js_divergence(x, y):
    m = 0.5 * (x + y)
    return 0.5 * kl_divergence(x, m) + 0.5 * kl_divergence(y, m)


def hellinger_distance(x, y):
    return np.sqrt(0.5 * np.sum((np.sqrt(x) - np.sqrt(y)) ** 2))


def total_variation_distance(x, y):
    return 0.5 * np.abs(x - y).sum()


def ks_test(x, y):
    return ks_2samp(x, y).statistic


def cvm_test(x, y):
    return epps_singleton_2samp(x, y).statistic


METRICS = {
    "kl": kl_divergence,
    "js": js_divergence,
    "hellinger": hellinger_distance,
    "tv": total_variation_distance,
    "ks_test": ks_test,
    "cvm_test": cvm_test,
}


def compare(
    r1: np.ndarray,
    d1: np.ndarray,
    r2: np.ndarray,
    d2: np.ndarray,
    metric="emd",
):
    metric = metric.lower()

    u1, v1 = 1 / r1, 1 / d1
    u2, v2 = 1 / r2, 1 / d2

    if metric in ["emd", "scaled_emd"]:
        rdist = wasserstein_distance(r1, r2)
        ddist = wasserstein_distance(d1, d2)

    elif metric in ["inv_emd", "inv_scaled_emd"]:
        rdist = wasserstein_distance(u1, u2)
        ddist = wasserstein_distance(v1, v2)

    elif metric in ["inv_sq_emd", "inv_sq_scaled_emd"]:
        rdist = wasserstein_distance(u1**2, u2**2)
        ddist = wasserstein_distance(v1**2, v2**2)

    elif metric in ["weighted_emd", "weighted_scaled_emd"]:
        rdist = wasserstein_distance(r1, r2, u1, u2)
        ddist = wasserstein_distance(d1, d2, v1, v2)

    elif metric in ["weighted_sq_emd", "weighted_sq_scaled_emd"]:
        rdist = wasserstein_distance(r1, r2, u1**2, u2**2)
        ddist = wasserstein_distance(d1, d2, v1**2, v2**2)

    elif metric in ["chebyshev"]:
        rdist = np.linalg.norm(r1 - r2, ord=np.inf)
        ddist = np.linalg.norm(d1 - d2, ord=np.inf)

    elif metric in ["median"]:
        rdist = np.median(np.abs(r1 - r2))
        ddist = np.median(np.abs(d1 - d2))

    elif metric in ["mean"]:
        rdist = np.mean(np.abs(r1 - r2))
        ddist = np.mean(np.abs(d1 - d2))

    elif metric in ["inv_mean"]:
        rdist = np.mean(np.abs(u1 - u2))
        ddist = np.mean(np.abs(v1 - v2))

    elif metric in ["inv_chebyshev"]:
        rdist = np.linalg.norm(u1 - u2, ord=np.inf)
        ddist = np.linalg.norm(v1 - v2, ord=np.inf)

    elif metric in ["frobenius"]:
        rdist = np.linalg.norm(r1 - r2)
        ddist = np.linalg.norm(d1 - d2)

    elif metric == "inv_js":
        fn = METRICS["js"]
        rdist = fn(u1, u2)
        ddist = fn(v1, v2)

    elif metric in METRICS.keys():
        fn = METRICS[metric]
        rdist = fn(r1, r2)
        ddist = fn(d1, d2)

    if "scaled" in metric:
        return (rdist * len(r1) + ddist * len(d1)) / (len(r1) + len(d1))

    return rdist + ddist


def normalize(_r, _d):
    """Normalize the descriptors by rescaling all by the first distance."""
    if len(_r.shape) == 1:
        rnorm = _r / _r[0]
        dnorm = _d / _d[0]

    elif len(_r.shape) == 2:
        rnorm = _r / _r[:, 0].reshape(-1, 1)
        dnorm = _d / _d[:, 0].reshape(-1, 1)

    return rnorm, dnorm


def normalize_distribution(_r, _d):
    """Normalize the descriptors by making the area under
    the curve equal to 1.
    """
    if len(_r.shape) == 1:
        rsum = _r.sum()
        dsum = _d.sum()

    elif len(_r.shape) == 2:
        rsum = _r.sum(1).reshape(-1, 1)
        dsum = _d.sum(1).reshape(-1, 1)

    return _r / rsum, _d / dsum


def compare_matrices(
    r1: np.ndarray,
    d1: np.ndarray,
    r2: np.ndarray,
    d2: np.ndarray,
    metric="emd",
    norm: bool = False,
):
    results = []

    for _r1, _d1 in zip(r1, d1):
        if norm:
            _r1, _d1 = normalize(_r1, _d1)

        results_1 = []
        for _r2, _d2 in zip(r2, d2):
            if norm:
                _r2, _d2 = normalize(_r2, _d2)

            dist = compare(_r1, _d1, _r2, _d2, metric=metric)
            results_1.append(dist)

        results.append(results_1)

    return np.array(results)
