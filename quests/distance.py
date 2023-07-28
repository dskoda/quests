import itertools
from typing import List
import multiprocess as mp

import numpy as np
import pandas as pd
from ase import Atoms
from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

from .descriptor import QUESTS


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
    metric="euclidean",
):
    metric = metric.lower()
    M, N = len(x1), len(y1)

    if metric in ["emd", "js"]:
        dist = [
            compare(x1[i], x2[i], y1[j], y2[j], metric=metric)
            for i, j in itertools.product(range(M), range(N))
        ]
        return np.array(dist).reshape(M, N)

    diff1 = x1.reshape(M, 1, -1) - y1.reshape(1, N, -1)
    diff2 = x2.reshape(M, 1, -1) - y2.reshape(1, N, -1)

    if metric == "chebyshev":
        rdist = np.linalg.norm(diff1, ord=np.inf, axis=-1)
        ddist = np.linalg.norm(diff2, ord=np.inf, axis=-1)

    elif metric == "median":
        rdist = np.median(np.abs(diff1), axis=-1)
        ddist = np.median(np.abs(diff2), axis=-1)

    elif metric == "mean":
        rdist = np.mean(np.abs(diff1), axis=-1)
        ddist = np.mean(np.abs(diff2), axis=-1)

    elif metric in ["euclidean", "frobenius"]:
        rdist = np.linalg.norm(diff1, axis=-1)
        ddist = np.linalg.norm(diff2, axis=-1)

    return rdist + ddist


def compare_datasets(
    dset1: List[Atoms],
    dset2: List[Atoms],
    q: QUESTS,
    metric: str = "euclidean",
    nprocs: int = 1,
) -> pd.DataFrame:
    """Compares different datasets according to the QUESTS descriptor.

    Arguments:
    ----------
        dset1 (List[Atoms]): first dataset to analyze
        dset2 (List[Atoms]): second dataset to analyze
        q (QUESTS): object that creates descriptors
        metric (str): name of the metric to use when comparing descriptors
        nprocs (int): number of processors to use when comparing datasets

    Returns:
    --------
        results (pd.DataFrame): dataframe containing statistics of 
            the comparison.
    """
    q1 = [q.get_descriptors(at) for at in dset1]
    q2 = [q.get_descriptors(at) for at in dset2]

    def worker_fn(ij):
        i, j = ij
        x1, x2 = q1[i]
        y1, y2 = q2[j]
        dm = compare_matrices(x1, x2, y1, y2, metric=metric)
        return {
            "index1": i,
            "index2": j,
            "min": dm.min(),
            "max": dm.max(),
            "mean": dm.mean(),
            "std": dm.std(),
            "q1": np.percentile(dm, 25),
            "q3": np.percentile(dm, 75),
        }

    results = []
    iterator = itertools.product(range(len(q1)), range(len(q2)))

    if nprocs == 1:
        for ij in iterator:
            result = worker_fn(ij)
            results.append(result)

    else:
        p = mp.Pool(nprocs)
        for result in p.imap_unordered(worker_fn, iterator, chunksize=1):
            results.append(result)

    return pd.DataFrame(results)


def batch_distances(x, y, batch_size=2000):
    Nx = x.shape[0]
    Ny = y.shape[0]
    dm = np.zeros((Nx, Ny))  # initialize distance matrix

    for i in range(0, Nx, batch_size):
        for j in range(0, Ny, batch_size):
            # create batch from data
            imax = min(i + batch_size, Nx)
            jmax = min(j + batch_size, Ny)
            batch1 = x[i:imax]
            batch2 = y[j:jmax]

            distances = cdist(batch1, batch2)

            # store distances in matrix
            dm[i:imax, j:jmax] = distances

    return dm


def matmul_distances(x, y):
    xn = np.linalg.norm(x, axis=-1).reshape(-1, 1)
    yn = np.linalg.norm(y, axis=-1, keepdims=True).reshape(1, -1)
    xy = x @ y
    dm  = xn + yn - 2 * xy
    return dm
