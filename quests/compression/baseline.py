import itertools
import random
from typing import List

import numpy as np


def random_sample(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int
) -> List[int]:
    indices = np.arange(len(descriptors)).tolist()
    selected = random.sample(indices, size)
    return selected


def mean_fps(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int
) -> List[int]:
    import fpsample

    avg_descriptors = np.array([x.mean(0) for x in descriptors])
    selected = fpsample.bucket_fps_kdtree_sampling(avg_descriptors, size)
    return selected


# TODO: Ben, please implement
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
def dbscan_sample(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int
) -> List[int]:
    raise NotImplementedError
