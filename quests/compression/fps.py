import itertools
from typing import List

import numpy as np
from quests.matrix import cdist


def select_fps_greedy(dm: np.ndarray, entropies: np.ndarray) -> int:
    return dm.min(0).argmax()


def select_msc_greedy(dm: np.ndarray, entropies: np.ndarray) -> int:
    return (dm.mean(0) * np.array(entropies)).argmax()


def select_msc_weighted(dm: np.ndarray, entropies: np.ndarray, weight=1.0) -> int:
    return (dm.mean(0) + weight * np.array(entropies)).argmax()


SELECT_FNS = {
    "fps": select_fps_greedy,
    "msc": select_msc_greedy,
    "mscw": select_msc_weighted,
}


def fps(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int, method: str = "fps"
) -> List[int]:
    # select the sampling strategy
    assert method in SELECT_FNS, f"Method {method} not supported"
    select_fn = SELECT_FNS[method]

    # setting up the calculation: the initial data point is selected to be
    # the one with highest entropy (most diversity of environments)
    remaining = list(range(len(descriptors)))
    next_i = entropies.argmax()
    compressed = [next_i]
    remaining.pop(next_i)
    entropies = entropies.tolist()
    entropies.pop(next_i)

    # now, we sample the dataset until convergence
    matrix = []
    size = len(descriptors) if size >= len(descriptors) else size
    while len(compressed) < size:
        # computes the distances towards the latest sampled configuration
        x = descriptors[next_i]
        dists = [np.min(cdist(x, descriptors[i])) for i in remaining]
        matrix.append(dists)

        # creates a temporary distance matrix to do the farthest point sampling
        dm = np.array(matrix).reshape(len(compressed), len(remaining))

        # select the element that has the largest distance towards all the existing
        # points in the compressed set AND has high entropy
        selected = select_fn(dm, entropies)

        # update the loop and the set of compressed data
        next_i = remaining.pop(selected)
        entropies.pop(selected)
        compressed.append(next_i)

        # delete the distances to reuse the matrix in the next iteration
        for dlist in matrix:
            dlist.pop(selected)

    return compressed
