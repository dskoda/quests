import itertools
import numpy as np
from typing import List
from scipy.spatial.distance import cdist


def fps(descriptors: List[np.ndarray], entropies: np.ndarray, size: int):
    # setting up the calculation
    remaining = list(range(len(descriptors)))
    next_i = entropies.argmax()
    compressed = [next_i]
    remaining.pop(next_i)
    entropies = entropies.tolist()
    entropies.pop(next_i)

    matrix = []
    while len(compressed) < size:
        # computes the distances towards the latest sampled configuration
        x = descriptors[next_i]
        dists = [
            np.min(cdist(x, descriptors[i]))
            for i in remaining
        ]
        matrix.append(dists)

        # creates a temporary distance matrix to do the farthest point sampling
        dm = np.array(matrix).reshape(len(compressed), len(remaining))

        # select the element that has the largest distance towards all the existing
        # points in the compressed set AND has high entropy
        selected = (dm.min(0) * np.array(entropies)).argmax()

        # update the loop and the set of compressed data
        next_i = remaining.pop(selected)
        entropies.pop(selected)
        compressed.append(next_i)

        # delete the distances to reuse the matrix in the next iteration
        for dlist in matrix:
            dlist.pop(selected)

    return compressed
