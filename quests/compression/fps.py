from typing import List

import numpy as np
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH, kernel_sum
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
        dists = [np.mean(cdist(x, descriptors[i])) for i in remaining]
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


def msc(
    descriptors: List[np.ndarray],
    entropies: np.ndarray,
    size: int,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
) -> List[int]:
    """Compresses the dataset using a special algorithm that accounts for
    the novelty of each environment in each structure to the compressed
    dataset. Each comparison allows us to select the most novel and diverse
    environment at once in the entire dataset. The compression method is
    written to avoid re-doing calculations.
    """

    # setting up the calculation: the initial data point is selected to be
    # the one with highest entropy (most diversity of environments)
    remaining = list(range(len(descriptors)))
    next_i = entropies.argmax()
    compressed = [next_i]
    remaining.pop(next_i)
    entropies = entropies.tolist()
    entropies.pop(next_i)

    # we will use the pre-computed descriptors
    remaining_x = np.concatenate([descriptors[i] for i in remaining])

    # and to save computation, we will keep this matrix as an accumulator
    # whose shape will change as we continue compressing the dataset
    remaining_kernels = np.zeros(len(remaining_x))

    size = len(descriptors) if size >= len(descriptors) else size
    while len(compressed) < size:
        # to avoid recomputing kernel matrices, we will perform this only once
        # per reference structure
        last_x = descriptors[next_i]

        # then, we will create some matrices to make it easier for us to
        # write the slices to come
        remaining_natoms = np.array([len(descriptors[i]) for i in remaining])

        # this creates an array that explains which environments belong to
        # each index in `remaining`
        remaining_idx = np.concatenate(
            [np.full(n, fill_value=i) for i, n in enumerate(remaining_natoms)]
        )

        # compute the kernel between the remaining environments and the last one
        remaining_kernels += kernel_sum(remaining_x, last_x)

        # now, define the value of the kernel in a greedy way, saying that the
        # kernel of a structure is equal to the furthest environment towards
        # the entire dataset
        per_struct_kernels = np.array(
            [remaining_kernels[remaining_idx == n].min() for n in range(len(remaining))]
        )

        # select the environment that both maximizes the dH = -np.log(K) and
        # is diverse enough that the entropy is large
        per_struct_dH = -np.log(per_struct_kernels)
        selected = (per_struct_dH + np.array(entropies)).argmax()

        # update the loop and the set of compressed data
        next_i = remaining.pop(selected)
        entropies.pop(selected)
        remaining_x = remaining_x[remaining_idx != selected]
        remaining_kernels = remaining_kernels[remaining_idx != selected]
        compressed.append(next_i)

    return compressed
