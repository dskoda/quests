import math

import numba as nb
import numpy as np

from .matrix import cdist
from .matrix import logsumexp


@nb.njit(fastmath=True, parallel=True)
def perfect_entropy(x: np.ndarray, h: float = 0.015, batch_size: int = 2000):
    """Computes the perfect entropy of a dataset using a batch distance
        calculation. This is necessary because the full distance matrix
        often does not fit in the memory for a big dataset. This function
        can be SLOW, despite the optimization of the computation, as it
        does not approximate the results.

    Arguments:
        x (np.ndarray): an (N, d) matrix with the descriptors
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropy (float): entropy of the dataset given by `x`.
    """
    N = x.shape[0]
    max_step = math.ceil(N / batch_size)

    entropies = np.empty(N, dtype=x.dtype)

    for step in nb.prange(0, max_step):
        i = step * batch_size
        imax = min(i + batch_size, N)
        batch = x[i:imax]

        d = cdist(batch, x)

        # computation of the entropy
        z = d / h
        entropy = logsumexp(-0.5 * (z**2))

        for j in range(i, imax):
            entropies[j] = entropy[j - i]

    return np.log(N) - np.mean(entropies)
