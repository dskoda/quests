import math

import numba as nb
import numpy as np

from .matrix import cdist
from .matrix import logsumexp
from .matrix import norm

DEFAULT_BANDWIDTH = 0.015
DEFAULT_BATCH = 2000


@nb.njit(fastmath=True)
def perfect_entropy(
    x: np.ndarray, h: float = DEFAULT_BANDWIDTH, batch_size: int = DEFAULT_BATCH
):
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
    entropies = delta_entropy(x, x, h=h, batch_size=batch_size)

    return np.log(N) - np.mean(entropies)


@nb.njit(fastmath=True, parallel=True)
def delta_entropy(
    ref: np.ndarray,
    test: np.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
):
    """Computes the delta entropy of a dataset `test` using the dataset
        `ref` as a reference. The function uses a batch distance
        calculation. This is necessary because the full distance matrix
        often does not fit in the memory for a big dataset. This function
        can be SLOW, despite the optimization of the computation, as it
        does not approximate the results.

    Arguments:
        ref (np.ndarray): an (N, d) matrix with the descriptors
        test (np.ndarray): an (M, d) matrix with the descriptors
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropies (np.ndarray): a (M,) vector containing all entropies of
            `test` computed with respect to `ref`.
    """
    N = test.shape[0]
    max_step = math.ceil(N / batch_size)

    norm_ref = norm(ref)
    norm_test = norm(test)
    entropies = np.empty(N, dtype=test.dtype)

    for step in nb.prange(0, max_step):
        i = step * batch_size
        imax = min(i + batch_size, N)
        batch = test[i:imax]
        batch_norm = norm_test[i:imax]

        d = cdist(batch, ref, batch_norm, norm_ref)

        # computation of the entropy
        z = d / h
        entropy = logsumexp(-0.5 * (z**2))

        for j in range(i, imax):
            entropies[j] = entropy[j - i]

    return entropies
