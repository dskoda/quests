import math

import numba as nb
import numpy as np

from .matrix import cdist
from .matrix import norm
from .matrix import sumexp

DEFAULT_BANDWIDTH = 0.015
DEFAULT_BATCH = 10000


@nb.njit(fastmath=True)
def perfect_entropy(
    x: np.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
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
    x: np.ndarray,
    y: np.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
):
    """Computes the delta entropy of a dataset `x` using the dataset
        `y` as a reference. The function uses a batch distance
        calculation. This is necessary because the full distance matrix
        often does not fit in the memory for a big dataset. This function
        can be SLOW, despite the optimization of the computation, as it
        does not approximate the results.

    Arguments:
        x (np.ndarray): an (M, d) matrix with the descriptors
        y (np.ndarray): an (N, d) matrix with the descriptors
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropies (np.ndarray): a (M,) vector containing all 
            differential entropies of `x` computed with respect to `y`.
    """
    M = x.shape[0]
    max_step_x = math.ceil(M / batch_size)

    N = y.shape[0]
    max_step_y = math.ceil(N / batch_size)

    # precomputing the norms saves us some time
    norm_x = norm(x)
    norm_y = norm(y)

    # variables that are going to store the results
    entropies = np.empty(M, dtype=x.dtype)

    # loops over rows and columns to compute the
    # distance matrix without keeping it entirely
    # in the memory
    for step_x in nb.prange(0, max_step_x):
        i = step_x * batch_size
        imax = min(i + batch_size, M)
        x_batch = x[i:imax]
        x_batch_norm = norm_x[i:imax]

        # loops over all columns in batches to prevent memory overflow
        for step_y in range(0, max_step_y):
            j = step_y * batch_size
            jmax = min(j + batch_size, N)
            y_batch = y[j:jmax]
            y_batch_norm = norm_y[j:jmax]

            d = cdist(x_batch, y_batch, x_batch_norm, y_batch_norm)
            z = d / h
            # p is the estimated probability distribution for the batch
            p = sumexp(-0.5 * (z**2))

            for j in range(i, imax):
                entropies[j] += p[j - i]

        # after summing everything, we take the log
        for j in range(i, imax):
            entropies[j] = math.log(entropies[j])

    return entropies
