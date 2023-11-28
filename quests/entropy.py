import math

import numba as nb
import numpy as np

from .matrix import cdist
from .matrix import logsumexp
from .matrix import norm

DEFAULT_BANDWIDTH = 0.015
DEFAULT_BATCH = 2000
DEFAULT_BATCH_REF = 100000


@nb.njit(fastmath=True)
def perfect_entropy(
    x: np.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    batch_size_ref: int = DEFAULT_BATCH_REF,
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
    entropies = delta_entropy(
        x, x, h=h, batch_size=batch_size, batch_size_ref=batch_size_ref
    )

    return np.log(N) - np.mean(entropies)


@nb.njit(fastmath=True, parallel=True)
def delta_entropy(
    x: np.ndarray,
    ref: np.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    batch_size_ref: int = DEFAULT_BATCH_REF,
):
    """Computes the delta entropy of a dataset `x` using the dataset
        `ref` as a reference. The function uses a batch distance
        calculation. This is necessary because the full distance matrix
        often does not fit in the memory for a big dataset. This function
        can be SLOW, despite the optimization of the computation, as it
        does not approximate the results.

    Arguments:
        x (np.ndarray): an (N, d) matrix with the descriptors
        ref (np.ndarray): an (M, d) matrix with the descriptors
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.
        ref_batch_size (int): maximum batch size to consider when
            looping over columns

    Returns:
        entropies (np.ndarray): a (M,) vector containing all entropies of
            `x` computed with respect to `ref`.
    """
    N = x.shape[0]
    max_step_x = math.ceil(N / batch_size)

    M = ref.shape[0]
    max_step_ref = math.ceil(M / batch_size_ref)

    # precomputing the norms saves us some time
    norm_ref = norm(ref)
    norm_x = norm(x)
    entropies = np.empty(N, dtype=x.dtype)

    # loops over rows and columns to compute the
    # distance matrix without keeping it entirely
    # in the memory
    for step_x in nb.prange(0, max_step_x):
        i = step_x * batch_size
        imax = min(i + batch_size, N)
        n_rows = imax - i
        x_batch = x[i:imax]
        x_batch_norm = norm_x[i:imax]

        # loops over all columns in batches to prevent memory overflow
        d = np.empty((n_rows, M))
        for step_ref in range(0, max_step_ref):
            j = step_ref * batch_size_ref
            jmax = min(j + batch_size_ref, M)
            ref_batch = ref[j:jmax]
            ref_batch_norm = norm_ref[j:jmax]

            _dists = cdist(x_batch, ref_batch, x_batch_norm, ref_batch_norm)
            d[:, j:jmax] = _dists

        # computation of the entropy
        z = d / h
        entropy = logsumexp(-0.5 * (z**2))

        for j in range(i, imax):
            entropies[j] = entropy[j - i]

    return entropies
