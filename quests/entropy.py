import math

import numba as nb
import numpy as np

from .matrix import cdist
from .matrix import norm
from .matrix import sumexp
from .geometry import cutoff_fn

DEFAULT_BANDWIDTH = 0.015
DEFAULT_BATCH = 20000


@nb.njit(fastmath=True, cache=True)
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
    p_x = kernel_sum(x, x, h=h, batch_size=batch_size)

    # normalizes the p(x) prior to the log for numerical stability
    for j in range(N):
        p_x[j] = math.log(p_x[j] / N)

    return -np.mean(p_x)


@nb.njit(fastmath=True, cache=True)
def delta_entropy(
    x: np.ndarray,
    y: np.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
):
    """Computes the differential entropy of a dataset `x` using the dataset
        `y` as reference. This function can be SLOW, despite the optimization 
        of the computation, as it does not approximate the results.

    Arguments:
        x (np.ndarray): an (N, d) matrix with the descriptors of the test set
        y (np.ndarray): an (N, d) matrix with the descriptors of the reference
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropy (float): entropy of the dataset given by `x`.
    """
    N = x.shape[0]
    p_x = kernel_sum(x, y, h=h, batch_size=batch_size)

    for j in range(N):
        p_x[j] = -math.log(p_x[j])

    return p_x


@nb.njit(fastmath=True, parallel=True, cache=True)
def kernel_sum(
    x: np.ndarray,
    y: np.ndarray,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
):
    """Computes the kernel matrix K_ij for the descriptors x_i and y_j.
        Because the entire matrix cannot fit in the memory, this function
        automatically applies the kernel and sums the results, essentially
        recovering the probability distribution p(x) up to a normalization
        constant.

    Arguments:
        x (np.ndarray): an (M, d) matrix with the test descriptors
        y (np.ndarray): an (N, d) matrix with the reference descriptors
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        ki (np.ndarray): a (M,) vector containing the probability of x_i
            given `y`
    """
    M = x.shape[0]
    max_step_x = math.ceil(M / batch_size)

    N = y.shape[0]
    max_step_y = math.ceil(N / batch_size)

    # precomputing the norms saves us some time
    norm_x = norm(x)
    norm_y = norm(y)

    # variables that are going to store the results
    p_x = np.zeros(M, dtype=x.dtype)

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

            # computing the estimated probability distribution for the batch
            z = cdist(x_batch, y_batch, x_batch_norm, y_batch_norm)
            z = z / h
            z = sumexp(-0.5 * (z**2))

            for k in range(i, imax):
                p_x[k] = p_x[k] + z[k - i]

    return p_x


@nb.njit(fastmath=True, cache=True)
def get_bandwidth(volume: float, method: str = "gaussian"):
    """Estimate of the bandwidth based on the dependence 
        of the entropy w.r.t. volume per atom (or density).
        The hard-coded parameters here were shown to work
        well for some systems.

    Arguments:
        volume (float): volume per atom (in Å^3/atom)

    Returns:
        bandwidth (float)
    """
    if method == "gaussian":
        z = volume / 10.896
        return 0.0897141 * np.exp(-0.5 * z ** 2) + 0.0119417

    if method == "cutoff":
        z = np.power(np.log(volume), 2)
        return 0.086164 * cutoff_fn(z, 11.61172) + 0.016
