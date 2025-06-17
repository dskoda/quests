import math

import torch
import numpy as np
from typing import Union, List

from .matrix import cdist, norm, sum_positive, sumexp, wsumexp

DEFAULT_BANDWIDTH = 0.015
DEFAULT_BATCH = 20000
DEFAULT_UQ_NBRS = 3
DEFAULT_GRAPH_NBRS = 10


def perfect_entropy(
    x: np.ndarray,
    h: Union[float, List[float]] = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    device: str = "cpu",
):
    """Deprecated. Please use `entropy`.

    Computes the perfect entropy of a dataset using a batch distance
        calculation. This is necessary because the full distance matrix
        often does not fit in the memory for a big dataset. This function
        can be SLOW, despite the optimization of the computation, as it
        does not approximate the results.

    Arguments:
        x (np.ndarray): an (N, d) matrix with the descriptors
        h (int or np.nadarray): bandwidth (value / vector) for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropy (float): entropy of the dataset given by `x`.
            or (np.ndarray): if 'h' is a vector
    """
    return entropy(x, h, batch_size, device=device)


def entropy(
    x: torch.tensor,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    device: str = "cpu",
):
    """Computes the perfect entropy of a dataset using a batch distance
        calculation. This is necessary because the full distance matrix
        often does not fit in the memory for a big dataset. This function
        can be SLOW, despite the optimization of the computation, as it
        does not approximate the results.

    Arguments:
        x (torch.tensor): an (N, d) matrix with the descriptors
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropy (float): entropy of the dataset given by `x`.
    """
    N = x.shape[0]
    p_x = kernel_sum(x, x, h=h, batch_size=batch_size, device=device)

    # normalizes the p(x) prior to the log for numerical stability
    p_x = torch.log(p_x / N)

    return -torch.mean(p_x)


def delta_entropy(
    x: torch.tensor,
    y: torch.tensor,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    device: str = "cpu",
):
    """Computes the differential entropy of a dataset `x` using the dataset
        `y` as reference. This function can be SLOW, despite the optimization
        of the computation, as it does not approximate the results.

    Arguments:
        x (torch.tensor): an (N, d) matrix with the descriptors of the test set
        y (torch.tensor): an (N, d) matrix with the descriptors of the reference
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropy (float): entropy of the dataset given by `x`.
    """
    p_x = kernel_sum(x, y, h=h, batch_size=batch_size, device=device)
    return -torch.log(p_x)


def diversity(
    x: torch.tensor,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    device: str = "cpu",
):
    """Computes the diversity of a dataset `x` by assuming a sum over the
        inverse p(x). This approximates the number of unique data points
        in the system, as Kij >= 1 for a kernel matrix of a dataset.

    Arguments:
        x (torch.tensor): an (N, d) matrix with the descriptors of the dataset
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        entropy (float): entropy of the dataset given by `x`.
    """
    p_x = kernel_sum(x, x, h=h, batch_size=batch_size, device=device)
    return torch.sum(1 / p_x)


def kernel_sum(
    x: torch.tensor,
    y: torch.tensor,
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    device: str = "cpu",
):
    """Computes the kernel matrix K_ij for the descriptors x_i and y_j.
        Because the entire matrix cannot fit in the memory, this function
        automatically applies the kernel and sums the results, essentially
        recovering the probability distribution p(x) up to a normalization
        constant.

    Arguments:
        x (torch.tensor): an (M, d) matrix with the test descriptors
        y (torch.tensor): an (N, d) matrix with the reference descriptors
        h (int): bandwidth for the Gaussian kernel
        batch_size (int): maximum batch size to consider when
            performing a distance calculation.

    Returns:
        ki (torch.tensor): a (M,) vector containing the probability of x_i
            given `y`
    """
    device = torch.device(device)

    M = x.shape[0]
    max_step_x = math.ceil(M / batch_size)

    N = y.shape[0]
    max_step_y = math.ceil(N / batch_size)

    # precomputing the norms saves us some time
    norm_x = norm(x)
    norm_y = norm(y)

    # variables that are going to store the results
    p_x = torch.zeros(M, dtype=x.dtype)

    # loops over rows and columns
    for step_x in range(0, max_step_x):
        i = step_x * batch_size
        imax = min(i + batch_size, M)
        x_batch = x[i:imax].to(device)
        x_batch_norm = norm_x[i:imax].to(device)

        # loops over all columns in batches to prevent memory overflow
        for step_y in range(0, max_step_y):
            j = step_y * batch_size
            jmax = min(j + batch_size, N)
            y_batch = y[j:jmax].to(device)
            y_batch_norm = norm_y[j:jmax].to(device)

            # computing the estimated probability distribution for the batch
            z = cdist(x_batch, y_batch, x_batch_norm, y_batch_norm)
            z = z / h
            z = sumexp(-0.5 * (z**2))

            p_x[i:imax] = p_x[i:imax] + z.to("cpu")

    return p_x
