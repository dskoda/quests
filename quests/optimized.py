import math

import numba as nb
import numpy as np


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

        d = cdist_numba(batch, x)

        # computation of the entropy
        z = d / h
        entropy = logsumexp(-0.5 * (z**2))

        for j in range(i, imax):
            entropies[j] = entropy[j - i]

    return np.log(N) - np.mean(entropies)


@nb.njit(fastmath=True)
def logsumexp(X):
    """logsumexp optimized for numba. Can lead to numerical
        instabilities, but it's really fast.

    Arguments:
        X (np.ndarray): an (N, d) matrix with the values. The
            summation will happen over the axis 1.

    Returns:
        logsumexp (np.ndarray): log(sum(exp(X), axis=1))
    """
    result = np.empty(X.shape[0], dtype=X.dtype)
    for i in range(X.shape[0]):
        _sum = 0.0
        for j in range(X.shape[1]):
            _sum += math.exp(X[i, j])

        result[i] = _sum

    return np.log(result)


@nb.njit(fastmath=True)
def cdist_numba(A, B):
    """Optimized distance calculation using numba.

    Arguments:
        A (np.ndarray): an (N, d) matrix with the descriptors
        B (np.ndarray): an (M, d) matrix with the descriptors

    Returns:
        dist (float): entropy of the dataset given by `x`.
    """
    # Computing the dot product
    dist = np.dot(A, B.T)

    # Computing the norm of A
    norm_A = np.empty(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        _sum = 0.0
        for j in range(A.shape[1]):
            _sum += A[i, j] ** 2
        norm_A[i] = _sum

    # Computing the norm of B
    norm_B = np.empty(B.shape[0], dtype=A.dtype)
    for i in range(B.shape[0]):
        _sum = 0.0
        for j in range(B.shape[1]):
            _sum += B[i, j] ** 2
        norm_B[i] = _sum

    # computes the distance using the dot product
    # | a - b | ** 2 = <a, a> + <b, b> - 2<a, b>
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d = -2.0 * dist[i, j] + norm_A[i] + norm_B[j]

            # numerical stability
            if d < 0:
                dist[i, j] = 0
            else:
                dist[i, j] = math.sqrt(d)

    return dist


@nb.njit(fastmath=True)
def argsort_numba(X: np.ndarray) -> np.ndarray:
    M, N = X.shape

    # Adapting argsort
    sorter = np.empty(X.shape, dtype=np.int64)
    for i in range(M):
        line_sorter = np.argsort(X[i])
        for j in range(N):
            sorter[i, j] = line_sorter[j]

    return sorter
