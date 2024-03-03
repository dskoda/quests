import math

import numba as nb
import numpy as np


@nb.njit(fastmath=True)
def sum_positive(X):
    """sumexp optimized for numba. Can lead to numerical
        instabilities, but it's really fast.

    Arguments:
        X (np.ndarray): an (N, d) matrix with the values. The
            summation will happen over the axis 1.

    Returns:
        sumexp (np.ndarray): sum(exp(X), axis=1)
    """
    result = np.empty(X.shape[0], dtype=X.dtype)
    for i in range(X.shape[0]):
        _sum = 0.0
        for j in range(X.shape[1]):
            _sum += max(X[i, j], 0)

        result[i] = _sum

    return result


@nb.njit(fastmath=True)
def sumexp(X):
    """sumexp optimized for numba. Can lead to numerical
        instabilities, but it's really fast.

    Arguments:
        X (np.ndarray): an (N, d) matrix with the values. The
            summation will happen over the axis 1.

    Returns:
        sumexp (np.ndarray): sum(exp(X), axis=1)
    """
    result = np.empty(X.shape[0], dtype=X.dtype)
    for i in range(X.shape[0]):
        _sum = 0.0
        for j in range(X.shape[1]):
            _sum += math.exp(X[i, j])

        result[i] = _sum

    return result


@nb.njit(fastmath=True)
def wsumexp(X, w):
    """weighted sumexp optimized for numba. Does not check any
        variable and can be unstable, but it's really fast.

    Arguments:
        X (np.ndarray): an (N, d) matrix with the values. The
            summation will happen over the axis 1.
        w (np.ndarray): a (d, ) vector with the weights.

    Returns:
        wsumexp (np.ndarray): sum(w * exp(X), axis=1)
    """
    result = np.empty(X.shape[0], dtype=X.dtype)
    for i in range(X.shape[0]):
        _sum = 0.0
        for j in range(X.shape[1]):
            _sum += math.exp(X[i, j]) * w[j]

        result[i] = _sum

    return result


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
    result = sumexp(X)
    return np.log(result)


@nb.njit(fastmath=True)
def norm(A):
    norm_A = np.empty(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        _sum = 0.0
        for j in range(A.shape[1]):
            a = A[i, j]
            _sum += a * a
        norm_A[i] = _sum

    return norm_A


@nb.njit(fastmath=True)
def cdist(A, B, norm_A=None, norm_B=None):
    """Optimized distance calculation using numba.

    Arguments:
        A (np.ndarray): an (N, d) matrix with the descriptors
        B (np.ndarray): an (M, d) matrix with the descriptors

    Returns:
        dist (float): entropy of the dataset given by `x`.
    """
    # Computing the dot product
    dist = np.dot(A, B.T)

    # Computing the norms if they are not given
    if norm_A is None:
        norm_A = norm(A)

    if norm_B is None:
        norm_B = norm(B)

    # computes the distance using the dot product
    # | a - b | ** 2 = <a, a> + <b, b> - 2<a, b>
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            d = -2.0 * dist[i, j] + norm_A[i] + norm_B[j]

            # sqrt with numerical stability
            dist[i, j] = math.sqrt(d) if d > 0 else 0

    return dist


@nb.njit(fastmath=True)
def cdist_Linf(A, B):
    """Optimized distance calculation using numba using the
        Chebyshev distance (L-infinity norm)

    Arguments:
        A (np.ndarray): an (N, d) matrix with the descriptors
        B (np.ndarray): an (M, d) matrix with the descriptors

    Returns:
        dist (np.ndarray): distance matrix
    """
    # Computing the dot product
    M = A.shape[0]
    N = B.shape[0]
    dm = np.empty((M, N), dtype=A.dtype)

    # computes the distance using the Chebyshev norm
    for i in range(A.shape[0]):
        a = A[i]
        for j in range(B.shape[0]):
            dm[i, j] = np.abs(a - B[j]).max()

    return dm


@nb.njit(fastmath=True)
def pdist(A):
    """Optimized distance matrix calculation using numba.

    Arguments:
        A (np.ndarray): an (N, d) matrix

    Returns:
        dm (np.ndarray): an (N, N) matrix with the distances
    """
    N, d = A.shape
    dm = np.empty((N, N), dtype=A.dtype)
    for i in range(N):
        dm[i, i] = 0

    # compute only the off-diagonal terms
    for i in range(N):
        for j in range(i + 1, N):
            dist = 0
            for k in range(d):
                diff = A[i, k] - A[j, k]
                dist += diff * diff

            dist = np.sqrt(dist)
            dm[i, j] = dist
            dm[j, i] = dist

    return dm


@nb.njit(fastmath=True)
def argsort(X: np.ndarray, sort_max: int = -1) -> np.ndarray:
    M, N = X.shape
    if sort_max > 0:
        M = sort_max

    # Adapting argsort
    sorter = np.empty((M, N), dtype=np.int64)
    for i in range(M):
        line_sorter = np.argsort(X[i])
        for j in range(N):
            sorter[i, j] = line_sorter[j]

    return sorter


@nb.njit(fastmath=True)
def inverse_3d(matrix: np.ndarray):
    bx = np.cross(matrix[1], matrix[2])
    by = np.cross(matrix[2], matrix[0])
    bz = np.cross(matrix[0], matrix[1])

    det = matrix[0, 0] * bx[0] + matrix[0, 1] * bx[1] + matrix[0, 2] * bx[2]

    inv = np.empty((3, 3))
    for i in range(3):
        inv[i, 0] = bx[i] / det
        inv[i, 1] = by[i] / det
        inv[i, 2] = bz[i] / det

    return inv


@nb.njit(fastmath=True)
def stack_xyz(arrays: list):
    n = len(arrays)
    stacked = np.empty((n, 3))
    for i in range(n):
        row = arrays[i]
        for j in range(3):
            stacked[i, j] = row[j]

    return stacked
