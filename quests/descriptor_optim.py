import numba as nb
import numpy as np

from .optimized import cdist_numba, argsort_numba


@nb.njit(fastmath=True)
def descriptor_weight(r: float, cutoff: float):
    if r > cutoff:
        r = cutoff

    z = r / cutoff
    return (1 - z**2) ** 2


@nb.njit(fastmath=True)
def descriptor_x1(
    dm: np.ndarray, sorter: np.ndarray, N: int, k: int = 32, cutoff: float = 5.0
) -> np.ndarray:
    # Lazy initialization of the descriptor matrix
    if N > k:
        x1 = np.empty((N, k))
        jmax = k
    else:
        x1 = np.full((N, k), fill_value=0.0)
        jmax = N - 1

    # Computes the descriptor x1 in parallel
    for i in range(N):
        for j in range(jmax):
            atom_j = sorter[i, j + 1]
            rij = dm[i, atom_j]
            wij = descriptor_weight(rij, cutoff)
            x1[i, j] = wij / rij

    return x1


@nb.njit(fastmath=True)
def descriptor_x2(
    dm: np.ndarray,
    sorter: np.ndarray,
    N: int,
    k: int = 32,
    cutoff: float = 5.0,
    eps: float = 1e-15,
) -> np.ndarray:
    # Lazy initialization of the matrix
    if N > k:
        x2 = np.empty((N, k))
        jmax = k
    else:
        x2 = np.full((N, k), fill_value=0.0)
        jmax = N - 1

    # Computes the second descriptor
    for i in range(N):
        rjl = np.full((k, k), fill_value=0.0)

        # first compute the cross distances
        for j in range(jmax):
            atom_j = sorter[i, j + 1]

            for l in range(jmax):
                atom_l = sorter[i, l + 1]

                rij = dm[i, atom_j]
                wij = descriptor_weight(rij, cutoff)

                ril = dm[i, atom_l]
                wil = descriptor_weight(ril, cutoff)

                rjl[j, l] = (wij * wil) / (dm[atom_j, atom_l] + eps)

        # then sort the matrix and remove the case where j == l
        # that corresponds to the largest rjl
        r_sort = np.sort(rjl)

        # now compute the mean and sorts largest first in x2
        for l in range(k - 1):
            _sum = 0.0
            for j in range(k):
                _sum += r_sort[j, l]

            # larger first
            x2[i, k - 2 - l] = _sum / k

    return x2


@nb.njit(fastmath=True)
def descriptor_nopbc(
    xyz: np.ndarray,
    k: int = 32,
    cutoff: float = 5.0,
    eps: float = 1e-15,
) -> np.ndarray:
    N = xyz.shape[0]
    dm = cdist_numba(xyz, xyz)
    sorter = argsort_numba(dm)

    x1 = descriptor_x1(dm, sorter, N, k, cutoff)
    x2 = descriptor_x2(dm, sorter, N, k, cutoff)
    return x1, x2
