from typing import Callable
from typing import List

import multiprocess as mp
import numba
import numpy as np
from ase import Atoms
from matscipy.neighbours import neighbour_list as nbrlist
from scipy.spatial.distance import cdist

from .batch import chunks
from .batch import split_array


class QUESTS:
    def __init__(
        self,
        cutoff: float = 5.0,
        k: int = 8,
        weight_fn: Callable = None,
        interaction_cutoff: float = None,
    ):
        """Class for generating the sorted distance lists for atomic
            environments using k-nearest neighbors. The class uses
            fast method to compute the descriptors for the whole
            structure at once by batching the calculations in arrays.

        Parameters:
        -----------
            cutoff (float): maximum distance to consider two atoms
                as neighbors when computing the neighbor list.
            k (int): number of nearest neighbors to fix the length
                of the descriptor.
            weight_fn (callable): function that smooths out the distances
                to make the descriptor continuous.
        """
        self.cutoff = float(cutoff)
        self.k = k
        if interaction_cutoff is None:
            interaction_cutoff = 0.75 * self.cutoff

        if weight_fn is None:
            self.weight = lambda r: smooth_weight(r, interaction_cutoff)

        elif isinstance(weight_fn, Callable):
            self.weight = weight_fn

    def get_neighborlist(self, atoms: Atoms, quantities: str = "ijD"):
        return nbrlist(quantities, atoms, cutoff=self.cutoff)

    def get_descriptors_serial(self, atoms: Atoms):
        return descriptors_serial(
            atoms, k=self.k, cutoff=self.cutoff, weight=self.weight
        )

    def get_descriptors_parallel(self, atoms: Atoms, jobs: int = 1):
        """Computes the (r, d) distances for all atoms in the structure.

        Arguments:
        -----------
            atoms (ase.Atoms): structure to be analyzed

        Returns:
        --------
            x1 (np.ndarray): radial distances for each atomic environment
            x2 (np.ndarray): propagated radial distances for environments
        """
        if jobs == 1 or len(atoms) == 1:
            return self.get_descriptors_serial(atoms)

        i, d, D = self.get_neighborlist(atoms, "idD")

        subarrays = split_array(i)
        subds = [(d[subarr], D[subarr]) for subarr in subarrays]
        subsets = chunks(subds, len(subds) // jobs)

        def worker_fn(subset):
            return [
                local_descriptor(d, D, k=self.k, weight=self.weight) for d, D in subset
            ]

        with mp.Pool(jobs) as p:
            results = p.map(worker_fn, subsets)

        if len(results) == 1:
            _x1, _x2 = results
            x1 = np.array(_x1).reshape(1, -1)
            x2 = np.array(_x2).reshape(1, -1)
            return x1, x2

        x1 = np.stack([_x1 for res in results for _x1, _x2 in res])
        x2 = np.stack([_x2 for res in results for _x1, _x2 in res])

        return x1, x2

    def get_all_descriptors(self, dset: List[Atoms], jobs: int = 1):
        if jobs > 1:
            return self.get_all_descriptors_parallel(dset, jobs)

        x1, x2 = [], []
        for at in dset:
            _x1, _x2 = self.get_descriptors_parallel(at, jobs=jobs)
            x1.append(_x1)
            x2.append(_x2)

        return np.concatenate(x1, axis=0), np.concatenate(x2, axis=0)

    def get_all_descriptors_parallel(self, dset: List[Atoms], jobs: int = 1):
        if len(dset) == 1 or jobs == 1:
            return self.get_all_descriptors(dset, jobs=1)

        def worker_fn(atoms):
            return descriptors_serial(
                atoms, k=self.k, cutoff=self.cutoff, weight=self.weight
            )

        with mp.Pool(jobs) as p:
            results = p.map(worker_fn, dset)

        if len(results) == 1:
            _x1, _x2 = results
            x1 = np.array(_x1).reshape(1, -1)
            x2 = np.array(_x2).reshape(1, -1)
            return x1, x2

        x1 = np.concatenate([_x1 for _x1, _x2 in results], axis=0)
        x2 = np.concatenate([_x2 for _x1, _x2 in results], axis=0)

        return x1, x2


def smooth_weight(r, cutoff):
    z = r.clip(max=cutoff) / cutoff
    return (1 - z**2) ** 2


@numba.jit(nopython=True, cache=True)
def numba_inv_dm(dm, w, eps=1e-15):
    n = dm.shape[0]
    new = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            new[i, j] = w[i] * w[j] / (dm[i, j] + eps)

    return new


def descriptors_serial(atoms: Atoms, k: int, cutoff: float, weight: Callable):
    i, r_ij, D_ij = nbrlist("idD", atoms, cutoff=cutoff)

    if len(r_ij) == 0:
        return np.zeros((1, k)), np.zeros((1, k - 1))

    subarrays = split_array(i)
    results = [
        local_descriptor(r_ij[_a], D_ij[_a], k=k, weight=weight) for _a in subarrays
    ]

    if len(results) == 1:
        _x1, _x2 = results
        x1 = np.array(_x1).reshape(1, -1)
        x2 = np.array(_x2).reshape(1, -1)
        return x1, x2

    x1 = np.stack([_x1 for _x1, _x2 in results])
    x2 = np.stack([_x2 for _x1, _x2 in results])

    return x1, x2


def local_descriptor(r_ij: np.ndarray, D_ij: np.ndarray, k: int, weight: Callable):
    sorter = np.argsort(r_ij)[:k]
    dist = r_ij[sorter]
    vecs = D_ij[sorter]

    w = weight(dist)
    x1 = w / dist

    r_jk = cdist(vecs, vecs)
    x2m = numba_inv_dm(r_jk, w)
    x2 = np.fliplr(np.sort(x2m, axis=1))[:, 1:].sum(0) / k

    if len(x1) < k:
        padding = k - len(x1)
        zeros = np.zeros((padding))
        x1 = np.concatenate([x1, zeros])
        x2 = np.concatenate([x2, zeros])

    return x1, x2
