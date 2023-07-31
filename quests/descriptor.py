from typing import Callable
from typing import List

import numpy as np
from ase import Atoms
from matscipy.neighbours import neighbour_list as nbrlist

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

    def get_descriptors(self, atoms: Atoms):
        """Computes the (r, d) distances for all atoms in the structure.

        Arguments:
        -----------
            atoms (ase.Atoms): structure to be analyzed

        Returns:
        --------
            x1 (np.ndarray): radial distances for each atomic environment
            x2 (np.ndarray): propagated radial distances for environments
        """
        i, j, d, D = nbrlist("ijdD", atoms, cutoff=self.cutoff)

        # reformat to get matrices [i, j], where rows are atom i
        # and columns are nearest neighbor j
        ij, r_ij, D_ij = self._format_array(i, j, d, D)
        ii = np.arange(0, len(ij)).reshape(-1, 1) * np.ones_like(ij)
        x1 = self.weight(r_ij) / r_ij

        D_il = D_ij.reshape(-1, 1, 3) + D_ij[ij.reshape(-1)]
        r_il = np.linalg.norm(D_il, axis=-1)

        r_jl = r_ij[ij.reshape(-1)]

        _x2 = self.weight(r_ij.reshape(-1, 1)) * self.weight(r_il) / r_jl
        _x2 = np.sort(_x2)[:, ::-1]

        # scatter add and normalize
        x2 = np.zeros_like(x1)
        np.add.at(x2, ii.reshape(-1), _x2)
        x2 = x2 / self.k

        return x1, x2

    def _format_array(self, i, j, d, D):
        subarrays = split_array(i)

        r, ij, vecs = [], [], []
        for subarray in subarrays:
            n = i[subarray][0]
            dist = d[subarray]
            nbrs = j[subarray]
            vec = D[subarray]
            sorter = np.argsort(dist)[: self.k]

            # padding
            if len(dist) < self.k:
                padding = self.k - len(dist)
                dist = np.concatenate([dist[sorter], np.array([np.inf] * padding)])
                nbrs = np.concatenate([nbrs[sorter], np.array([n] * padding)])
                vec = np.concatenate([vec[sorter], np.array([[np.inf] * 3] * padding)])

            else:
                dist = dist[sorter]
                nbrs = nbrs[sorter]
                vec = vec[sorter]

            r.append(dist)
            ij.append(nbrs)
            vecs.append(vec)

        if len(ij) > 1:
            return np.stack(ij), np.array(r), np.stack(vecs, axis=0)

        return np.array([ij]), np.array(r), np.array([vecs])

    def get_all_descriptors(self, dset: List[Atoms]):
        x1, x2 = [], []
        for at in dset:
            _x1, _x2 = self.get_descriptors(at)
            x1.append(_x1)
            x2.append(_x2)

        return np.concatenate(x1, axis=0), np.concatenate(x2, axis=0)

    def x1_iterator(self, i, j, d):
        for split in split_array(i):
            dist = np.sort(d[split])[:self.k]
            if len(dist) < self.k:
                padding = self.k - len(dist)
                dist = np.concatenate([dist, np.array([np.inf] * padding)])

            x1 = self.weight(dist) / dist

            yield x1

def smooth_weight(r, cutoff):
    x = r.clip(max=cutoff) / cutoff
    return (1 - x**2) ** 2
