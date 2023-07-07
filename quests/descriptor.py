from typing import List

import numpy as np
from ase import Atoms
from matscipy.neighbours import neighbour_list as nbrlist
from scipy.spatial.distance import pdist


class QUESTS:
    def __init__(
        self,
        cutoff: float = 5.0,
        k: int = 8,
    ):
        """Computes the QUESTS descriptor in batches.

        Arguments:
        ----------
            cutoff (float): maximum distance to consider two atoms
                as neighbors when computing the neighbor list.
            k (int): number of nearest neighbors to fix the length
                of the descriptor.
        """
        self.cutoff = cutoff
        self.k = k

    def get_descriptors(self, atoms: Atoms):
        i, j, d, D = nbrlist("ijdD", atoms, cutoff=self.cutoff)
        subarrays = self.split_array(i)

        rs, ds = [], []
        for subarray in subarrays:
            dist = d[subarray]
            sorter = np.argsort(dist)[: self.k + 1]
            xyz = D[subarray][sorter]

            rs.append(dist[sorter][1:])
            ds.append(np.sort(pdist(xyz)[self.k :]))

        return np.array(rs), np.array(ds)

    def split_array(self, sorted_array: np.ndarray) -> List[np.ndarray]:
        """Splits a sorted array of ints in different arrays according
        to their number.

        Arguments:
        ----------
            sorted_array: array of ints

        Returns:
        --------
            subarrays: list of arrays of ints
        """
        sorted_indices = np.arange(len(sorted_array))
        unique_elements, start_indices = np.unique(
            sorted_array,
            return_index=True,
        )

        start_indices = np.append(start_indices, len(sorted_array))
        subarrays = [
            sorted_indices[start_indices[i] : start_indices[i + 1]]
            for i in range(len(unique_elements))
        ]

        return subarrays
