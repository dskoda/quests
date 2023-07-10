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
        """
        self.cutoff = float(cutoff)
        self.k = k

    def get_descriptors(self, atoms: Atoms):
        """Computes the (r, d) distances for all atoms in the structure.

        Arguments:
        -----------
            atoms (ase.Atoms): structure to be analyzed

        Returns:
        --------
            r (np.ndarray): radial distances for each atomic environment
            d (np.ndarray): cross distances for each atomic environment
        """
        i, j, d, D = nbrlist("ijdD", atoms, cutoff=self.cutoff)
        subarrays = self._split_array(i)

        rs, ds = [], []
        for subarray in subarrays:
            dist = d[subarray]
            sorter = np.argsort(dist)[: self.k]
            xyz = D[subarray][sorter]

            rs.append(dist[sorter])
            ds.append(np.sort(pdist(xyz)))

        return np.array(rs), np.array(ds)

    def _split_array(self, sorted_array: np.ndarray) -> List[np.ndarray]:
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
