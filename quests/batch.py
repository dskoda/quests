import numpy as np
from typing import List

def split_array(sorted_array: np.ndarray) -> List[np.ndarray]:
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
