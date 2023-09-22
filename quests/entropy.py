from typing import Callable
from typing import Union

import numpy as np
from scipy.special import logsumexp

from .distance import batch_distances
from .finder import TreeNeighbors


class EntropyEstimator:
    def __init__(
        self,
        x: np.ndarray,
        h: float = 0.015,
        nbrs: int = 100,
        finder: NeighborsFinder = "pykdtree",
        kernel: str = "gaussian",
        metric: str = "euclidean",
    ):
        """Initializes the kernel-based entropy estimator.

        Parameters:
        -----------
            x (np.ndarray): reference points to be used for the KDE.
            h (float): bandwidth of the KDE.
            nbrs (int): number of nearest-neighbors to use when
                computing the overlap between points in the distribution.
                More neighbors increase the accuracy, but also add
                computational overhead.
        """
        self.x = x
        self.n = len(x)
        self.h = h
        self.nbrs = nbrs
        self.finder = self._get_finder(finder, x)
        self.kernel = self._get_kernel(kernel)
        self.metric = metric

    def _get_finder(self, finder: Union[str, TreeNeighbors], x: np.ndarray) -> Callable:
        if finder is None:
            return None

        if isinstance(finder, TreeNeighbors):
            return finder

        if not isinstance(finder, str):
            raise ValueError(f"Tree type {type(finder)} not recognized")

        name = finder.lower()

        if name == "pykdtree":
            from .finder.pykdtree import KDTreeFinder
            finder = KDTreeFinder(x)

        else:
            raise ValueError(f"Tree name {name} not recognized")

        finder.build()
        return finder

    def _get_kernel(self, name: str) -> Callable:
        name = name.lower()
        if name == "epanechnikov":
            return epanechnikov_kernel
        
        elif name == "gaussian":
            return gaussian_kernel

        else:
            raise ValueError(f"Kernel {name} not supported")

    def get_distances(self, x: np.ndarray) -> np.ndarray:
        if self.finder is not None:
            return self._get_distances_finder(x)

        return self._get_distances_batch(x)

    def _get_distances_finder(self, x: np.ndarray) -> np.ndarray:
        return self.finder.query(x, k=self.nbrs)

    def _get_distances_batch(self, x: np.ndarray) -> np.ndarray:
        return batch_distances(x, self.x, metric=self.metric)

    def zij(self, x: np.ndarray) -> np.ndarray:
        """constructs the distance matrices"""
        dij = self.get_distances(x)
        if isinstance(dij, np.ndarray):
            return dij / self.h

        return [
            d / self.h
            for d in dij
        ]

    def entropy(self, x: np.ndarray) -> float:
        """Computes the entropy of the points with respect to the
            initial dataset.

        Arguments:
        ----------
            x (np.ndarray): points where the entropy will be computed.

        Returns:
        --------
            entropy (float): total entropy of the system.
        """
        logp = self.delta_entropy(x)
        logn = np.log(self.n)

        return logn + logp.mean()

    @property
    def dataset_entropy(self) -> float:
        """Computes the entropy of the initial dataset.

        Returns:
        --------
            entropy (float): total entropy of the system.
        """
        return self.entropy(self.x)

    def delta_entropy(self, x: np.ndarray) -> np.ndarray:
        """Computes the pointwise entropy of the points `x` with respect
            to the initial dataset.

        Arguments:
        ----------
            x (np.ndarray): points where the entropy will be computed.

        Returns:
        --------
            entropy (np.ndarray): total entropy of the system.
        """
        z = self.zij(x)
        logp = self.kernel(z)
        return -logp


def epanechnikov_kernel(z: np.ndarray, eps: float = 1e-15):
    """Computes the Epachenikov kernel for normalized values z,
        z_i = (x - x_i) / h,
        where z is a matrix (n, nbrs), with n being the number of
        points evaluated at once and nbrs the number of neighbors.
    """

    u = z * z
    k = 1 - u.clip(max=1)
    return np.log(k.sum(axis=-1) + eps)


def gaussian_kernel(z: np.ndarray):
    return logsumexp(-(z**2) / 2, axis=-1)
