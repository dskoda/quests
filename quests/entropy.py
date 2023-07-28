import numpy as np
from annoy import AnnoyIndex
from pykdtree.kdtree import KDTree
from scipy.special import logsumexp

from .distance import batch_distances
from .tree import TreeNeighbors


class EntropyEstimator:
    def __init__(
        self,
        x: np.ndarray,
        h: float,
        nbrs: int = 20,
        tree: TreeNeighbors = None,
        kernel: str = "epanechnikov",
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
        self.tree = tree

        kernel = kernel.lower()
        if kernel == "epanechnikov":
            self.kernel = epanechnikov_kernel
        
        elif kernel == "gaussian":
            self.kernel = gaussian_kernel

        else:
            raise ValueError(f"Kernel {kernel} not supported")

    def get_distances(self, x: np.ndarray) -> np.ndarray:
        if self.tree is not None:
            return self._get_distances_tree(x)

        return self._get_distances_batch(x)

    def _get_distances_tree(self, x: np.ndarray) -> np.ndarray:
        return self.tree.query(x, k=self.nbrs)

    def _get_distances_batch(self, x: np.ndarray) -> np.ndarray:
        return batch_distances(x, self.x)

    def zij(self, x: np.ndarray) -> np.ndarray:
        """constructs the distance matrices"""
        dij = self.get_distances(x)
        return dij / self.h

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


def epanechnikov_kernel(z: np.ndarray):
    """Computes the Epachenikov kernel for normalized values z,
        z_i = (x - x_i) / h,
        where z is a matrix (n, nbrs), with n being the number of
        points evaluated at once and nbrs the number of neighbors.
    """

    u = z * z
    k = 1 - u.clip(max=1)
    return np.log(k.sum(axis=-1))


def gaussian_kernel(z: np.ndarray):
    return logsumexp(-(z**2) / 2, axis=-1)
