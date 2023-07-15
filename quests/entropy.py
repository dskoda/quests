import numpy as np
from pykdtree.kdtree import KDTree
from scipy.special import logsumexp

from .distance import batch_distances


class EntropyEstimator:
    def __init__(
        self,
        x: np.ndarray,
        h: float,
        nbrs: int = 20,
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
        self.tree = KDTree(x)

    def get_distances(self, x: np.ndarray) -> np.ndarray:
        if self.nbrs is not None:
            dij, _ = self.tree.query(x, k=self.nbrs)
            return dij

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

    def dataset_entropy(self) -> float:
        """Computes the entropy of the initial dataset.

        Returns:
        --------
            entropy (float): total entropy of the system.
        """
        return self.entropy(self.x)

    def kernel(self, x: np.ndarray) -> np.ndarray:
        """Computes the kernel between the given data points and the
            initial dataset.

        Arguments:
        ----------
            x (np.ndarray): points where the entropy will be computed.

        Returns:
        --------
            overlap (np.ndarray): total entropy of the system.
        """
        logp = -self.delta_entropy(x)
        return np.exp(logp) / self.n

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
        logp = logsumexp(-(z**2) / 2, axis=-1)
        return -logp
