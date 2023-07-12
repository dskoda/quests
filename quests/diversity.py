import numpy as np
from scipy.special import digamma
from sklearn.neighbors import BallTree
from pykdtree.kdtree import KDTree


EULER_CONSTANT = 0.5772156649
ENTROPY_MIN_CONSTANT = np.log(2) + EULER_CONSTANT


def add_noise(x, eps=1e-10):
    return x + eps * np.random.random_sample(x.shape)


def entropy_knn(x, k=3, base=2, metric="chebyshev"):
    """
    https://github.com/gregversteeg/NPEET/tree/master
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138
    https://journals.aps.org/pre/abstract/10.1103/PhysRevE.76.026209
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = BallTree(x, metric=metric)
    d, i = tree.query(x, k + 1)
    nn = d[:, k]
    const = digamma(n_elements) - digamma(k) + n_features * np.log(2)
    return (const + n_features * np.log(nn).mean()) / np.log(base)


def entropy_min(x: np.ndarray, eps: float = 1e-10):
    """
    http://jimbeck.caltech.edu/summerlectures/references/Entropy%20estimation.pdf
    """
    tree = KDTree(x)
    d, i = tree.query(x, 2)
    d = add_noise(d[:, 1], eps=eps)
    n = x.shape[0]
    return np.log(n * d).mean() + ENTROPY_MIN_CONSTANT
