import numpy as np
import multiprocess as mp

from quests.batch import chunks
from .base import TreeNeighbors
from pykdtree.kdtree import KDTree


class TreePyKDTree(TreeNeighbors):
    def __init__(
        self,
        x: np.ndarray,
    ):
        super().__init__(x)
        self.tree = None

    def build(self):
        self.tree = KDTree(self.x)

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        dij, _ = self.tree.query(x, k=k)
        return dij

    def query_parallel(self, x: np.ndarray, k: int, jobs: int = 1) -> np.ndarray:
        subx = chunks(x, len(x) // jobs)

        def worker_fn(_x):
            dij, _ = self.tree.query(_x, k=k)
            return dij

        with mp.Pool(jobs) as p:
            results = p.map(worker_fn, subx)

        return np.concatenate(results)
