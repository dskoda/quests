import multiprocess as mp
import numpy as np
from quests.batch import chunks

from .base import FinderNeighbors
from pykdtree.kdfinder import KDTree


class KDTreeFinder(FinderNeighbors):
    def __init__(
        self,
        x: np.ndarray,
    ):
        super().__init__(x)
        self.finder = None

    def build(self):
        self.finder = KDTree(self.x)

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        dij, _ = self.finder.query(x, k=k)
        return dij

    def query_parallel(self, x: np.ndarray, k: int, jobs: int = 1) -> np.ndarray:
        subx = chunks(x, len(x) // jobs)

        def worker_fn(_x):
            dij, _ = self.finder.query(_x, k=k)
            return dij

        with mp.Pool(jobs) as p:
            results = p.map(worker_fn, subx)

        return np.concatenate(results)
