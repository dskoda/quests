import numpy as np

import glassppy as glass
from .base import TreeNeighbors


class TreeGlass(TreeNeighbors):
    def __init__(
        self,
        x: np.ndarray,
        index_type: str = "HNSW",
        metric: str = "L2",
        R: int = 32,
        L: int = 50,
        optimize_level: int = 1,
    ):
        super().__init__(x)
        self.graph = self.build(
            index_type=index_type,
            metric=metric,
            R=R,
            L=L,
        )
        self.searcher = glass.Searcher(
            graph=self.graph,
            data=self.x,
            metric=metric,
            level=optimize_level,
        )
        self.searcher.set_ef(32)
        self.searcher.optimize()

    def build(
        self, index_type: str = "HNSW", metric: str = "L2", R: int = 32, L: int = 50
    ):
        dim = self.x.shape[1]
        index = glass.Index(index_type=index_type, dim=dim, metric=metric, R=R, L=L)
        graph = index.build(self.x)
        return graph

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        return self.searcher.search(query=x, k=k)
