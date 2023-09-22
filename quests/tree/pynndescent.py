import numpy as np

from .base import FinderNeighbors
from pynndescent import NNDescent


class FinderPyNNDescent(FinderNeighbors):
    def __init__(
        self,
        x: np.ndarray,
        **kwargs
    ):
        super().__init__(x)
        self.finder = self.build(**kwargs)

    def build(self, **kwargs):
        return NNDescent(self.x, **kwargs)

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        dij, _ = self.finder.query(x, k=k)
        return dij
