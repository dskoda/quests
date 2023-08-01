import numpy as np

from .base import TreeNeighbors
from pynndescent import NNDescent


class TreePyNNDescent(TreeNeighbors):
    def __init__(
        self,
        x: np.ndarray,
        **kwargs
    ):
        super().__init__(x)
        self.tree = self.build(**kwargs)

    def build(self, **kwargs):
        return NNDescent(self.x, **kwargs)

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        dij, _ = self.tree.query(x, k=k)
        return dij
