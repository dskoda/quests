import numpy as np

from .base import TreeNeighbors
from pynndescent import NNDescent


class TreePyNNDescent(TreeNeighbors):
    def __init__(
        self,
        x: np.ndarray,
    ):
        super().__init__(x)
        self.tree = self.build()

    def build(self):
        return NNDescent(self.x)

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        dij, _ = self.tree.query(x, k=k)
        return dij
