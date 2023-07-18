from abc import ABC
from abc import abstractmethod

import numpy as np


class TreeNeighbors(ABC):
    def __init__(
        self,
        x: np.ndarray,
        **kwargs,
    ):
        self.x = x
        self.n = len(x)

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def query(self, x: np.ndarray, k: int):
        pass
