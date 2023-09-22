import numpy as np

from .base import NeighborsFinder
from annoy import AnnoyIndex


class FinderAnnoy(NeighborsFinder):
    def __init__(
        self,
        x: np.ndarray,
        metric: str = "euclidean",
        n_jobs: int = -1,
        n_finders: int = 100,
    ):
        super().__init__(x)
        self.finder = self.build(
            metric=metric,
            n_jobs=n_jobs,
            n_finders=n_finders,
        )

    def build(
        self,
        metric: str = "euclidean",
        n_jobs: int = -1,
        n_finders: int = 100,
    ):

        n_feats = self.x.shape[1]
        t = AnnoyIndex(n_feats, metric)

        for i, v in enumerate(self.x):
            t.add_item(i, v)

        t.build(n_finders, n_jobs)
        return t

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        d = []
        for _x in x:
            _, dij = self.finder.get_nns_by_vector(_x, k, include_distances=True)
            d.append(dij)

        return np.stack(d)
