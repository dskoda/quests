import numpy as np
from quests.batch import split_array
from sklearn.cluster import DBSCAN

from .base import NeighborsFinder
from pykdtree.kdfinder import KDTreeFinder


class FinderDBScan(NeighborsFinder):
    def __init__(self, x: np.ndarray, eps: float 0.03, min_samples: int = 5, **kwargs):
        self.x = x
        self.db = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        self.finder = None

    def build(self):
        self.db.fit(self.x)
        labels = self.db.labels_

        finders = {}
        for label in np.unique(labels):
            y = self.x[labels == label]
            finders[label] = KDTreeFinder(y)

        self.finders = finders

    def query(self, x: np.ndarray, k: int):
        labels, _ = self.db.predict(x)

        if x.shape[0] == 1:
            finder = self.finders[labels[0]]
            dij, _ = finder.query(x, k=k)
            return dij

        sorter = np.argsort(labels)
        unsort = np.argsort(sorter)

        y = x[sorter]
        labels = labels[sorter]
        unique = np.sort(np.unique(labels))

        distances = []
        for i, label in zip(split_array(labels), unique):
            finder = self.finders[label]
            dij, _ = finder.query(y[i], k=k)
            distances.append(dij)

        distances = np.concatenate(distances)

        return distances[unsort]
