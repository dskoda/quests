import numpy as np
from quests.batch import split_array
from sklearn.cluster import DBSCAN

from .base import FinderNeighbors
from pykdtree.kdtree import KDTreeFinder


class FinderDBScan(FinderNeighbors):
    def __init__(self, x: np.ndarray, **kwargs):
        self.x = x
        self.db = DBSCAN(eps=0.03, min_samples=5)
        self.finder = None

    def build(self):
        self.db.fit(self.x)
        labels = self.db.labels_

        trees = {}
        for label in np.unique(labels):
            y = self.x[labels == label]
            trees[label] = KDTreeFinder(y)

        self.finders = trees

    def query(self, x: np.ndarray, k: int):
        labels, _ = self.db.predict(x)

        if x.shape[0] == 1:
            tree = self.finders[labels[0]]
            dij, _ = tree.query(x, k=k)
            return dij

        sorter = np.argsort(labels)
        unsort = np.argsort(sorter)

        y = x[sorter]
        labels = labels[sorter]
        unique = np.sort(np.unique(labels))

        distances = []
        for i, label in zip(split_array(labels), unique):
            tree = self.finders[label]
            dij, _ = tree.query(y[i], k=k)
            distances.append(dij)

        distances = np.concatenate(distances)

        return distances[unsort]
