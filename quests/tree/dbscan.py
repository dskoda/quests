import numpy as np
from quests.batch import split_array
from sklearn.cluster import DBSCAN

from .base import TreeNeighbors
from pykdtree.kdtree import KDTree


class TreeDBScan(TreeNeighbors):
    def __init__(self, x: np.ndarray, **kwargs):
        self.x = x
        self.db = DBSCAN(eps=0.03, min_samples=5)
        self.tree = None

    def build(self):
        self.db.fit(self.x)
        labels = self.db.labels_

        trees = {}
        for label in np.unique(labels):
            y = self.x[labels == label]
            trees[label] = KDTree(y)

        self.trees = trees

    def query(self, x: np.ndarray, k: int):
        labels, _ = self.db.predict(x)

        if x.shape[0] == 1:
            tree = self.trees[labels[0]]
            dij, _ = tree.query(x, k=k)
            return dij

        sorter = np.argsort(labels)
        unsort = np.argsort(sorter)

        y = x[sorter]
        labels = labels[sorter]
        unique = np.sort(np.unique(labels))

        distances = []
        for i, label in zip(split_array(labels), unique):
            tree = self.trees[label]
            dij, _ = tree.query(y[i], k=k)
            distances.append(dij)

        distances = np.concatenate(distances)

        return distances[unsort]
