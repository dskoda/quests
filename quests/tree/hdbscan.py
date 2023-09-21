import hdbscan as hdb
import numpy as np
from pykdtree.kdtree import KDTree

from .base import TreeNeighbors
from quests.batch import split_array


class TreeHDBScan(TreeNeighbors):
    def __init__(self, x: np.ndarray, **kwargs):
        self.x = x
        self.clusterer = hdb.HDBSCAN(prediction_data=True, **kwargs)
        self.tree = None

    def build(self):
        self.clusterer.fit(self.x)
        labels = self.clusterer.labels_

        trees = {}
        for label in np.unique(labels):
            y = self.x[labels == label]
            trees[label] = KDTree(y)

        self.trees = trees

    def query(self, x: np.ndarray, k: int):
        labels, _ = hdb.approximate_predict(self.clusterer, x)

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
