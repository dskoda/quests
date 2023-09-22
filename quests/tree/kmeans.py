import multiprocess as mp
import numpy as np
import tqdm
from quests.distance import batch_distances
from sklearn.cluster import MiniBatchKMeans

from .base import FinderNeighbors


class FinderKMeans(FinderNeighbors):
    def __init__(
        self,
        x: np.ndarray,
        h: float,
        n_clusters: int,
        n_centroid_nbrs: int = 5,
        metric: str = "euclidean",
        batch_size: int = 2000,
        **kwargs,
    ):
        super().__init__(x)
        self.h = h
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, **kwargs)
        self.distance = lambda x, y: batch_distances(
            x, y, batch_size=batch_size, metric=metric
        )
        self.n_centroid_nbrs = n_centroid_nbrs

    def build(self):
        # find the clusters
        self.kmeans.fit(self.x)

        self.labels = self.kmeans.labels_

        # find the points closest to each centroid
        self.centroids = self.kmeans.cluster_centers_
        self.nbrs = {n: self.x[self.labels == n] for n in range(self.n_clusters)}

    def query(self, x: np.ndarray) -> np.ndarray:
        x_i = np.arange(0, len(x))
        x_centroids = self.distance(x, self.centroids)
        x_clusters = np.argsort(x_centroids, axis=1)[:, : self.n_centroid_nbrs]

        dists = {i: [] for i in x_i}

        # loops over a constant number of clusters for speed
        for n in tqdm.tqdm(range(self.n_clusters)):
            nbrs = self.nbrs[n]
            has_cluster = np.any(x_clusters == n, axis=1)

            # p has the query points for a given cluster
            p = x[has_cluster]
            dist = self.distance(p, nbrs)

            # stores the results based on the clusters
            for i, x_d in zip(x_i[has_cluster], dist):
                dists[i] = dists[i] + [x_d]

        dists = [
            np.concatenate(d)
            for d in dists.values()
        ]
        return dists
