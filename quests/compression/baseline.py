import itertools
import random
from typing import List

import numpy as np


def random_sample(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int
) -> List[int]:
    indices = np.arange(len(descriptors)).tolist()
    selected = random.sample(indices, size)
    return selected


def mean_fps(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int
) -> List[int]:

    avg_descriptors = np.array([x.mean(0) for x in descriptors])
    n_points, n_dim = avg_descriptors.shape

    start_idx = np.random.randint(0, n_points)

    sampled_indices = [start_idx]
    min_distances = np.full(n_points, np.inf)
    
    for _ in range(size - 1):
        current_point = avg_descriptors[sampled_indices[-1]]
        dist_to_current_point = np.linalg.norm(avg_descriptors - current_point, axis=1)
        min_distances = np.minimum(min_distances, dist_to_current_point)
        for num in sampled_indices:
            min_distances[num] = -np.inf
        farthest_point_idx = np.argmax(min_distances)
        sampled_indices.append(farthest_point_idx)

    return sampled_indices

def k_means(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int
) -> List[int]:
    from sklearn.cluster import KMeans
    
    avg_descriptors = np.array([x.mean(0) for x in descriptors])
    kmeans = KMeans(n_clusters=size, n_init="auto").fit(avg_descriptors)
    selected = []
    for num in range(kmeans.labels_.max()):
        if sum(kmeans.labels_ == num) != 0:
            selected.append(np.random.choice(np.where(kmeans.labels_ == num)[0]))
    return selected        

# TODO: Ben, please implement
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
def dbscan_sample(
    descriptors, entropies: np.ndarray, size: int
):
    from sklearn.cluster import DBSCAN
    from bayes_opt import BayesianOptimization 
    
    avg_descriptors = np.array([x.mean(0) for x in descriptors])
    
    def cost_fn(x):
        dbscan = DBSCAN(eps = x, min_samples=1).fit(avg_descriptors)
        return -abs(size-dbscan.labels_.max())
    bounds = {'x': (0.0001, 0.05)}
    
    optimizer = BayesianOptimization(f=cost_fn, pbounds=bounds, allow_duplicate_points=True)
    optimizer.set_gp_params
    optimizer.maximize(init_points=15, n_iter=15)
    
    dbscan = DBSCAN(eps=optimizer.max['params']['x'], min_samples=1).fit(avg_descriptors)
    
    selected = []
    for num in range(dbscan.labels_.max()):
        selected.append(np.random.choice(np.where(dbscan.labels_ == num)[0]))
    return selected
    
