import itertools
import random
from typing import List

import numpy as np
from sklearn.cluster import KMeans


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

    avg_descriptors = np.array([x.mean(0) for x in descriptors])
    kmeans = KMeans(n_clusters=size).fit(avg_descriptors)
    selected = []
    for num in range(kmeans.labels_.max()):
        if sum(kmeans.labels_ == num) != 0:
            selected.append(np.random.choice(np.where(kmeans.labels_ == num)[0]))
    return selected


def k_means_entropy(
    descriptors: List[np.ndarray], entropies: np.ndarray, size: int
) -> List[int]:

    avg_descriptors = np.array([x.mean(0) for x in descriptors])
    kmeans = KMeans(n_clusters=size).fit(avg_descriptors)
    selected = []
    for num in range(kmeans.labels_.max()):
        if sum(kmeans.labels_ == num) == 0:
            continue

        in_cluster = np.where(kmeans.labels_ == num)[0]
        entr_cluster = entropies[in_cluster]
        idx = in_cluster[entr_cluster.argmax()]
        selected.append(idx)

    return selected
