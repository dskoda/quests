from typing import List

import numpy as np
from ase import Atoms
from quests.descriptor import get_descriptors
from quests.entropy import diversity, perfect_entropy

from .fps import fps
import ray

DEFAULT_CUTOFF: float = 5.0
DEFAULT_K: int = 32
EPS: float = 1e-15
DEFAULT_H: float = 0.015
DEFAULT_BS: int = 10000


def segment_compress(
    descriptors: np.ndarray,
    entropies: np.ndarray,
    num_sample: int,
    num_processes: int,
    num_chunks: int,
):

    ray.init(ignore_reinit_error=True)

    chunk_size = int(np.ceil(len(descriptors) / num_processes))

    start_indexes = [i * chunk_size for i in range(num_processes)]
    sample_mini_size = [num_sample // num_processes] * (num_processes - 1)
    last_val = (
        (num_sample - (num_processes - 1) * (num_sample // num_processes))
        if num_sample % num_processes != 0
        else num_sample // num_processes
    )
    sample_mini_size.append(last_val)
    result_ids = [
        process_dataset.remote(
            descriptors[start : start + chunk_size],
            entropies[start : start + chunk_size],
            num_chunks=num_chunks,
            num_sample=mini_sample,
        )
        for start, mini_sample in zip(start_indexes, sample_mini_size)
    ]

    results = ray.get(result_ids)

    ray.shutdown()

    y = []
    for i in range(len(results)):
        y = np.concatenate([y, np.array(results[i]) + start_indexes[i]])

    return y.astype(int)


@ray.remote
def process_dataset(
    x: np.ndarray, initial_entropies: np.ndarray, num_chunks: int, num_sample: int
):

    N = len(x)

    if N <= num_sample:
        return np.arange(N)

    if N <= num_chunks * num_sample:
        return fps(x, initial_entropies, size=num_sample, method="msc")

    chunk_size = num_chunks * num_sample
    num_subsets = int(np.ceil(N / chunk_size))

    y = []
    for i in range(num_subsets):
        start = i * chunk_size
        chunk = x[start : start + chunk_size]
        initial_entropies_chunk = initial_entropies[start : start + chunk_size]
        y.append(
            start
            + np.array(
                fps(chunk, initial_entropies_chunk, size=num_sample, method="msc")
            )
        )

    y = np.concatenate(y)
    result = []
    for ind in y:
        result.append(x[ind])
    i = process_dataset.remote(result, initial_entropies[y], num_chunks, num_sample)

    return y[np.array(ray.get(i))]
