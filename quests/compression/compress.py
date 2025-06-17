from typing import Callable, List

import numpy as np
from ase import Atoms
from bayes_opt import BayesianOptimization
from quests.descriptor import get_descriptors
from quests.entropy import (
    DEFAULT_BANDWIDTH,
    DEFAULT_BATCH,
    diversity,
    entropy,
    delta_entropy,
)

from .baseline import k_means, mean_fps, random_sample
from .fps import fps, msc

EPSILON = 1e-5
METHODS = {
    "fps": fps,
    "msc": msc,
    "random": random_sample,
    "mean_fps": mean_fps,
    "k_means": k_means,
}


class DatasetCompressor:
    def __init__(
        self,
        dset: List[Atoms],
        descriptor_fn: Callable = None,
        bandwidth: float = DEFAULT_BANDWIDTH,
        batch_size: int = DEFAULT_BATCH,
    ):
        self.dset = dset

        if descriptor_fn is None:
            self.descriptor_fn = lambda _data: get_descriptors([_data])
        else:
            self.descriptor_fn = descriptor_fn

        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self._descriptors = [self.descriptor_fn(at) for at in dset]
        self._entropies = np.array(
            [entropy(x, h=bandwidth, batch_size=batch_size) for x in self._descriptors]
        )

    def entropy(self, selected: List[int] = None):
        if selected is None:
            data = np.concatenate(self._descriptors, axis=0)
        else:
            data = np.concatenate([self._descriptors[i] for i in selected], axis=0)

        return float(entropy(data, h=self.bandwidth, batch_size=self.batch_size))

    def diversity(self, selected: List[int] = None):
        if selected is None:
            data = np.concatenate(self._descriptors, axis=0)
        else:
            data = np.concatenate([self._descriptors[i] for i in selected], axis=0)

        return float(diversity(data, h=self.bandwidth, batch_size=self.batch_size))

    def overlap(self, selected: List[int] = None):
        if selected is None:
            return 1.0

        data = np.concatenate([self._descriptors[i] for i in selected], axis=0)
        full = np.concatenate(self._descriptors, axis=0)

        dH = delta_entropy(full, data, h=self.bandwidth, batch_size=self.batch_size)
        return float((dH < EPSILON).mean())

    def num_envs(self, selected: List[int] = None) -> int:
        if selected is None:
            selected = range(len(self.dset))

        return sum([len(self.dset[i]) for i in selected])

    def get_summary(self, selected: List[int] = None):
        size = self.dataset_size

        return {
            "entropy": self.entropy(selected),
            "diversity": self.diversity(selected),
            "overlap": self.overlap(selected),
            "original_size": size,
            "compressed_size": len(selected) if selected is not None else size,
            "original_envs": self.num_envs(),
            "compressed_envs": self.num_envs(selected)
        }

    @property
    def dataset_size(self):
        return len(self.dset)

    def _check_compression_method(self, method: str):
        assert method in METHODS, (
            f"Compression method {method} not known."
            + f"Acceptable values are: {list(METHODS.keys())}"
        )

    def _check_frac(self, frac: float):
        assert isinstance(frac, float), f"Argument frac = {frac} is not a float"
        assert frac >= 0.0 and frac <= 1.0, "frac has to be between zero and one"

    def frac_to_size(self, frac):
        size = int(np.ceil(self.dataset_size * frac))
        return min(size, self.dataset_size)

    def fixed_compression(self, method: str = "msc", frac: float = None):
        self._check_frac(frac)

        size = self.frac_to_size(frac)
        indices = self.get_indices(method, size)
        return [x for i, x in enumerate(self.dset) if i in indices]

    def cost_fn(self, frac, method):
        size = self.frac_to_size(frac)
        selected = self.get_indices(method, size)
        entropy = self.entropy(selected)
        div = self.diversity(selected)
        return entropy * div

    def optimal_compression(
        self,
        method: str = "msc",
        min_frac: float = 0.1,
        random_state: int = None,
        init_points: int = 5,
        n_iter: int = 20,
    ):
        self._check_frac(min_frac)

        fn = lambda frac: self.cost_fn(frac=frac, method=method)

        bounds = {"frac": (min_frac, 1)}
        opt = BayesianOptimization(
            f=fn,
            pbounds=bounds,
            random_state=random_state,
            allow_duplicate_points=True,  # potentially suboptional cost function
        )
        opt.maximize(init_points=init_points, n_iter=n_iter)
        optimal_frac = opt.max["params"]["frac"]

        return self.fixed_compression(method, optimal_frac), optimal_frac

    def get_indices(self, method: str, size: int, **kwargs):
        self._check_compression_method(method)
        compress_fn = METHODS[method]

        if method == "msc":
            kwargs = {
                **kwargs,
                "h": self.bandwidth,
                "batch_size": self.batch_size,
            }

        return compress_fn(self._descriptors, self._entropies, size, **kwargs)

    def segment_compress(self, method: str, size: int, num_chunks: int, **kwargs):
        self._check_compression_method(method)

        N = len(self._descriptors)

        if method == "msc":
            kwargs = {
                **kwargs,
                "h": self.bandwidth,
                "batch_size": self.batch_size,
            }

        return self.compress_chunk(
            self._descriptors, self._entropies, method, size, num_chunks, **kwargs
        )

    def compress_chunk(
        self,
        descriptors: np.ndarray,
        entropies: np.ndarray,
        method: str,
        size: int,
        num_chunks: int,
        **kwargs,
    ):

        compress_fn = METHODS[method]

        N = len(descriptors)

        if N <= size:
            return np.arange(N)

        chunk_size = num_chunks * size
        num_subsets = int(np.ceil(N / chunk_size))
        y = []
        for i in range(num_subsets):
            start = i * chunk_size
            chunk = descriptors[start : start + chunk_size]
            initial_entropies_chunk = entropies[start : start + chunk_size]
            y.append(
                start
                + np.array(compress_fn(chunk, initial_entropies_chunk, size, **kwargs))
            )

        y = np.concatenate(y)
        result = []
        for ind in y:
            result.append(descriptors[ind])
        i = self.compress_chunk(
            result, entropies[y], method, size, num_chunks, **kwargs
        )
        return y[i]
