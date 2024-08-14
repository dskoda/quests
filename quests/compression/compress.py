from typing import List, Callable, Tuple

import numpy as np
from ase import Atoms
from quests.descriptor import get_descriptors
from quests.entropy import diversity, perfect_entropy, DEFAULT_BANDWIDTH, DEFAULT_BATCH

from .farthest_point_sampling import farthest_point_sampling
from .minimum_set_coverage import minimum_set_coverage


class DatasetCompressor:
    def __init__(
        self,
        dset: List[Atoms],
        descriptor_fn: Callable,
        bandwidth: float = DEFAULT_BANDWIDTH,
        batch_size: int = DEFAULT_BATCH,
    ):
        self.dset = dset
        self.descriptor_fn = descriptor_fn
        self.bandwidth = bandwidth
        self.batch_size = batch_size
        self._frames = [descriptor_fn(at) for at in dset]
        self._entropies = np.array([
            perfect_entropy(frame, h=bandwidth, batch_size=batch_size)
            for frame in self._frames
        ])

    def entropy(self, selected: List[int] = None):
        if selected is None:
            data = np.concatenate(self._frames, axis=0)
        else:
            data = np.concatenate([self._frames[i] for i in selected], axis=0)

        return perfect_entropy(data, h=self.bandwidth, batch_size=self.batch_size)

    def diversity(self, selected: List[int] = None):
        if selected is None:
            data = np.concatenate(self._frames, axis=0)
        else:
            data = np.concatenate([self._frames[i] for i in selected], axis=0)

        return diversity(data, h=self.bandwidth, batch_size=self.batch_size)

    @property
    def dataset_size(self):
        return len(self.dset)

    def _check_compression_method(self, method: str):
        acceptable = ["msc", "fps"]
        assert method in acceptable, (
            f"Compression method {method} not known."
            + f"Acceptable values are: {acceptable}"
        )

    def _check_frac(self, frac: float):
        assert isinstance(frac, float), f"Argument frac = {frac} is not a float"
        assert frac >= 0.0 and frac <= 1.0, "frac has to be between zero and one"

    def frac_to_size(self, frac):
        return int(np.ceil(self.dataset_size * frac))

    def fixed_compression(self, method: str = "msc", frac: float = None):
        self._check_frac(frac)

        indices = self.get_indices(method)
        final_size = self.frac_to_size(frac)
        return [x for i, x in enumerate(self.dset) if i in indices[:final_size]]

    def cost_fn(self, frac, indices: List[int]):
        size = self.frac_to_size(frac)
        selected = indexes[:size]
        entropy = self.entropy(selected)
        div = self.diversity(selected)
        return entropy * div

    def optimal_compression(
        self,
        method: str = "msc",
        min_frac: float = 0.1,
        random_state: int = 1243,
        init_points: int = 5,
        n_iter: int = 20,
    ):
        self._check_frac(min_frac)

        indices = self.get_indices(method)
        fn = lambda x: self.cost_fn(x, indices=indices)

        bounds = {"frac": (min_frac, 1)}
        opt = BayesianOptimization(f=fn, pbounds=bounds, random_state=random_state)
        opt.maximize(init_points=init_points, n_iter=n_iter)
        final_size = self.frac_to_size(opt.max["params"]["frac"])

        return [x for i, x in enumerate(self.dset) if i in indices[:final_size]]

    def get_indices(self, method, **kwargs):
        self._check_compression_method(method)
        if method == "fps":
            return farthest_point_sampling(self._frames, self._entropies, **kwargs)

        if method == "msc":
            return minimum_set_coverage(
                self._frames, self._entropies, self.bandwidth, **kwargs
            )

        raise ValueError("Compression method not known")
