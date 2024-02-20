import json
import math
import time
import argparse

from typing import Union
import numpy as np
import numba as nb
from ase.io import read

from quests.cli.log import format_time
from quests.cli.log import logger
from quests.cli.load_file import descriptors_from_file
from quests.descriptor import get_descriptors, DEFAULT_CUTOFF, DEFAULT_K
from quests.entropy import perfect_entropy, DEFAULT_BATCH
from quests.tools.time import Timer


def sample_dataset(x: np.ndarray, n: Union[int, float]):
    size = x.shape[0]

    if n < 1:
        n = math.ceil(n * size)
    else:
        n = int(n)

    if size < n:
        return x

    indices = np.random.randint(0, size, n)
    return x[indices]


def compute_entropy(x: np.ndarray, frac: float, n_runs: int, batch_size: int = 2000):
    entropies = []
    times = []
    for run in range(n_runs):
        xsample = sample_dataset(x, frac)

        with Timer() as t:
            entropy = perfect_entropy(xsample, batch_size=batch_size)
        entropy_time = t.time
        entropies.append(entropy)
        times.append(entropy_time)

    logger(f"Entropy: {np.mean(entropies): .2f} ± {np.std(entropies): .2f} (nats)")
    logger(f"computed from {n_runs} runs and {frac} samples.")
    logger(f"Max theoretical entropy: {np.log(xsample.shape[0]): .2f} (nats)")
    logger(f"Mean to compute: {format_time(np.mean(times))}/run")

    return entropies


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    parser.add_argument("file", type=str, help="Path to the file")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH, help="Batch size")
    parser.add_argument(
        "--n_samples", type=float, nargs="+", help="List of sample counts"
    )
    parser.add_argument("--jobs", type=int, default=None, help="number of jobs")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.jobs is not None:
        nb.set_num_threads(args.jobs)

    x, descriptor_time = descriptors_from_file(args.file, k=DEFAULT_K, cutoff=DEFAULT_CUTOFF)

    results = []
    for n_samples in args.n_samples:
        logger("-" * 30)
        entropies = compute_entropy(x, n_samples, args.n_runs, batch_size=args.batch_size)
        results.append(
            {
                "n_samples": n_samples,
                "entropies": entropies,
                "H": np.mean(entropies),
            }
        )

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
