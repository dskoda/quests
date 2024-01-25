import json
import time
import argparse

import numpy as np
import numba as nb
from ase.io import read

from quests.cli.log import format_time
from quests.cli.log import logger
from quests.descriptor import get_descriptors
from quests.entropy import perfect_entropy
from quests.tools.time import Timer


def compute_descriptors(file: str):
    logger(f"Loading and creating descriptors for file {file}")
    dset = read(file, index=":")
    n_atoms = len(dset[0])

    with Timer() as t:
        x = get_descriptors(dset, k=n_atoms)
    descriptor_time = t.time
    logger(f"Descriptors built in: {format_time(descriptor_time)}")

    n_atoms = len(dset[0])
    x = x.reshape(len(dset), n_atoms, -1)
    logger(f"Descriptors shape: {x.shape}")

    return x


def sample_dataset(x: np.ndarray, n: int):
    size = x.shape[0]
    if size < n:
        return x

    indices = np.random.randint(0, size, n)
    return x[indices]


def compute_entropy(x: np.ndarray, n_samples: int, n_runs: int, batch_size: int = 2000):
    entropies = []
    times = []
    for run in range(n_runs):
        xsample = sample_dataset(x, n_samples)
        n_envs = xsample.shape[0] * xsample.shape[1]
        xsample = xsample.reshape(n_envs, -1)

        with Timer() as t:
            entropy = perfect_entropy(xsample, batch_size=batch_size)
        entropy_time = t.time
        entropies.append(entropy)
        times.append(entropy_time)

    logger(f"Entropy: {np.mean(entropies): .2f} ± {np.std(entropies): .2f} (nats)")
    logger(f"computed from {n_runs} runs and {n_samples} samples.")
    logger(f"Max theoretical entropy: {np.log(xsample.shape[0]): .2f} (nats)")
    logger(f"Mean to compute: {format_time(np.mean(times))}/run")

    return entropies


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    parser.add_argument("file", type=str, help="Path to the file")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs")
    parser.add_argument(
        "--n_samples", type=int, nargs="+", help="List of sample counts"
    )
    parser.add_argument(
        "--jobs", type=int, default=None, help="number of jobs"
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.jobs is not None:
        nb.set_num_threads(args.jobs)

    x = compute_descriptors(args.file)

    results = []
    for n_samples in args.n_samples:
        logger("-" * 30)
        batch_size = int(min(500, 500 / (n_samples * x.shape[1] / 100000)))
        entropies = compute_entropy(x, n_samples, args.n_runs, batch_size=batch_size)
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
