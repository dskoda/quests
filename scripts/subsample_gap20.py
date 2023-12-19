import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import tqdm
from ase.io import read
from quests.cli.log import logger
from quests.descriptor import get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH
from quests.entropy import DEFAULT_BATCH
from quests.entropy import perfect_entropy


def compute_descriptors(file: str):
    dset = read(file, index=":")
    return get_descriptors(dset)


def sample_dataset(x: np.ndarray, n: int):
    size = x.shape[0]
    if size < n:
        return x

    indices = np.random.randint(0, size, n)
    return x[indices]


def compute_entropy(x: np.ndarray, n_samples: int, n_runs: int, batch_size: int = 2000):
    entropies = []
    for run in range(n_runs):
        xsample = sample_dataset(x, n_samples)
        n_envs = xsample.shape[0]
        xsample = xsample.reshape(n_envs, -1)

        entropy = perfect_entropy(xsample, batch_size=batch_size)
        entropies.append(entropy)

    return entropies


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing the GAP-20 dataset splits",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output folder")
    parser.add_argument(
        "-b",
        "--bandwidth",
        type=float,
        default=DEFAULT_BANDWIDTH,
        help=f"Bandwidth when computing the kernel (default: {DEFAULT_BANDWIDTH})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Size of the batches when computing the distances (default: {DEFAULT_BATCH})",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    files = sorted([f for f in os.listdir(args.folder) if f.endswith(".xyz")])
    samples = [0.2, 0.4, 0.6, 0.8, 1.0]

    logger("Subsampling the descriptors to obtain the entropy curves")
    for file in tqdm.tqdm(files):
        name = file.replace(".xyz", "")
        path = Path(os.path.join(args.output, name + ".json"))
        if path.exists():
            continue

        path.touch()

        x = compute_descriptors(os.path.join(args.folder, file))

        results = []
        for sample in samples:
            n = int(np.floor(len(x) * sample))
            n_runs = 5 if sample != 1.0 else 1
            entropies = compute_entropy(
                x, n, n_runs=n_runs, batch_size=args.batch_size
            )
            results.append(
                {
                    "file": file,
                    "name": name,
                    "n": n,
                    "frac": samples,
                    "n_runs": n_runs,
                    "entropies": entropies,
                    "mean_entropy": np.mean(entropies),
                }
            )

        with path.open("w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
