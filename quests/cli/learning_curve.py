import json
import os
import sys
import time
from typing import List

import click
import numba as nb
import numpy as np
from ase import Atoms
from ase.io import read

from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
from quests.entropy import (
    DEFAULT_BANDWIDTH,
    DEFAULT_BATCH,
    get_bandwidth,
    perfect_entropy,
)
from quests.tools.time import Timer

from .load_file import descriptors_from_file
from .log import format_time, logger


def sample_indices(size: int, n: int):
    if size < n:
        return np.arange(0, size, 1, dtype=int)

    return np.random.randint(0, size, n)


def get_sampling_fn(x: np.ndarray, fraction):
    # sample environments
    def sample_items():
        sample_size = int(len(x) * fraction)
        indices = sample_indices(len(x), sample_size)
        return x[indices]

    return sample_items


@click.command("learning_curve")
@click.argument("file", required=1)
@click.option(
    "-c",
    "--cutoff",
    type=float,
    default=DEFAULT_CUTOFF,
    help=f"Cutoff (in Å) for computing the neighbor list (default: {DEFAULT_CUTOFF:.1f})",
)
@click.option(
    "-k",
    "--nbrs",
    type=int,
    default=DEFAULT_K,
    help=f"Number of neighbors when creating the descriptor (default: {DEFAULT_K})",
)
@click.option(
    "-b",
    "--bandwidth",
    type=float,
    default=DEFAULT_BANDWIDTH,
    help=f"Bandwidth when computing the kernel (default: {DEFAULT_BANDWIDTH})",
)
@click.option(
    "-f",
    "--fractions",
    type=str,
    default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
    help="Comma-separated list of dataset fractions to sample (default: 0.1 to 0.9 every 0.1)",
)
@click.option(
    "-n",
    "--num_runs",
    type=int,
    default=3,
    help="Number of runs to resample for each fraction (default: 3)",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=None,
    help="Number of jobs to distribute the calculation in (default: all)",
)
@click.option(
    "--batch_size",
    type=int,
    default=DEFAULT_BATCH,
    help=f"Size of the batches when computing the distances (default: {DEFAULT_BATCH})",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="path to the json file that will contain the output\
            (default: no output produced)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="If True, overwrite the output file",
)
def learning_curve(
    file,
    cutoff,
    nbrs,
    bandwidth,
    fractions,
    num_runs,
    jobs,
    batch_size,
    output,
    overwrite,
):
    if output is not None and os.path.exists(output) and not overwrite:
        logger(f"Output file {output} exists. Aborting...")
        sys.exit(0)

    if jobs is not None:
        nb.set_num_threads(jobs)

    x, descriptor_time = descriptors_from_file(file, k=nbrs, cutoff=cutoff)

    fractions = [float(f) for f in fractions.split(",")]

    results = {
        "file": file,
        "n_envs": x.shape[0],
        "k": nbrs,
        "cutoff": cutoff,
        "bandwidth": bandwidth,
        "jobs": jobs,
        "fractions": fractions,
        "num_runs": num_runs,
        "descriptor_time": descriptor_time,
        "learning_curve": [],
    }

    for fraction in fractions:
        logger(f"Computing entropy for fraction: {fraction}")

        # determine how the dataset is going to be sampled
        sample_items = get_sampling_fn(x, fraction)

        # compute the entropy `num_runs` times
        entropies = []
        entropies_times = []
        for n in range(num_runs):
            xsample = sample_items()
            with Timer() as t:
                entropy = perfect_entropy(xsample, h=bandwidth, batch_size=batch_size)
            entropy_time = t.time

            entropies.append(float(entropy))
            entropies_times.append(entropy_time)

        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)

        logger(f"Entropy: {mean_entropy:.3f} ± {std_entropy:.3f} (nats)")
        logger(f"computed from {num_runs} runs.")
        logger(f"Max theoretical entropy: {np.log(len(xsample)):.3f} (nats)")

        results["learning_curve"].append(
            {
                "fraction": fraction,
                "entropies": entropies,
                "entropies_times": entropies_times,
                "mean_entropy": mean_entropy,
                "std_entropy": std_entropy,
            }
        )

    # log the results
    if output is not None:
        with open(output, "w") as f:
            json.dump(results, f, indent=4)

    logger("Learning curve computation completed.")
