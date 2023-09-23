import os
import sys
import json
import time

from typing import Iterable
import click
import numpy as np
from ase.io import read

from .log import logger
from quests.descriptor import QUESTS
from quests.entropy import EntropyEstimator
from quests.pbc import add_box


def sample_indices(size: int, n: int):
    if size < n:
        return np.arange(0, size, 1, dtype=int)

    return np.random.randint(0, size, n)


@click.command("entropy_sampler")
@click.argument("file", required=1)
@click.option(
    "-c",
    "--cutoff",
    type=float,
    default=6.0,
    help="Cutoff (in Å) for computing the neighbor list (default: 6.0)",
)
@click.option(
    "-r",
    "--cutoff_interaction",
    type=float,
    default=5.0,
    help="Cutoff (in Å) for considering interactions between atoms \
            (default: 5.0)",
)
@click.option(
    "-k",
    "--nbrs_descriptor",
    type=int,
    default=32,
    help="Number of neighbors when creating the descriptor (default: 32)",
)
@click.option(
    "-t",
    "--nbrs_finder",
    type=int,
    default=100,
    help="Number of neighbors when computing the kernel (default: 100)",
)
@click.option(
    "-b",
    "--bandwidth",
    type=float,
    default=0.015,
    help="Bandwidth when computing the kernel (default: 0.015)",
)
@click.option(
    "--kernel",
    type=str,
    default="gaussian",
    help="Name of the kernel to use when computing the delta entropy (default: gaussian)",
)
@click.option(
    "-s",
    "--sample",
    type=int,
    default=1000,
    help="If given, takes a sample of the environments before computing \
            its entropy (default: uses the entire dataset)",
)
@click.option(
    "--sample_dataset",
    is_flag=True,
    default=False,
    help="If True, subsamples the dataset as opposed to the environment.",
)
@click.option(
    "-n",
    "--num_runs",
    type=int,
    default=20,
    help="Number of runs to resample (default: 20)",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=1,
    help="Number of jobs to distribute the calculation in (default: 1)",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="path to the json file that will contain the output\
            (default: no output produced)",
)
def entropy_sampler(
    file,
    cutoff,
    cutoff_interaction,
    nbrs_descriptor,
    nbrs_finder,
    bandwidth,
    kernel,
    sample,
    sample_dataset,
    num_runs,
    jobs,
    output,
):
    if output is not None and os.path.exists(output):
        logger(f"Output file {output} exists. Aborting...")
        sys.exit(0)

    logger(f"Sampling entropies for: {file}")
    dset = read(file, index=":")
    dset = [add_box(atoms) for atoms in dset]

    q = QUESTS(
        cutoff=cutoff,
        k=nbrs_descriptor,
        interaction_cutoff=cutoff_interaction,
    )

    start_time = time.time()
    x1, x2 = q.get_all_descriptors_parallel(dset, jobs=jobs)
    x = np.concatenate([x1, x2], axis=1)
    end_time = time.time()
    descriptor_time = end_time - start_time

    logger(f"Descriptors built in: {descriptor_time * 1000: .2f} ms")

    # if dataset is smaller than sample, no need to
    # run multiple times
    if len(x) <= sample:
        num_runs = 1

    if sample_dataset:
        # create indices for the dataset
        start = 0
        dset_indices = []
        for i, atoms in enumerate(dset):
            num_atoms = len(atoms)
            idx = np.arange(start, start + num_atoms, 1, dtype=int)
            dset_indices.append(idx)
            start = start + num_atoms

        def sample_items():
            indices = sample_indices(len(dset_indices), sample) 
            x_indices = np.concatenate([
                dset_indices[i] for i in indices
            ])
            return x[x_indices]

    else:
        def sample_items():
            indices = sample_indices(len(x), sample)
            return x[indices]

        
    # computing the entropies
    entropies = []
    for n in range(num_runs):
        xsample = sample_items()

        H = EntropyEstimator(
            xsample,
            h=bandwidth,
            nbrs=nbrs_finder,
            kernel=kernel,
        )

        entropy = H.dataset_entropy
        entropies.append(entropy)

        logger(f"Entropy {n:02d}: {entropy: .2f} (nats)")

    if output is not None:
        results = {
            "file": file,
            "cutoff": cutoff,
            "cutoff_interaction": cutoff_interaction,
            "nbrs_descriptor": nbrs_descriptor,
            "nbrs_finder": nbrs_finder,
            "bandwidth": bandwidth,
            "sample": sample,
            "num_runs": num_runs,
            "jobs": jobs,
            "entropies": entropies,
        }

        with open(output, "w") as f:
            json.dump(results, f, indent=4)
