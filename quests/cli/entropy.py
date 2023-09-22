import json
import time

import click
import numpy as np
from ase.io import read

from .log import format_time
from .log import logger
from quests.descriptor import descriptor_nopbc
from quests.descriptor import descriptor_pbc
from quests.entropy import perfect_entropy


@click.command("entropy")
@click.argument("file", required=1)
@click.option(
    "-c",
    "--cutoff",
    type=float,
    default=5.0,
    help="Cutoff (in Å) for computing the neighbor list (default: 5.0)",
)
@click.option(
    "-k",
    "--nbrs",
    type=int,
    default=32,
    help="Number of neighbors when creating the descriptor (default: 32)",
)
@click.option(
    "-b",
    "--bandwidth",
    type=float,
    default=0.015,
    help="Bandwidth when computing the kernel (default: 0.015)",
)
@click.option(
    "-s",
    "--sample",
    type=int,
    default=None,
    help="If given, takes a sample of the environments before computing \
            its entropy (default: uses the entire dataset)",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=None,
    help="Number of jobs to distribute the calculation in (default: all)",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="path to the json file that will contain the output\
            (default: no output produced)",
)
def entropy(
    file,
    cutoff,
    nbrs,
    bandwidth,
    kernel,
    sample,
    jobs,
    output,
):
    dset = read(file, index=":")

    start_time = time.time()
    x1, x2 = q.get_all_descriptors_parallel(dset, jobs=jobs)
    x = np.concatenate([x1, x2], axis=1)
    end_time = time.time()
    descriptor_time = end_time - start_time

    logger(f"Descriptors built in: {format_time(descriptor_time)}")

    if sample is not None:
        if len(x) > sample:
            i = np.random.randint(0, len(x), sample)
            x = x[i]
        else:
            sample = len(x)

    start_time = time.time()
    H = EntropyEstimator(
        x,
        h=bandwidth,
        nbrs=nbrs_finder,
        kernel=kernel,
    )
    end_time = time.time()
    build_time = end_time - start_time

    logger(f"Tree/entropy built in: {format_time(build_time)}")

    start_time = time.time()
    entropy = H.dataset_entropy
    end_time = time.time()
    entropy_time = end_time - start_time

    logger(f"Entropy computed in: {format_time(entropy_time)}")
    logger(
        f"Dataset entropy: {entropy: .2f} (nats)"
        + f"for a bandwidth {bandwidth: 0.3f}"
    )

    if output is not None:
        results = {
            "file": file,
            "cutoff": cutoff,
            "cutoff_interaction": cutoff_interaction,
            "nbrs_descriptor": nbrs_descriptor,
            "nbrs_finder": nbrs_finder,
            "bandwidth": bandwidth,
            "sample": sample,
            "jobs": jobs,
            "entropy": entropy,
            "descriptor_time": descriptor_time,
            "build_time": build_time,
            "entropy_time": entropy_time,
        }

        with open(output, "w") as f:
            json.dump(results, f, indent=4)
