import json
import time

import click
import numba as nb
import numpy as np
from ase.io import read

from .log import format_time
from .log import logger
from quests.descriptor import DEFAULT_CUTOFF
from quests.descriptor import DEFAULT_K
from quests.descriptor import get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH
from quests.entropy import DEFAULT_BATCH
from quests.entropy import perfect_entropy
from quests.tools.time import Timer


@click.command("entropy")
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
    "-j",
    "--jobs",
    type=int,
    default=None,
    help="Number of jobs to distribute the calculation in (default: all)",
)
@click.option(
    "-b",
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
def entropy(
    file,
    cutoff,
    nbrs,
    bandwidth,
    kernel,
    jobs,
    batch_size,
    output,
):
    if jobs is not None:
        nb.set_num_threads(jobs)

    dset = read(file, index=":")

    with Timer() as t:
        x = get_descriptors(dset, k=nbrs, cutoff=cutoff)
    descriptor_time = t.time
    logger(f"Descriptors built in: {format_time(descriptor_time)}")

    with Timer() as t:
        entropy = perfect_entropy(x, h=bandwidth, batch_size=batch_size)
    entropy_time = t.time
    logger(f"Entropy computed in: {format_time(entropy_time)}")

    logger(f"Dataset entropy: {entropy: .2f} (nats)")

    if output is not None:
        results = {
            "file": file,
            "cutoff": cutoff,
            "bandwidth": bandwidth,
            "jobs": jobs,
            "entropy": entropy,
            "descriptor_time": descriptor_time,
            "entropy_time": entropy_time,
        }

        with open(output, "w") as f:
            json.dump(results, f, indent=4)
