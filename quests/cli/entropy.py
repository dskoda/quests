import gc
import json
import os
import time

import click
import numba as nb
import numpy as np

from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH, perfect_entropy
from quests.tools.time import Timer

from .load_file import descriptors_from_file
from .log import format_time, logger


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
def entropy(
    file,
    cutoff,
    nbrs,
    bandwidth,
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

    logger(f"Loading and creating descriptors for file {file}")
    x, descriptor_time = descriptors_from_file(file, k=nbrs, cutoff=cutoff)
    logger(f"Descriptors built in: {format_time(descriptor_time)}")
    logger(f"Descriptors shape: {x.shape}")

    with Timer() as t:
        entropy = perfect_entropy(x, h=bandwidth, batch_size=batch_size)
    entropy_time = t.time
    logger(f"Entropy computed in: {format_time(entropy_time)}")

    logger(f"Dataset entropy: {entropy: .3f} (nats)")
    logger(f"Max theoretical entropy: {np.log(x.shape[0]): .3f} (nats)")

    if output is not None:
        results = {
            "file": file,
            "n_envs": x.shape[0],
            "k": nbrs,
            "cutoff": cutoff,
            "bandwidth": bandwidth,
            "jobs": jobs,
            "entropy": entropy,
            "descriptor_time": descriptor_time,
            "entropy_time": entropy_time,
        }

        with open(output, "w") as f:
            json.dump(results, f, indent=4)
