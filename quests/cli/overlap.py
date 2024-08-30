import gc
import json
import os
import sys
import time

import click
import numba as nb
import numpy as np
from ase.io import read, write

from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH, delta_entropy
from quests.tools.time import Timer

from .load_file import descriptors_from_file
from .log import format_time, logger


@click.command("overlap")
@click.argument("test_set", required=1)
@click.argument("reference_set", required=1)
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
    "-e",
    "--eps",
    type=float,
    default=1e-5,
    help="Threshold for considering environments as overlapping (default: 1e-3)",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="path to the json file that will contain the output (default: no output produced)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="If True, overwrite the output file",
)
def overlap(
    test_set,
    reference_set,
    cutoff,
    nbrs,
    bandwidth,
    jobs,
    batch_size,
    eps,
    output,
    overwrite,
):
    if output is not None and os.path.exists(output) and not overwrite:
        logger(f"Output file {output} exists. Aborting...")
        sys.exit(0)

    if jobs is not None:
        nb.set_num_threads(jobs)

    xt, _ = descriptors_from_file(test_set, nbrs, cutoff)
    xr, _ = descriptors_from_file(reference_set, nbrs, cutoff)

    logger("Computing overlap...")
    with Timer() as t:
        delta = delta_entropy(xt, xr, h=bandwidth, batch_size=batch_size)
        overlap_value = (delta < eps).mean()
    overlap_time = t.time
    logger(f"Overlap computed in: {format_time(overlap_time)}")
    logger(f"Overlap value: {overlap_value:.4f}")

    if output is None:
        sys.exit()

    results = {
        "test_file": test_set,
        "reference_file": reference_set,
        "test_envs": xt.shape[0],
        "reference_envs": xr.shape[0],
        "k": nbrs,
        "cutoff": cutoff,
        "bandwidth": bandwidth,
        "jobs": jobs,
        "eps": eps,
        "overlap": float(overlap_value),
        "computation_time": overlap_time,
    }

    with open(output, "w") as f:
        json.dump(results, f, indent=4)

    logger(f"Results written to {output}")
