import gc
import json
import os
import sys
import time

import click
import numba as nb
import numpy as np

from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
from quests.entropy import (
    DEFAULT_BANDWIDTH,
    DEFAULT_BATCH,
    DEFAULT_GRAPH_NBRS,
    DEFAULT_UQ_NBRS,
    approx_delta_entropy,
)
from quests.tools.time import Timer

from .load_file import descriptors_from_file
from .log import format_time, logger


@click.command("approx_dH")
@click.argument("test", required=1)
@click.argument("reference", required=1)
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
    "-n",
    "--uq_nbrs",
    type=int,
    default=DEFAULT_UQ_NBRS,
    help=f"Number of neighbors when creating the descriptor (default: {DEFAULT_UQ_NBRS})",
)
@click.option(
    "-g",
    "--graph_nbrs",
    type=int,
    default=DEFAULT_GRAPH_NBRS,
    help=f"Number of neighbors when creating the index (default: {DEFAULT_GRAPH_NBRS})",
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
def approx_dH(
    test,
    reference,
    cutoff,
    nbrs,
    uq_nbrs,
    graph_nbrs,
    bandwidth,
    jobs,
    output,
    overwrite,
):
    if output is not None and os.path.exists(output) and not overwrite:
        logger(f"Output file {output} exists. Aborting...")
        sys.exit(0)

    if jobs is not None:
        nb.set_num_threads(jobs)

    x, _ = descriptors_from_file(test, nbrs, cutoff)
    ref, _ = descriptors_from_file(reference, nbrs, cutoff)

    logger("Computing dH...")
    with Timer() as t:
        delta = approx_delta_entropy(
            x, ref, h=bandwidth, n=uq_nbrs, graph_neighbors=graph_nbrs
        )
    entropy_time = t.time
    logger(f"dH computed in: {format_time(entropy_time)}")

    if output is not None:
        results = {
            "reference_file": reference,
            "test_file": test,
            "test_envs": x.shape[0],
            "ref_envs": ref.shape[0],
            "k": nbrs,
            "n": uq_nbrs,
            "cutoff": cutoff,
            "bandwidth": bandwidth,
            "jobs": jobs,
            "delta_entropy": list(delta.astype(float)),
            "time": entropy_time,
        }

        with open(output, "w") as f:
            json.dump(results, f)
