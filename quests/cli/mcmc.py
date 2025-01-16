import gc
import json
import os
import sys
import time
import random

import click
import numba as nb
import numpy as np
from ase.io import read, write

from quests.mcmc import augment_pbc, DEFAULT_TARGET
from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH
from quests.tools.time import Timer

from .log import format_time, logger


@click.command("mcmc")
@click.argument("reference", required=1)
@click.option(
    "-i",
    "--index",
    type=int,
    default=None,
    help=f"Index of the object that will serve as starting point. If not given, randomly samples one from the dataset",
)
@click.option(
    "-n",
    "--n_steps",
    type=int,
    default=1000,
    help=f"Number of Monte Carlo steps (default: 1000)",
)
@click.option(
    "-t",
    "--target",
    type=int,
    default=DEFAULT_TARGET,
    help=f"Target dH for the generation of new structures (default: {DEFAULT_TARGET:.0f})",
)
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
@click.option(
    "--compare",
    is_flag=True,
    default=False,
    help="If True, create an output file that compares the initial and final structures",
)
def mcmc(
    reference,
    index,
    n_steps,
    target,
    cutoff,
    nbrs,
    bandwidth,
    jobs,
    batch_size,
    output,
    overwrite,
    compare,
):
    if output is not None and os.path.exists(output) and not overwrite:
        logger(f"Output file {output} exists. Aborting...")
        sys.exit(0)

    if jobs is not None:
        nb.set_num_threads(jobs)

    dset = read(reference, index=":")

    if index is None:
        at = random.sample(dset, k=1)
    elif index < 0 or index > len(dset):
        raise ValueError("Index should be between 0 and the dataset size ({len(dset)})")
    else:
        at = dset[index]

    logger("Sampling new structure")
    with Timer() as t:
        best, res = augment_pbc(
            at, dset, n_steps=n_steps, target_dH=target, k=nbrs, cutoff=cutoff
        )
    entropy_time = t.time
    logger(f"Structure sampled in: {format_time(entropy_time)}")

    if output is None:
        sys.exit()

    # exports the structure
    out = [at, best] if compare else best
    if output.endswith(".xyz"):
        write(output, out, format="extxyz")
        sys.exit()
