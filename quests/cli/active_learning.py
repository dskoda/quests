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


@click.command("active_learning")
@click.argument("reference", required=1)
@click.option(
    "-s",
    "--structures",
    type=int,
    default=None,
    help=f"Number of structures to sample from the reference dataset",
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
    "-g",
    "--generations",
    type=int,
    default=10,
    help=f"Number of generations of active learning steps (default: 10)",
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
    "--full",
    is_flag=True,
    default=False,
    help="If True, create an output file that is the union of the original dataset and the new dataset.",
)
def active_learning(
    reference,
    structures,
    n_steps,
    target,
    generations,
    cutoff,
    nbrs,
    bandwidth,
    jobs,
    batch_size,
    output,
    overwrite,
    full,
):
    if output is not None and os.path.exists(output) and not overwrite:
        logger(f"Output file {output} exists. Aborting...")
        sys.exit(0)

    if jobs is not None:
        nb.set_num_threads(jobs)

    dset = read(reference, index=":")

    logger("Starting active learning loop")
    new = []
    for gen in range(generations):
        with Timer() as t:
            initial = random.sample(dset + new, k=structures)

            for at in initial:
                best, res = augment_pbc(
                    at,
                    dset + new,
                    n_steps=n_steps,
                    target_dH=target,
                    k=nbrs,
                    cutoff=cutoff,
                )
                best.info["gen"] = gen + 1
                new.append(best)

        gen_time = t.time
        logger(f"Gen {gen + 1} complete in: {format_time(gen_time)}")

    if output is None:
        sys.exit()

    # exports the structure
    out = dset + new if full else new
    if output.endswith(".xyz"):
        write(output, out, format="extxyz")
        sys.exit()
