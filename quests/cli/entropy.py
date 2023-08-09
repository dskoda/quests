import json
import os
import random
import sys
import time

import click
import numpy as np
from ase.io import read

from .log import logger
from quests.descriptor import QUESTS
from quests.entropy import EntropyEstimator
from quests.tree.pykdtree import TreePyKDTree


@click.command("entropy")
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
    "--nbrs_tree",
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
    default=None,
    help="If given, takes a sample of the dataset before computing \
            its entropy (default: uses the entire dataset)",
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
def entropy(
    file,
    cutoff,
    cutoff_interaction,
    nbrs_descriptor,
    nbrs_tree,
    bandwidth,
    kernel,
    sample,
    jobs,
    output,
):
    dset = read(file, index=":")
    if sample is not None and len(dset) > sample:
        dset = random.sample(dset, sample)

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

    start_time = time.time()
    tree = TreePyKDTree(x)
    tree.build()
    H = EntropyEstimator(
        x,
        h=bandwidth,
        nbrs=nbrs_tree,
        tree=tree,
        kernel=kernel,
    )
    end_time = time.time()
    build_time = end_time - start_time

    logger(f"Tree/entropy built in: {build_time * 1000: .2f} ms")

    start_time = time.time()
    entropy = H.dataset_entropy
    end_time = time.time()
    entropy_time = end_time - start_time

    logger(f"Entropy computed in: {entropy_time: .3f} s")
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
            "nbrs_tree": nbrs_tree,
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
