import json
import time

import click
import numpy as np
from ase.io import read

from .log import logger
from quests.descriptor import QUESTS
from quests.entropy import EntropyEstimator
from quests.tree.pykdtree import TreePyKDTree


@click.command("dentropy")
@click.argument("file_1", required=1)
@click.argument("file_2", required=1)
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
    "-j",
    "--jobs",
    type=int,
    default=1,
    help="Number of jobs to distribute the calculation in (default: 1)",
)
@click.option(
    "--kernel",
    type=str,
    default="epanechnikov",
    help="Name of the kernel to use when computing the delta entropy",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="path to the json file that will contain the output\
            (default: no output produced)",
)
def dentropy(
    file_1,
    file_2,
    cutoff,
    cutoff_interaction,
    nbrs_descriptor,
    nbrs_tree,
    bandwidth,
    jobs,
    kernel,
    output,
):
    dset_1 = read(file_1, index=":")
    dset_2 = read(file_2, index=":")

    q = QUESTS(
        cutoff=cutoff,
        k=nbrs_descriptor,
        interaction_cutoff=cutoff_interaction,
    )

    start_time = time.time()
    x1, x2 = q.get_all_descriptors_parallel(dset_1, jobs=jobs)
    x = np.concatenate([x1, x2], axis=1)
    end_time = time.time()
    descriptor_time_1 = end_time - start_time

    logger(f"Descriptors for dataset 1 built in: {descriptor_time_1 * 1000: .2f} ms")

    start_time = time.time()
    y1, y2 = q.get_all_descriptors_parallel(dset_2, jobs=jobs)
    y = np.concatenate([y1, y2], axis=1)
    end_time = time.time()
    descriptor_time_2 = end_time - start_time

    logger(f"Descriptors for dataset 2 built in: {descriptor_time_2 * 1000: .2f} ms")

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
    dH = H.delta_entropy(y)
    end_time = time.time()
    entropy_time = end_time - start_time

    logger(f"Delta entropy computed in: {entropy_time: .3f} s")
    logger(f"Avg dH: {dH.mean(): .2f}")
    logger(f"Std dH: {dH.std(): .2f}")
    logger(f"Max dH: {dH.max(): .2f}")
    logger(f"Min dH: {dH.min(): .2f}")

    if output is not None:
        results = {
            "file_1": file_1,
            "file_2": file_2,
            "cutoff": cutoff,
            "cutoff_interaction": cutoff_interaction,
            "nbrs_descriptor": nbrs_descriptor,
            "nbrs_tree": nbrs_tree,
            "bandwidth": bandwidth,
            "kernel": kernel,
            "jobs": jobs,
            "dH": dH.tolist(),
            "descriptor_time_1": descriptor_time_1,
            "descriptor_time_2": descriptor_time_2,
            "build_time": build_time,
            "entropy_time": entropy_time,
        }

        with open(output, "w") as f:
            json.dump(results, f)
