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


@click.command("mutual_info")
@click.argument("file_1", required=1)
@click.argument("file_2", required=1)
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
def mutual_info(
    file_1,
    file_2,
    cutoff,
    nbrs,
    bandwidth,
    jobs,
    batch_size,
    output,
):
    if jobs is not None:
        nb.set_num_threads(jobs)

    # descriptors 1
    logger(f"Loading and creating descriptors for file {file_1}")
    dset_1 = read(file_1, index=":")

    with Timer() as t:
        x1 = get_descriptors(dset_1, k=nbrs, cutoff=cutoff)
    descriptor_time_1 = t.time
    logger(f"Descriptors built in: {format_time(descriptor_time_1)}")
    logger(f"Descriptors shape: {x1.shape}")

    # descriptors 2
    logger(f"Loading and creating descriptors for file {file_2}")

    dset_2 = read(file_2, index=":")

    with Timer() as t:
        x2 = get_descriptors(dset_2, k=nbrs, cutoff=cutoff)
    descriptor_time_2 = t.time
    logger(f"Descriptors built in: {format_time(descriptor_time_2)}")
    logger(f"Descriptors shape: {x2.shape}")

    # entropy 1
    logger("Computing entropy for dataset 1")
    with Timer() as t:
        entropy_1 = perfect_entropy(x1, h=bandwidth, batch_size=batch_size)
    entropy_time_1 = t.time
    logger(f"Entropy computed in: {format_time(entropy_time_1)}")
    logger(f"Dataset entropy: {entropy_1: .2f} (nats)")
    logger(f"Max theoretical entropy: {np.log(x1.shape[0]): .2f} (nats)")

    # entropy 2
    logger("Computing entropy for dataset 2")
    with Timer() as t:
        entropy_2 = perfect_entropy(x2, h=bandwidth, batch_size=batch_size)
    entropy_time_2 = t.time
    logger(f"Entropy computed in: {format_time(entropy_time_2)}")
    logger(f"Dataset entropy: {entropy_2: .2f} (nats)")
    logger(f"Max theoretical entropy: {np.log(x2.shape[0]): .2f} (nats)")

    # mutual information
    logger("Computing entropy for both datasets")
    x = np.concatenate([x1, x2], axis=0)
    with Timer() as t:
        joint_entropy = perfect_entropy(x, h=bandwidth, batch_size=batch_size)
    entropy_time_3 = t.time
    logger(f"Entropy computed in: {format_time(entropy_time_3)}")
    logger(f"Dataset entropy: {joint_entropy: .2f} (nats)")

    mutual = entropy_1 + entropy_2 - joint_entropy
    logger(f"Mutual Information: {mutual: .2f} (nats)")
    logger(f"Max mutual information: {np.log(x.shape[0]): .2f} (nats)")

    if output is not None:
        results = {
            "file_1": file_1,
            "file_2": file_2,
            "n_envs_1": x1.shape[0],
            "n_envs_2": x2.shape[0],
            "n_envs_joint": x.shape[0],
            "k": nbrs,
            "cutoff": cutoff,
            "bandwidth": bandwidth,
            "jobs": jobs,
            "entropy_1": entropy_1,
            "entropy_2": entropy_2,
            "entropy_joint": joint_entropy,
            "descriptor_time_1": descriptor_time_1,
            "descriptor_time_2": descriptor_time_2,
            "entropy_time_1": entropy_time_1,
            "entropy_time_2": entropy_time_2,
            "entropy_time_3": entropy_time_3,
        }

        with open(output, "w") as f:
            json.dump(results, f, indent=4)
