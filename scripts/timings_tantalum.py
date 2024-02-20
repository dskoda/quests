import os
import gc
import sys
import json
import time
import itertools
import argparse

import click
import numba as nb
import numpy as np

from quests.entropy import DEFAULT_BANDWIDTH
from quests.matrix import sumexp
from quests.cli.log import format_time, logger
from quests.tools.time import Timer
import pynndescent as nnd


N_NEIGHBORS = [50, 100]
N_QUERY = [3, 10, 30]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute the times for Ta as a function of n_neighbors from the index"
    )

    parser.add_argument(
        "test",
        type=str,
        help="Path to the npz file containing the Ta descriptors",
    )
    parser.add_argument(
        "ref",
        type=str,
        help="Path to the npz file containing the reference dataset",
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=None, help="Number of threads to use"
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output folder")

    return parser.parse_args()


def prepare_index(ref, n_neighbors):
    with Timer() as t:
        index = nnd.NNDescent(ref, n_neighbors=n_neighbors)
        index.prepare()

    index_time = t.time
    return index, index_time


def query_index(x, index, n):
    with Timer() as t:
        _, d = index.query(x, k=n)
        z = d / DEFAULT_BANDWIDTH
        p_x = sumexp(-0.5 * z**2)
        dH = -np.log(p_x)

    query_time = t.time

    return dH, query_time


def main():
    args = parse_arguments()

    if args.jobs is not None:
        nb.set_num_threads(jobs)

    logger("Loading files")
    with open(args.test, "rb") as f:
        x = np.load(f)

    with open(args.ref, "rb") as f:
        ref = np.load(f)

    logger("Initializing experiments")

    results = []
    for nbrs, n in itertools.product(N_NEIGHBORS, N_QUERY):
        index, index_time = prepare_index(ref, nbrs)
        _, query_time = query_index(x, index, n)

        logger(
            f"(nbrs={nbrs}, n={n}) -> index_time = {format_time(index_time)}, query_time = {format_time(query_time)}"
        )
        results.append(
            {
                "nbrs": nbrs,
                "n": n,
                "index_time": index_time,
                "query_time": query_time,
            }
        )

    if output is not None:
        with open(output, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
