import itertools
import os

import click
import multiprocess as mp
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read

from quests.descriptor import QUESTS
from quests.distance import compare_matrices


@click.command("compare")
@click.argument(
    "file_1",
    type=str,
    help="First file to be compared",
)
@click.argument(
    "file_2",
    type=str,
    help="second file to be compared",
)
@click.option(
    "-k",
    type=int,
    default=32,
    help="Number of nearest neighbors",
)
@click.option(
    "-c",
    "--cutoff",
    type=float,
    default=5.0,
    help="Cutoff (in Å) for considering interactions between atoms",
)
@click.option(
    "-m",
    "--metric",
    type=str,
    default="frobenius",
    help="Name of the metric to use when comparing the datasets",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default="./output.json",
    help="path to the JSON file that will contain the output",
)
@click.option(
    "-p",
    "--nprocs",
    type=int,
    default=1,
    help="number of processors to use (default: 1)",
)
def compare(file_1, file_2, k, cutoff, metric, output, nprocs):
    dset1 = read(file_1, index=":")
    dset2 = read(file_2, index=":")

    q = QUESTS(cutoff=cutoff, k=k)

    q1 = [
        q.get_descriptors(at)
        for at in dset1
    ]
    q2 = [
        q.get_descriptors(at)
        for at in dset2
    ]

    def worker_fn(ij):
        i, j = ij
        x1, x2 = q1[i]
        y1, y2 = q2[j]
        dm = compare_matrices(x1, x2, y1, y2, metric=metric)
        return {
            "dset1": file_1,
            "dset2": file_2,
            "index1": i,
            "index2": j,
            "min": dm.min(),
            "max": dm.max(),
            "mean": dm.mean(),
            "std": dm.std(),
            "q1": np.percentile(dm, 25),
            "q3": np.percentile(dm, 75),
        }
    
    results = []
    p = mp.Pool(nprocs)
    iterator = itertools.product(range(len(q1)), range(len(q2)))

    for result in p.imap_unordered(worker_fn, iterator, chunksize=1):
        results.append(result)

    df = pd.DataFrame(results)
    df.to_json(output)
