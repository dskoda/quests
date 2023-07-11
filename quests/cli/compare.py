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
@click.argument("file_1", required=1)
@click.argument("file_2", required=1)
@click.option(
    "-k",
    type=int,
    default=32,
    help="Number of nearest neighbors (default: 32)",
)
@click.option(
    "-c",
    "--cutoff",
    type=float,
    default=5.0,
    help="Cutoff (in Å) for considering interactions between atoms (default: 5.0)",
)
@click.option(
    "-m",
    "--metric",
    type=str,
    default="frobenius",
    help="Name of the metric to use when comparing the datasets (default: frobenius)",
)
@click.option(
    "-o",
    "--output",
    type=str,
    default="./output.csv",
    help="path to the csv file that will contain the output (default: output.csv)",
)
@click.option(
    "-p",
    "--nprocs",
    type=int,
    default=1,
    help="number of processors to use (default: 1)",
)
def compare(file_1, file_2, k, cutoff, metric, output, nprocs):
    """Compares different files according to the QUESTS descriptor.

        FILE_1: path to first file \n
        FILE_2: path to second file
    """
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
    df.to_csv(output)
