import os
import sys

import click
from ase.io import read

from .log import logger
from quests.flex.descriptor import QUESTS
from quests.flex.distance import compare_datasets


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
    default="euclidean",
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
    if os.path.exists(output):
        logger(f"File {output} already exists! Exiting...")
        sys.exit(1)

    logger(f"Reading files {file_1} and {file_2}...")
    dset1 = read(file_1, index=":")
    dset2 = read(file_2, index=":")

    q = QUESTS(cutoff=cutoff, k=k)

    logger("Comparing datasets")
    df = compare_datasets(
        dset1,
        dset2, 
        q,
        metric=metric,
        nprocs=nprocs
    )

    logger("Saving results")
    df.to_csv(output)
