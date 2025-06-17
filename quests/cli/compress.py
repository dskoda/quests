import json
import os
import sys

import click
import numba as nb
import numpy as np
from ase.io import read, write

from quests.compression.compress import DatasetCompressor
from quests.descriptor import DEFAULT_CUTOFF, DEFAULT_K, get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH, DEFAULT_BATCH
from quests.tools.time import Timer

from .log import format_time, logger


@click.command("compress")
@click.argument("file", required=1)
@click.option(
    "-s",
    "--size",
    type=float,
    default=0.5,
    help=f"Size of the final dataset (either in % or in number of structures) (default: 0.5)",
)
@click.option(
    "-m",
    "--method",
    type=str,
    default="msc",
    help=f"Method to use when compressing the dataset (default: msc)",
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
def compress(
    file,
    size,
    method,
    cutoff,
    nbrs,
    bandwidth,
    jobs,
    batch_size,
    output,
    overwrite,
):
    if output is not None and os.path.exists(output) and not overwrite:
        logger(f"Output file {output} exists. Aborting...")
        sys.exit(0)

    if jobs is not None:
        nb.set_num_threads(jobs)

    logger(f"Compressing {file} using method {method}")

    dset = read(file, index=":")

    descriptor_fn = lambda ds: get_descriptors([ds], k=nbrs, cutoff=cutoff)
    compressor = DatasetCompressor(
        dset, descriptor_fn, bandwidth=bandwidth, batch_size=batch_size
    )

    if (size > 0) and (size < 1):
        size = int(np.round(len(dset) * size))
    else:
        size = int(size)

    with Timer() as t:
        selected = compressor.get_indices(method, size)
    compress_time = t.time

    logger(f"Compressed dataset in {format_time(compress_time)}")
    logger("Computing metrics...")

    with Timer() as t:
        comp_H = float(compressor.entropy(selected))
        comp_D = float(compressor.diversity(selected))
        overlap = float(compressor.overlap(selected))
        orig_structs = len(dset)
        comp_structs = len(selected)
        orig_envs = sum([len(at) for at in dset])
        comp_envs = sum([len(dset[i]) for i in selected])

    metrics_time = t.time
    logger(f"Computed metrics in {format_time(metrics_time)}")
    logger(f"Entropy: {comp_H:.2f}")
    logger(f"Diversity: {comp_D:.2f}")
    logger(f"# Structs: {comp_structs:.0f} (out of {orig_structs:.0f})")
    logger(f"# Envs: {comp_envs:.0f} (out of {orig_envs:.0f})")
    logger(f"Overlap: {overlap * 100:.1f}%")

    if output is None:
        sys.exit()

    if output.endswith(".xyz"):
        new = [dset[i] for i in selected]
        write(output, new, format="extxyz")
        sys.exit()

    results = {
        "file": file,
        "k": nbrs,
        "cutoff": cutoff,
        "bandwidth": bandwidth,
        "jobs": jobs,
        "compress_time": compress_time,
        "metrics_time": metrics_time,
        "orig_n_structs": orig_structs,
        "orig_n_envs": orig_envs,
        "comp_H": comp_H,
        "comp_D": comp_D,
        "comp_n_structs": comp_structs,
        "comp_n_envs": comp_envs,
        "overlap": overlap,
        "selected": [int(i) for i in selected],
    }
    with open(output, "w") as f:
        json.dump(results, f, indent=4)
