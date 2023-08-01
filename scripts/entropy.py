#!/usr/bin/python3
import argparse
import time

import numpy as np
from ase.io import read
from quests.descriptor import QUESTS
from quests.entropy import EntropyEstimator
from quests.tree.pykdtree import TreePyKDTree


def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Computes the entropy of a dataset")

    # Add arguments to the parser
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input xyz file to benchmark"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of jobs (optional, default=1)"
    )
    parser.add_argument(
        "-c", "--cutoff", type=float, default=6.0, help="Cutoff for neighbor list"
    )
    parser.add_argument(
        "-r",
        "--interaction_cutoff",
        type=float,
        default=5.0,
        help="Cutoff for interactions",
    )
    parser.add_argument(
        "-k",
        "--nbrs_descriptor",
        type=int,
        default=32,
        help="Number of nearest neighbors used to define the descriptor",
    )
    parser.add_argument(
        "-t",
        "--nbrs_tree",
        type=int,
        default=100,
        help="Number of nearest neighbors used to approximate the kernel",
    )
    parser.add_argument(
        "-b",
        "--bandwidth",
        type=float,
        default=0.015,
        help="Bandwidth for computation of the kernel",
    )

    # Parse the arguments from the command line
    return parser.parse_args()


def main():
    args = get_args()
    assert args.input is not None, "Please specify an input file"

    dset = read(args.input, index=":")
    q = QUESTS(
        cutoff=args.cutoff,
        k=args.nbrs_descriptor,
        interaction_cutoff=args.interaction_cutoff,
    )

    start_time = time.time()
    x1, x2 = q.get_all_descriptors_parallel(dset, jobs=args.jobs)
    x = np.concatenate([x1, x2], axis=1)
    end_time = time.time()
    delta_ms = (end_time - start_time) * 1000

    print(f"Descriptors built in: {delta_ms: .2f} ms")

    start_time = time.time()
    tree = TreePyKDTree(x)
    tree.build()
    H = EntropyEstimator(
        x,
        h=args.bandwidth,
        nbrs=args.nbrs_tree,
        tree=tree,
    )
    end_time = time.time()
    delta_ms = (end_time - start_time) * 1000

    print(f"Tree/entropy built in: {delta_ms: .2f} ms")

    start_time = time.time()
    entropy = H.dataset_entropy
    end_time = time.time()
    delta_s = (end_time - start_time)

    print(f"Entropy computed in: {delta_s: .3f} s")
    print(f"Dataset entropy: {entropy: .2f} (nats) for a bandwidth {args.bandwidth: 0.3f}")


if __name__ == "__main__":
    main()
