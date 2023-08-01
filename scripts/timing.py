#!/usr/bin/python3
import argparse
import time

from ase.io import read
from quests.descriptor import QUESTS


def get_args():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Benchmarks the timings between the get_descriptors approach"
    )

    # Add arguments to the parser
    parser.add_argument(
        "-i", "--input", type=str, default=None, help="Input xyz file to benchmark"
    )
    parser.add_argument(
        "-j", "--jobs", type=int, default=1, help="Number of jobs (optional, default=1)"
    )
    parser.add_argument(
        "-c", "--cutoff", type=float, default=5.0, help="Cutoff for neighbor list"
    )
    parser.add_argument(
        "-r",
        "--interaction_cutoff",
        type=float,
        default=4.0,
        help="Cutoff for interactions",
    )
    parser.add_argument(
        "-k", "--nbrs", type=int, default=32, help="Number of nearest neighbors"
    )
    parser.add_argument(
        "-t",
        "--threads",
        action="store_true",
        default=None,
        help="If True, uses threads instead of processes",
    )
    parser.add_argument(
        "-a",
        "--all_dset",
        action="store_true",
        default=None,
        help="If True, uses all the dataset",
    )
    parser.add_argument(
        "-n",
        "--numba",
        action="store_true",
        default=None,
        help="If True, uses numba instead of processes",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file (optional, default=None)",
    )

    # Parse the arguments from the command line
    return parser.parse_args()


def main():
    args = get_args()
    assert args.input is not None, "Please specify an input file"

    dset = read(args.input, index=":")
    q = QUESTS(
        cutoff=args.cutoff, k=args.nbrs, interaction_cutoff=args.interaction_cutoff
    )

    if args.all_dset:
        inp = dset
        fn = q.get_all_descriptors_parallel
    else:
        inp = dset[0]
        fn = q.get_descriptors_parallel

    start_time = time.time()
    x1, x2 = fn(inp, jobs=args.jobs)
    end_time = time.time()

    delta_ms = (end_time - start_time) * 1000

    print(f"Time: {delta_ms: .3f} ms")


if __name__ == "__main__":
    main()
