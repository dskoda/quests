import json
import time
import argparse

import numpy as np
import numba as nb
from ase.io import read
from ase import Atoms

from quests.cli.log import format_time
from quests.cli.log import logger
from quests.descriptor import get_descriptors
from quests.tools.time import Timer


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    parser.add_argument("file", type=str, help="Path to the file")
    parser.add_argument(
        "--n_threads", type=int, nargs="+", help="List of number of threads to consider"
    )
    parser.add_argument("-s", "--single", action="store_true", default=False, help="If True, loads the file as a single Atoms")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file")

    return parser.parse_args()


def main():
    args = parse_arguments()
    logger(f"Testing scaling for {args.file}")
    if args.single:
        dset = [read(args.file)]
    else:
        dset = read(args.file, index=":")

    # call once to compile
    x = get_descriptors(dset)

    results = []
    for n_threads in args.n_threads:
        nb.set_num_threads(n_threads)

        with Timer() as t:
            x = get_descriptors(dset)
        descriptor_time = t.time

        logger(f"{n_threads} threads: {format_time(descriptor_time)}")

        results.append(
            {
                "n_threads": n_threads,
                "size": x.shape[0],
                "time": descriptor_time,
            }
        )

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
