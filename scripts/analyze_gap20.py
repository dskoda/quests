import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import tqdm
from ase.io import read
from quests.cli.log import format_time
from quests.cli.log import logger
from quests.descriptor import get_descriptors
from quests.entropy import DEFAULT_BANDWIDTH
from quests.entropy import DEFAULT_BATCH
from quests.entropy import delta_entropy
from quests.entropy import perfect_entropy


def compute_descriptors(file: str):
    dset = read(file, index=":")
    return get_descriptors(dset)


def get_name(f1: str, f2: str):
    n1 = f1.replace(".xyz", "")
    n2 = f2.replace(".xyz", "")

    return n1 + "-" + n2


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing the GAP-20 dataset splits",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output folder")
    parser.add_argument(
        "-b",
        "--bandwidth",
        type=float,
        default=DEFAULT_BANDWIDTH,
        help=f"Bandwidth when computing the kernel (default: {DEFAULT_BANDWIDTH})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Size of the batches when computing the distances (default: {DEFAULT_BATCH})",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    files = sorted([f for f in os.listdir(args.folder) if f.endswith(".xyz")])
    logger(f"Computing descriptors for {len(files)} files")
    x = {}
    for f in tqdm.tqdm(files):
        x[f] = compute_descriptors(os.path.join(args.folder, f))

    logger("Computing pairwise delta entropies")
    for f1, f2 in tqdm.tqdm(itertools.product(files, repeat=2)):
        name = get_name(f1, f2)
        path = Path(os.path.join(args.output, name + ".json"))
        if path.exists():
            continue

        path.touch()

        delta = delta_entropy(
            x[f1], x[f2], h=args.bandwidth, batch_size=args.batch_size
        )

        data = {
            "root_path": args.folder,
            "file_1": f1,
            "file_2": f2,
            "bandwidth": args.bandwidth,
            "delta_entropy": list(delta),
        }

        with path.open("w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
