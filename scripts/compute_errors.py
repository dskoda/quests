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
from quests.entropy import weighted_kernel_sum


def compute_descriptors(file: str):
    dset = read(file, index=":")
    return get_descriptors(dset)


def compute_descriptors_and_errors(file: str):
    dset = read(file, index=":")
    x = get_descriptors(dset)

    # estimate the errors
    errs = []
    for atoms in dset:
        true = atoms.arrays["forces"]
        pred = atoms.arrays["MACE_forces"]
        err = np.linalg.norm(true - pred, axis=1)
        errs.append(err)

    errs = np.concatenate(errs)

    return x, errs


def get_name(fname: str):
    return os.path.basename(fname).replace(".xyz", "")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse command line arguments.")

    parser.add_argument(
        "reference",
        type=str,
        help="Path to the file that is the reference for the dH computations \
                and contains the predictions from MACE.",
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing the other splits",
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

    ref, errs = compute_descriptors_and_errors(args.reference)

    files = sorted([f for f in os.listdir(args.folder) if f.endswith(".xyz")])
    logger(f"Computing pairwise entropies for {len(files)} files")
    for f in tqdm.tqdm(files):
        name = get_name(f)
        path = Path(os.path.join(args.output, name + ".json"))
        if path.exists():
            continue

        path.touch()

        x = compute_descriptors(os.path.join(args.folder, f))

        pred_errors, p_x = weighted_kernel_sum(x, ref, errs, h=args.bandwidth, batch_size=args.batch_size)
        logp = -np.log(p_x)

        data = {
            "reference": args.reference,
            "root_path": args.folder,
            "dset": f,
            "bandwidth": args.bandwidth,
            "pred_errors": list(pred_errors),
            "delta_entropy": list(logp),
        }

        with path.open("w") as f:
            json.dump(data, f)


if __name__ == "__main__":
    main()
