#!/usr/bin/env python3

import os
from ase.io import read, write

if __name__ == "__main__":
    dset = read("gap20-full.xyz", index=":")
    subsets = {}

    for at in dset:
        cfg = at.info["config_type"]
        subsets[cfg] = subsets.get(cfg, []) + [at]

    outdir = "gap20"
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for name, subset in subsets.items():
        write(f"{outdir}/{name}.xyz", subset, format="extxyz")
