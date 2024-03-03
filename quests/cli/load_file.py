import numpy as np
from ase.io import read

from quests.descriptor import get_descriptors
from quests.tools.time import Timer


def descriptors_from_file(file, k, cutoff):
    if file.endswith(".npz"):
        with Timer() as t:
            with open(file, "rb") as f:
                x = np.load(f)
        descriptor_time = t.time
        return x, descriptor_time

    dset = read(file, index=":")

    with Timer() as t:
        x = get_descriptors(dset, k=k, cutoff=cutoff)
    descriptor_time = t.time

    return x, descriptor_time
