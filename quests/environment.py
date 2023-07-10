import numpy as np
from ase import Atoms
from matscipy.neighbours import neighbour_list as nbrlist


def extract_environment(atoms: Atoms, idx: int, cutoff: float, k: int) -> Atoms:
    """Extracts the k-nearest neighbors of the given Atoms
    and returns another Atoms object without periodic
    boundary conditions. This is useful for visualization
    purposes.

    Arguments:
    ----------
        atoms (Atoms): structure to be analyzed.
        idx (int): index of the environment to be isolated.
        cutoff (float): cutoff to consider when searching for
            nearest neighbors.
        k (int): number of nearest neighbors

    Returns:
    --------
        env (Atoms): isolated environment of `atoms[idx]`
    """
    i, j, d, D = nbrlist("ijdD", atoms, cutoff=cutoff)

    env = i == idx
    k_env = np.argsort(d[env])[:k]
    xyz = np.concatenate(
        [
            np.array([[0, 0, 0]]),
            D[env][k_env],
        ]
    )
    xyz = xyz + atoms.positions[idx]
    indices = [idx] + j[env][k_env].tolist()

    return Atoms(
        symbols=[atoms.symbols[x] for x in indices],
        positions=xyz,
    )
