import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


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
    i, j, d, D = neighbor_list("ijdD", atoms, cutoff=cutoff)

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


def estimate_neighbors(density: float, cutoff: float, molar_mass: float):
    """Estimate how many neighbors are expected to be within
        a given cutoff from the density of the material.

    Arguments:
    ----------
        density (float): density of the material in g/cm3
        cutoff (float): radius of the shell to be considered
        molar_mass (float): molar mass of the material per formula unit
            in g/mol.

    Returns:
    --------
        neighbors (float): average number of neighbors
    """
    # 1 mol = 6.02E23 atoms
    # 1 cm^3 = 1E24 Å^3
    num_atoms = density / molar_mass * 0.6022  # atoms/Å^3
    volume = 4 * np.pi * (cutoff**3) / 3

    return num_atoms * volume
