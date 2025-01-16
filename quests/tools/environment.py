import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def extract_environment(atoms: Atoms, idx: int, cutoff: float, k: int) -> Atoms:
    """
    Extracts the k-nearest neighbors of the given Atoms and returns another Atoms object without periodic boundary conditions.

    This function is useful for visualization purposes.

    Parameters
    ----------
    atoms : Atoms
        Structure to be analyzed.
    idx : int
        Index of the environment to be isolated.
    cutoff : float
        Cutoff to consider when searching for nearest neighbors.
    k : int
        Number of nearest neighbors.

    Returns
    -------
    Atoms
        Isolated environment of `atoms[idx]`.
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


def estimate_neighbors(density: float, cutoff: float, molar_mass: float) -> float:
    """
    Estimate how many neighbors are expected to be within a given cutoff from the density of the material.

    Parameters
    ----------
    density : float
        Density of the material in g/cm3.
    cutoff : float
        Radius of the shell to be considered.
    molar_mass : float
        Molar mass of the material per formula unit in g/mol.

    Returns
    -------
    float
        Average number of neighbors.
    """
    # 1 mol = 6.02E23 atoms
    # 1 cm^3 = 1E24 Å^3
    num_atoms = density / molar_mass * 0.6022  # atoms/Å^3
    volume = 4 * np.pi * (cutoff**3) / 3

    return num_atoms * volume
