import numpy as np
from ase import Atoms


def add_box(atoms: Atoms, vacuum: float = 25) -> Atoms:
    """Adds a periodic box for an atoms object
    in the case of non-periodic systems with
    cell equals to the zero matrix. This is useful
    because some molecular datasets are loaded
    with no unit cell.
    """

    if atoms.pbc.any():
        return atoms

    if np.linalg.det(atoms.cell) > 0:
        return atoms

    r = atoms.positions
    L = (r.max(0) - r.min(0)).max() + vacuum
    diag = np.array([L, L, L])

    box = np.diag(diag)
    newpos = r - r.mean(0) + diag / 2

    new = atoms.copy()
    new.positions = newpos
    new.cell = box

    return new
