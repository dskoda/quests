import math
import random
import numpy as np
import numba as nb
from typing import List

from ase import Atoms

from quests.descriptor import (
    get_descriptors,
    descriptor_pbc,
    DEFAULT_K,
    DEFAULT_CUTOFF,
    EPS,
)
from quests.entropy import (
    delta_entropy,
    diversity,
    DEFAULT_BANDWIDTH,
    DEFAULT_BATCH,
    DEFAULT_GRAPH_NBRS,
    DEFAULT_UQ_NBRS,
)
from quests.matrix import sumexp


DEFAULT_TARGET = 30
DEFAULT_GRAPH_NBRS = 10


@nb.njit(fastmath=True, cache=True)
def random_translation(points: np.ndarray, std: float) -> np.ndarray:
    """Apply random translation to points."""
    matrix = std * np.random.randn(*points.shape)
    return points + matrix


@nb.njit(fastmath=True, cache=True)
def accept(new_score, old_score, kT=1):
    """Acceptance criterion for a Monte Carlo movement"""
    ds = new_score - old_score
    p = np.exp(-ds / kT)
    return p > random.random()


@nb.njit(fastmath=True, cache=True)
def temperature_ramp(
    step: int, n_steps: int, cutoff: float = 0.9, tmin: float = 1e-3
) -> float:
    kT = 1 - (step / (n_steps * cutoff))
    return max(kT, tmin)


@nb.njit(fastmath=True, cache=True)
def annealing(
    step: int, n_steps: int, n_cycles: int = 3, cutoff: float = 0.9, tmin: float = 1e-3
) -> float:
    period = n_steps // n_cycles
    phase = (step % period) / period
    kT = 1 - phase
    return max(kT, tmin)


def compute_score(
    y: np.ndarray,
    ref: np.ndarray,
    method: str = "target",
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    target_dH: float = DEFAULT_TARGET,
) -> float:
    """Score: lower is better"""
    dH = delta_entropy(y, ref, h=h, batch_size=batch_size)

    if method.lower() == "greedy":
        return -dH.max()

    if method.lower() == "average":
        return -dH.mean()

    if method.lower() == "target":
        return np.abs(dH - target_dH).mean()

    raise ValueError(f"Method {method} does not exist")


def compute_approx_score(
    y: np.ndarray,
    index,
    method: str = "target",
    h: float = DEFAULT_BANDWIDTH,
    batch_size: int = DEFAULT_BATCH,
    target_dH: float = DEFAULT_TARGET,
    n: int = DEFAULT_UQ_NBRS,
) -> float:
    """Score: lower is better"""
    _, d = index.query(y, k=n)
    z = d / h
    p_y = sumexp(-0.5 * z**2)
    dH = -np.log(p_y)

    if method.lower() == "greedy":
        return -dH.max()

    if method.lower() == "average":
        return -dH.mean()

    if method.lower() == "target":
        return np.abs(dH - target_dH).mean()

    raise ValueError(f"Method {method} does not exist")


def get_index(x: np.ndarray, graph_neighbors: int = DEFAULT_GRAPH_NBRS, **kwargs):
    import pynndescent as nnd

    index = nnd.NNDescent(x, n_neighbors=graph_neighbors, **kwargs)
    index.prepare()

    return index


def augment_pbc(
    atoms: Atoms,
    dset: List[Atoms],
    n_steps: int = 1000,
    temperature: float = 1,
    translation_std: float = 0.02,
    target_dH: float = DEFAULT_TARGET,
    k: int = DEFAULT_K,
    cutoff: float = DEFAULT_CUTOFF,
):
    """Creates a new xyz file given that ref is the reference dataset."""

    xyz = atoms.positions
    cell = np.array(atoms.cell)

    ref_x = get_descriptors(dset, k=k, cutoff=cutoff)

    # initializes the actions for the MCMC
    _translate = lambda xyz: random_translation(xyz, std=translation_std)
    _descriptor = lambda xyz: np.concatenate(
        descriptor_pbc(xyz, cell=cell, k=k, cutoff=cutoff), axis=1
    ).astype(ref_x.dtype)

    # MCMC loop
    old_score = 1000
    old_xyz = xyz.copy()
    best_score = 1000
    best_xyz = xyz.copy()
    results = np.zeros(n_steps)
    for step in range(n_steps):
        kT = temperature * annealing(step, n_steps)

        new_xyz = _translate(old_xyz)
        new_y = _descriptor(new_xyz)
        new_score = compute_score(new_y, ref_x)

        if accept(new_score, old_score, kT=kT):
            old_score = new_score
            old_xyz = new_xyz

        if new_score < best_score:
            best_score = new_score
            best_xyz = new_xyz

        results[step] = new_score

    # create the final atoms
    best = atoms.copy()
    best.set_positions(best_xyz)

    return best, results


def augment_pbc_approx(
    atoms: Atoms,
    index,
    n_steps: int = 100,
    temperature: float = 1,
    translation_std: float = 0.02,
    target_dH: float = DEFAULT_TARGET,
    k: int = DEFAULT_K,
    cutoff: float = DEFAULT_CUTOFF,
):
    """Creates a new xyz file given that ref is the reference dataset."""

    xyz = atoms.positions
    cell = np.array(atoms.cell)

    # initializes the actions for the MCMC
    _translate = lambda xyz: random_translation(xyz, std=translation_std)
    _descriptor = lambda xyz: np.concatenate(
        descriptor_pbc(xyz, cell=cell, k=k, cutoff=cutoff), axis=1
    )

    # MCMC loop
    old_score = 1000
    old_xyz = xyz.copy()
    best_score = 1000
    best_xyz = xyz.copy()
    results = np.zeros(n_steps)
    for step in range(n_steps):
        kT = temperature * annealing(step, n_steps)

        new_xyz = _translate(old_xyz)
        new_y = _descriptor(new_xyz)
        new_score = compute_approx_score(new_y, index)

        if accept(new_score, old_score, kT=kT):
            old_score = new_score
            old_xyz = new_xyz

        if new_score < best_score:
            best_score = new_score
            best_xyz = new_xyz

        results[step] = new_score

    # create the final atoms
    best = atoms.copy()
    best.set_positions(best_xyz)

    return best, results
