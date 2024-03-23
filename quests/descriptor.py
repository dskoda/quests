import math

import numba as nb
import numpy as np
from ase import Atoms
from numba import types
from numba.typed import Dict, List

from .geometry import cutoff_fn
from .matrix import argsort, cdist, inverse_3d, pdist, stack_xyz

IntList = types.ListType(types.int64)
FloatArrayList = types.Array(types.float64, 1, "C")

DEFAULT_CUTOFF: float = 5.0
DEFAULT_K: int = 32
EPS: float = 1e-15


@nb.njit(fastmath=True, cache=True)
def descriptor_x1(
    dm: np.ndarray,
    sorter: np.ndarray,
    k: int = 32,
    cutoff: float = 5.0,
    max_rows: int = -1,
    eps: float = 1e-16,
) -> np.ndarray:
    N = dm.shape[0]
    if max_rows <= 0:
        max_rows = N

    # Lazy initialization of the descriptor matrix
    if N > k:
        x1 = np.empty((max_rows, k))
        jmax = k
    else:
        x1 = np.full((max_rows, k), fill_value=0.0)
        jmax = N - 1

    # Computes the descriptor x1
    for i in range(max_rows):
        for j in range(jmax):
            atom_j = sorter[i, j + 1]
            rij = dm[i, atom_j] + eps
            wij = cutoff_fn(rij, cutoff)
            x1[i, j] = wij / rij

    return x1


@nb.njit(fastmath=True, cache=True)
def descriptor_x2(
    dm: np.ndarray,
    sorter: np.ndarray,
    k: int = 32,
    cutoff: float = 5.0,
    eps: float = 1e-15,
    max_rows: int = -1,
) -> np.ndarray:
    N = dm.shape[0]
    if max_rows <= 0:
        max_rows = N

    # Lazy initialization of the matrix
    x2 = np.full((max_rows, k - 1), fill_value=0.0)
    jmax = k if N > k else (N - 1)

    # Computes the second descriptor
    for i in range(max_rows):
        rjl = np.full((k, k), fill_value=0.0)

        # first compute the cross distances
        for j in range(jmax):
            atom_j = sorter[i, j + 1]
            rij = dm[i, atom_j]
            wij = cutoff_fn(rij, cutoff)

            for l in range(j + 1, jmax):
                atom_l = sorter[i, l + 1]
                ril = dm[i, atom_l]
                wil = cutoff_fn(ril, cutoff)

                x2_jl = math.sqrt(wij * wil) / (dm[atom_j, atom_l] + eps)
                rjl[j, l] = x2_jl
                rjl[l, j] = x2_jl

        r_sort = np.sort(rjl)

        # now compute the mean over rows
        # and sorts largest first in x2
        for l in range(1, k):
            _sum = 0.0
            for j in range(k):
                _sum += r_sort[j, l]

            # larger first
            x2[i, k - 1 - l] = _sum / k

    return x2


@nb.njit(fastmath=True, cache=True)
def descriptor_nopbc(
    xyz: np.ndarray,
    k: int = DEFAULT_K,
    cutoff: float = DEFAULT_CUTOFF,
    eps: float = EPS,
) -> np.ndarray:
    dm = pdist(xyz)
    sorter = argsort(dm)

    x1 = descriptor_x1(dm, sorter, k, cutoff)
    x2 = descriptor_x2(dm, sorter, k, cutoff)
    return x1, x2


@nb.njit(fastmath=True, cache=True)
def get_num_bins(cell: np.ndarray, cutoff: float):
    bx = np.cross(cell[1], cell[2])
    by = np.cross(cell[2], cell[0])
    bz = np.cross(cell[0], cell[1])

    bx_norm = np.sqrt(bx[0] * bx[0] + bx[1] * bx[1] + bx[2] * bx[2])
    by_norm = np.sqrt(by[0] * by[0] + by[1] * by[1] + by[2] * by[2])
    bz_norm = np.sqrt(bz[0] * bz[0] + bz[1] * bz[1] + bz[2] * bz[2])

    volume = np.dot(cell[0], bx)

    lx = volume / bx_norm
    ly = volume / by_norm
    lz = volume / bz_norm
    lengths = np.array([lx, ly, lz])

    nx = max(np.floor(lx / cutoff), 1)
    ny = max(np.floor(ly / cutoff), 1)
    nz = max(np.floor(lz / cutoff), 1)

    n_bins = np.array([nx, ny, nz], dtype=np.int64)
    n_nbr_bins = np.ceil((cutoff * n_bins) / lengths).astype(np.int64)

    return n_bins, n_nbr_bins


@nb.njit(fastmath=True, cache=True)
def to_contiguous_index(nx, ny, nz, n_bins):
    return nx + ny * n_bins[0] + nz * n_bins[0] * n_bins[1]


@nb.njit(fastmath=True, cache=True)
def to_tuple_index(idx, n_bins):
    nz = idx // (n_bins[0] * n_bins[1])
    idx -= nz * n_bins[0] * n_bins[1]

    ny = idx // (n_bins[0])
    idx -= ny * n_bins[0]

    nx = idx
    return nx, ny, nz


@nb.njit(cache=True)
def create_bin_dict(bins: np.ndarray, max_bins: int):
    # initializes the list of atoms per bin
    bin_dict = Dict.empty(key_type=types.int64, value_type=IntList)
    for i in range(max_bins):
        bin_dict[i] = List.empty_list(types.int64)

    # create the linked list
    for atom_i in range(len(bins)):
        atom_bin = bins[atom_i]
        bin_dict[atom_bin].append(atom_i)

    return bin_dict


@nb.njit(fastmath=True, cache=True)
def wrap_pbc(xyz: np.ndarray, cell: np.ndarray):
    inv = inverse_3d(cell)
    frac_coords = np.dot(xyz, inv)
    frac_coords = np.round(frac_coords, decimals=12)  # numerical stability
    frac_coords = frac_coords % 1.0  # wrap back to the unit cell
    cart_coords = np.dot(frac_coords, cell)
    return frac_coords, cart_coords


@nb.njit(fastmath=True, cache=True)
def bin_atoms(xyz: np.ndarray, cell: np.ndarray, n_bins: np.ndarray):
    """Separates the atoms into bins by splitting the `cell` into
    `n_bins` depending on the vector directions.
    """
    # make sure the coordinates are in fractional ones first
    frac_coords, cart_coords = wrap_pbc(xyz, cell)

    # each direction will be split in bins
    bin_size = 1.0 / n_bins

    # bins the atoms using vectorized approaches
    xyz_bins = (frac_coords // bin_size).astype(np.int64)
    bins = to_contiguous_index(xyz_bins[:, 0], xyz_bins[:, 1], xyz_bins[:, 2], n_bins)

    return bins, cart_coords


@nb.njit(fastmath=True, cache=True, parallel=True)
def descriptor_pbc(
    xyz: np.ndarray,
    cell: np.ndarray,
    k: int = DEFAULT_K,
    cutoff: float = DEFAULT_CUTOFF,
    eps: float = EPS,
) -> np.ndarray:
    N = xyz.shape[0]

    n_bins, n_nbr_bins = get_num_bins(cell, cutoff)

    # this is how many bins we will have to explore to make sure we get
    # all atoms within the cutoff
    delta_x, delta_y, delta_z = n_nbr_bins

    # this is the number of bins we will partition the existing cell into
    max_bins = np.prod(n_bins)
    n_bins_x, n_bins_y, n_bins_z = n_bins

    bins, cart_coords = bin_atoms(xyz, cell, n_bins)

    bin_dict = create_bin_dict(bins, max_bins)

    # initializes the descriptors
    x1 = np.full((N, k), fill_value=0.0)
    x2 = np.full((N, k - 1), fill_value=0.0)

    # now we can compute the descriptors by looping over bins
    # this should be computed in parallel, so we simply use nb.prange
    # to separate this into different threads
    for i in nb.prange(max_bins):
        # this identifies the bin we are at
        ix, iy, iz = to_tuple_index(i, n_bins)

        # find the positions of the atoms within the cell
        atoms = bin_dict[i]
        n_atoms_bin = len(atoms)

        # if the cell does not contain atoms, stop
        if n_atoms_bin == 0:
            continue

        # bin_xyz is the list of all positions within and adjacent
        # to the bin up to the cutoff
        bin_xyz = List.empty_list(FloatArrayList)
        nbrs_xyz = List.empty_list(FloatArrayList)

        # we start with the positions within the bin
        for at in atoms:
            bin_xyz.append(cart_coords[at])
            nbrs_xyz.append(cart_coords[at])

        # then, we explore all bins adjacent to the current bin
        # within the cutoff
        for dx in range(-delta_x, delta_x + 1):
            shift_dx, cell_dx = np.divmod(ix + dx, n_bins_x)

            for dy in range(-delta_y, delta_y + 1):
                shift_dy, cell_dy = np.divmod(iy + dy, n_bins_y)

                for dz in range(-delta_z, delta_z + 1):
                    shift_dz, cell_dz = np.divmod(iz + dz, n_bins_z)

                    # do not double count the main positions
                    if dx == 0 and dy == 0 and dz == 0:
                        continue

                    # total shift due to periodic boundary conditions
                    shift = shift_dx * cell[0] + shift_dy * cell[1] + shift_dz * cell[2]

                    # finding which bin we are exploring now
                    nbrs_i = to_contiguous_index(cell_dx, cell_dy, cell_dz, n_bins)

                    # now appends all shifted positions of atoms within
                    # this neighboring bin
                    for atom_j in bin_dict[nbrs_i]:
                        nbrs_xyz.append(cart_coords[atom_j] + shift)

        # now concatenate all positions to obtain all atoms
        # relevant for the bin
        bin_xyz = stack_xyz(bin_xyz)
        nbrs_xyz = stack_xyz(nbrs_xyz)

        # compute the distance between the bins and all neighbors
        dm = cdist(bin_xyz, nbrs_xyz)

        # do not sort neighbors outside of the bin to save time
        sorter = argsort(dm)
        k_min = min([k + 1, len(nbrs_xyz)])

        # loops over the atoms in the bin to avoid computing the
        # distance matrix between all neighbors
        for j in range(n_atoms_bin):
            atom_j = atoms[j]

            # get the positions from the neighbors
            atom_xyz = np.empty((k_min, 3))
            for nbr in range(k_min):
                nbr_idx = sorter[j, nbr]
                atom_xyz[nbr] = nbrs_xyz[nbr_idx]

            # computes the distance matrix only for the atom and its
            # k-nearest neighbors
            atom_dm = pdist(atom_xyz)

            # the new distance matrix is already sorted, so the new
            # sorter is basically an arange
            atom_sorter = np.empty((1, k_min), dtype=sorter.dtype)
            for v in range(k_min):
                atom_sorter[0, v] = v

            # compute the descriptors only for the single atom under analysis
            atom_x1 = descriptor_x1(atom_dm, atom_sorter, k, cutoff, max_rows=1)
            atom_x2 = descriptor_x2(atom_dm, atom_sorter, k, cutoff, max_rows=1)

            # x1 has k columns
            for col in range(k):
                x1[atom_j, col] = atom_x1[0, col]

            # x2 has k-1 columns
            for col in range(k - 1):
                x2[atom_j, col] = atom_x2[0, col]

    # finished processing this bin. Now, return and process
    # another bin until everything is ready.

    return x1, x2


def get_descriptors(
    dset: List[Atoms],
    k: int = DEFAULT_K,
    cutoff: float = DEFAULT_CUTOFF,
    concat: bool = True,
    dtype: str = "float32",
):
    """Computes the default representation for the QUESTS approach given a dataset
        `dset`. The computation of atom-centered descriptors is parallelized over
        the maximum number of threads set by numba.

    Arguments:
        dset (List[Atoms]): dataset for which the descriptors will be computed.
        k (int): number of nearest neighbors to use when computing descriptors.
        cutoff (float): cutoff radius for the weight function.
        concat (bool): if True, concatenates X1 and X2 column-wise and returns a
            single matrix X.
        dtype (str): dtype for the matrix.

    Returns:
        X (np.ndarray): matrix containing descriptors for all atoms in `dset`.
    """
    x1, x2 = [], []
    for atoms in dset:
        if not np.all(atoms.pbc):
            _x1, _x2 = descriptor_nopbc(atoms.positions, k=k, cutoff=cutoff)
        else:
            _x1, _x2 = descriptor_pbc(
                atoms.positions, cell=np.array(atoms.cell), k=k, cutoff=cutoff
            )

        x1.append(_x1)
        x2.append(_x2)

    x1 = np.concatenate(x1)
    x2 = np.concatenate(x2)

    x1 = x1.astype(dtype)
    x2 = x2.astype(dtype)

    if concat:
        return np.concatenate([x1, x2], axis=1)

    return x1, x2
