import numba as nb
import numpy as np
from numba import types
from numba.typed import Dict
from numba.typed import List

from .optimized import argsort_numba
from .optimized import cdist_numba

IntList = types.ListType(types.int64)
FloatArrayList = types.Array(types.float64, 1, "C")


@nb.njit(fastmath=True)
def descriptor_weight(r: float, cutoff: float):
    if r > cutoff:
        r = cutoff

    z = r / cutoff
    return (1 - z**2) ** 2


@nb.njit(fastmath=True)
def descriptor_x1(
    dm: np.ndarray,
    sorter: np.ndarray,
    k: int = 32,
    cutoff: float = 5.0,
    max_rows: int = -1,
) -> np.ndarray:
    N = dm.shape[0]
    # Lazy initialization of the descriptor matrix
    if N > k:
        x1 = np.empty((N, k))
        jmax = k
    else:
        x1 = np.full((N, k), fill_value=0.0)
        jmax = N - 1

    if max_rows <= 0:
        max_rows = N

    # Computes the descriptor x1 in parallel
    for i in range(max_rows):
        for j in range(jmax):
            atom_j = sorter[i, j + 1]
            rij = dm[i, atom_j]
            wij = descriptor_weight(rij, cutoff)
            x1[i, j] = wij / rij

    return x1


@nb.njit(fastmath=True)
def descriptor_x2(
    dm: np.ndarray,
    sorter: np.ndarray,
    k: int = 32,
    cutoff: float = 5.0,
    eps: float = 1e-15,
    max_rows: int = -1,
) -> np.ndarray:
    N = dm.shape[0]
    # Lazy initialization of the matrix
    if N > k:
        x2 = np.empty((N, k - 1))
        jmax = k
    else:
        x2 = np.full((N, k - 1), fill_value=0.0)
        jmax = N - 1

    n_rows = N
    if max_rows > 0:
        n_rows = max_rows

    # Computes the second descriptor
    for i in range(n_rows):
        rjl = np.full((k, k), fill_value=0.0)

        # first compute the cross distances
        for j in range(jmax):
            atom_j = sorter[i, j + 1]

            for l in range(jmax):
                atom_l = sorter[i, l + 1]

                rij = dm[i, atom_j]
                wij = descriptor_weight(rij, cutoff)

                ril = dm[i, atom_l]
                wil = descriptor_weight(ril, cutoff)

                rjl[j, l] = (wij * wil) / (dm[atom_j, atom_l] + eps)

        # then sort the matrix and remove the case where j == l
        # that corresponds to the largest rjl
        r_sort = np.sort(rjl)

        # now compute the mean and sorts largest first in x2
        for l in range(k - 1):
            _sum = 0.0
            for j in range(k):
                _sum += r_sort[j, l]

            # larger first
            x2[i, k - 2 - l] = _sum / k

    return x2


@nb.njit(fastmath=True)
def descriptor_nopbc(
    xyz: np.ndarray,
    k: int = 32,
    cutoff: float = 5.0,
    eps: float = 1e-15,
) -> np.ndarray:
    dm = cdist_numba(xyz, xyz)
    sorter = argsort_numba(dm)

    x1 = descriptor_x1(dm, sorter, k, cutoff)
    x2 = descriptor_x2(dm, sorter, k, cutoff)
    return x1, x2


@nb.njit(fastmath=True)
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


@nb.njit(fastmath=True)
def stack_xyz(arrays: list):
    n = len(arrays)
    stacked = np.empty((n, 3))
    for i in range(n):
        row = arrays[i]
        for j in range(3):
            stacked[i, j] = row[j]

    return stacked


@nb.njit(fastmath=True)
def to_contiguous_index(nx, ny, nz, n_bins):
    return nx + ny * n_bins[0] + nz * n_bins[0] * n_bins[1]


@nb.njit(fastmath=True)
def to_tuple_index(idx, n_bins):
    nz = idx // (n_bins[0] * n_bins[1])
    idx -= nz * n_bins[0] * n_bins[1]

    ny = idx // (n_bins[0])
    idx -= ny * n_bins[0]

    nx = idx
    return nx, ny, nz


@nb.njit()
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


@nb.njit(fastmath=True, parallel=True)
def descriptor_pbc(
    xyz: np.ndarray,
    cell: np.ndarray,
    k: int = 32,
    cutoff: float = 5.0,
    eps: float = 1e-15,
) -> np.ndarray:
    N = xyz.shape[0]
    n_bins, n_nbr_bins = get_num_bins(cell, cutoff)

    # this is how many bins we will have to explore to make sure we get
    # all atoms within the cutoff
    delta_x, delta_y, delta_z = n_nbr_bins

    # this is the number of bins we will partition the existing cell into
    n_bins_x, n_bins_y, n_bins_z = n_bins

    # separate the atoms into bins by splitting the cell into
    # the appropriate number of bins
    inv = np.linalg.inv(cell)
    frac_coords = np.dot(xyz, inv)
    # wrap the coordinates back into the cell
    frac_coords = frac_coords % 1.0
    cart_coords = np.dot(frac_coords, cell)
    bin_size = 1.0 / n_bins

    # bins the atoms using vectorized approaches
    xyz_bins = (frac_coords // bin_size).astype(np.int64)
    bins = to_contiguous_index(xyz_bins[:, 0], xyz_bins[:, 1], xyz_bins[:, 2], n_bins)

    max_bins = np.prod(n_bins)
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

        # we start with the positions within the bin
        for at in atoms:
            bin_xyz.append(cart_coords[at])

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
                        bin_xyz.append(cart_coords[atom_j] + shift)

        # now concatenate all positions to obtain all atoms
        # relevant for the bin
        bin_xyz = stack_xyz(bin_xyz)

        # finally, compute the distance matrix and the descriptor
        dm = cdist_numba(bin_xyz, bin_xyz)

        # do not sort neighbors outside of the bin to save time
        sorter = argsort_numba(dm, sort_max=n_atoms_bin)

        # compute the descriptors only for atoms within the bin
        bin_x1 = descriptor_x1(dm, sorter, k, cutoff, max_rows=n_atoms_bin)
        bin_x2 = descriptor_x2(dm, sorter, k, cutoff, max_rows=n_atoms_bin)

        # transfer the computed descriptors to the x1, x2 matrices
        for j in range(n_atoms_bin):
            atom_j = atoms[j]

            # x2 has k-1 columns
            for col in range(k - 1):
                x1[atom_j, col] = bin_x1[j, col]
                x2[atom_j, col] = bin_x2[j, col]

            # x1 has k columns
            x1[atom_j, k - 1] = bin_x1[j, k - 1]

        # finished processing this bin. Now, return and process
        # another bin until everything is ready.

    return x1, x2
