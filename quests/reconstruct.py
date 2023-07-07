import numba
import itertools
import numpy as np

from scipy.spatial.distance import pdist, cdist, squareform


@numba.jit()
def get_p0(r0):
    return np.array([0, 0, r0])


@numba.jit()
def get_p1(r1, r0, d10):
    cos_theta = (r0**2 + r1**2 - d10**2) / (2 * r0 * r1)
    theta = np.arccos(cos_theta)

    z = r1 * np.cos(theta)
    x = r1 * np.sin(theta)

    return np.array([x, 0, z])


@numba.jit()
def get_pn(rn, dn0, dn1, p0, p1, sign=1):
    r0 = np.linalg.norm(p0)
    r1 = np.linalg.norm(p1)
    x1, y1, z1 = p1

    cos_theta = (r0**2 + rn**2 - dn0**2) / (2 * r0 * rn)
    theta = np.arccos(cos_theta)

    zn = rn * np.cos(theta)

    xn = (rn**2 + r1**2 - 2 * zn * z1 - dn1**2) / (2 * x1)

    yn = sign * np.sqrt(rn**2 - xn**2 - zn**2)

    return np.array([xn, yn, zn])


def distance_is_valid(rn, rx, dx):
    dmin = np.abs(rx - rn)
    dmax = rx + rn
    return dmin <= dx and dx <= dmax


def recursive_solve(radii, dists, points, tol=1e-3):
    """Solve recursively all the points.

    r (list): remaining radial distances
    d (list): remaining cross distances
    p (np.array): matrix of shape (n, 3), where n is the number of points
        solved in past recursions
    """

    if len(dists) == 0 or len(radii) == 0:
        return [points]

    rn = radii[0]
    p0 = points[0]
    p1 = points[1]
    r0 = np.linalg.norm(p0)
    r1 = np.linalg.norm(p1)

    solutions = []
    for dn0, dn1 in itertools.permutations(dists, 2):
        if not distance_is_valid(rn, r0, dn0) or not distance_is_valid(rn, r1, dn1):
            continue

        for sign in [1, -1]:
            pn = get_pn(rn, dn0, dn1, p0, p1, sign=sign)

            if np.isnan(pn).any():
                continue

            if len(points) == 2:
                remaining = np.array(list(set(dists) - {dn0, dn1}))
                new_points = np.concatenate([points, pn.reshape(1, 3)]).reshape(-1, 3)
                solutions += recursive_solve(radii[1:], remaining, new_points)

                # early stopping: one solution is enough
                if len(solutions) > 0:
                    return solutions

                continue

            dn = cdist(pn.reshape(1, 3), points[2:].reshape(-1, 3)).ravel()

            remaining = np.array(list(set(dists) - {dn0, dn1}))
            closeness = np.isclose(dn.reshape(-1, 1), remaining, rtol=tol)
            is_valid = closeness.any(1).all()

            if is_valid:
                new_points = np.concatenate([points, pn.reshape(1, 3)]).reshape(-1, 3)
                new_dists = remaining[~closeness.any(0)].tolist()
                solutions += recursive_solve(radii[1:], new_dists, new_points)

                # early stopping: one solution is enough
                if len(solutions) > 0:
                    return solutions

    return solutions


def reconstruct_attempt(r, d, d_init: int = 0):
    p0 = get_p0(r[0])
    p1 = get_p1(r[1], r[0], d[d_init])

    dists = list(set(d) - {d[d_init]})
    radii = r[2:]
    points = np.concatenate([p0, p1]).reshape(-1, 3)

    return recursive_solve(radii, dists, points)


def reconstruct_parallel(r, d, num_workers: int = 8):
    import multiprocess as mp

    p = mp.Pool(num_workers)

    def worker_fn(i):
        return reconstruct_attempt(r, d, i)

    for result in p.imap_unordered(worker_fn, range(len(d)), chunksize=1):
        if len(result) > 0:
            p.terminate()
            break

    return result
