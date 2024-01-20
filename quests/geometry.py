import numba as nb


@nb.njit(fastmath=True, cache=True)
def cutoff_fn(r: float, cutoff: float):
    if r > cutoff:
        r = cutoff

    z = r / cutoff
    return (1 - z**2) ** 2
