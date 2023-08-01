# from: https://m3g.github.io/CellListMap.jl/stable/python/
import numpy as np
from juliacall import Main as jl

jl.seval("using CellListMap")

jl.seval(
    """
function copy_to_numpy_arrays(nb_list, i_inds, j_inds, d)
    for i in eachindex(nb_list)
        i_inds[i], j_inds[i], d[i] = nb_list[i]
    end
    return nothing
end
"""
)


def neighborlist(x, cutoff, unitcell=None):
    x_t = x.transpose()

    # Perform neighbor list computation w/o
    # garbage collector to avoid segfaults
    jl.GC.enable(False)
    nb_list = jl.neighborlist(x_t, cutoff, unitcell=unitcell)
    jl.GC.enable(True)

    # copy result from julia to numpy
    i_inds = np.full((len(nb_list),), 0, dtype=np.int64)
    j_inds = np.full((len(nb_list),), 0, dtype=np.int64)
    d = np.full((len(nb_list),), 0.0, dtype=np.float64)
    jl.copy_to_numpy_arrays(nb_list, i_inds, j_inds, d)

    return i_inds, j_inds, d
