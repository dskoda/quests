import numpy as np
from juliacall import Main as jl

from .base import TreeNeighbors

jl.seval("using SimilaritySearch")
jl.seval(
    """
function build_index(db, dist, recall)
    G = SearchGraph(; dist, db, verbose=false)
    index!(G)
    optimize!(G, MinRecall(recall))
    return G
end
"""
)


class TreeJulia(TreeNeighbors):
    def __init__(
        self, x: np.ndarray, recall: float = 0.9, metric="euclidean", **kwargs
    ):
        super().__init__(x, **kwargs)
        self.recall = recall
        self.graph = None

        if metric.lower() in ["euclidean", "l2"]:
            self.metric = jl.SqL2Distance()

    def build(self):
        x_t = self.x.transpose()
        db = jl.MatrixDatabase(x_t)
        jl.GC.enable(False)
        self.graph = jl.build_index(db, self.metric, self.recall)
        jl.GC.enable(True)
        return self.graph

    def query(self, x: np.ndarray, k: int):
        x_t = self.x.transpose()
        q = jl.MatrixDatabase(x_t)

        jl.GC.enable(False)
        knn, dists = jl.searchbatch(self.graph, q, k)
        jl.GC.enable(True)

        return np.array(dists).transpose()
