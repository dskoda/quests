import torch


def sum_positive(X):
    return torch.clamp(X, min=0).sum(dim=1)


def sumexp(X):
    return torch.exp(X).sum(dim=1)


def wsumexp(X, w):
    return (w * torch.exp(X)).sum(dim=1)


def logsumexp(X):
    return torch.logsumexp(X, dim=1)


def norm(A):
    return torch.sum(A * A, dim=1)


def cdist(A, B, norm_A=None, norm_B=None):
    if norm_A is None:
        norm_A = norm(A)
    if norm_B is None:
        norm_B = norm(B)
    dist = torch.mm(A, B.t())
    dist = -2 * dist + norm_A.unsqueeze(1) + norm_B.unsqueeze(0)
    return torch.sqrt(torch.clamp(dist, min=0))


def cdist_Linf(A, B):
    return torch.cdist(A, B, p=float("inf"))


def pdist(A):
    return torch.pdist(A, p=2).square_().view(A.size(0), A.size(0))


def argsort(X, sort_max=-1):
    M, N = X.shape
    if sort_max > 0:
        M = sort_max
    return torch.argsort(X[:M])


def inverse_3d(matrix):
    bx = torch.cross(matrix[1], matrix[2], dim=0)
    by = torch.cross(matrix[2], matrix[0], dim=0)
    bz = torch.cross(matrix[0], matrix[1], dim=0)
    det = torch.dot(matrix[0], bx)
    return torch.stack([bx / det, by / det, bz / det], dim=0)


def stack_xyz(arrays):
    return torch.stack(arrays)
