import torch
from typing import Optional, Tuple


def sum_positive(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of positive elements along each row of the input tensor.

    Args:
        X (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Sum of positive elements for each row.
    """
    return torch.clamp(X, min=0).sum(dim=1)


def sumexp(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of exponentials of elements along each row of the input tensor.

    Args:
        X (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Sum of exponentials for each row.
    """
    return torch.exp(X).sum(dim=1)


def wsumexp(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute the weighted sum of exponentials of elements along each row of the input tensor.

    Args:
        X (torch.Tensor): Input tensor.
        w (torch.Tensor): Weight tensor.

    Returns:
        torch.Tensor: Weighted sum of exponentials for each row.
    """
    return (w * torch.exp(X)).sum(dim=1)


def logsumexp(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the log of the sum of exponentials of elements along each row of the input tensor.

    Args:
        X (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Log of the sum of exponentials for each row.
    """
    return torch.logsumexp(X, dim=1)


def norm(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared L2 norm of each row in the input tensor.

    Args:
        A (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Squared L2 norm for each row.
    """
    return torch.sum(A * A, dim=1)


def cdist(
    A: torch.Tensor,
    B: torch.Tensor,
    norm_A: Optional[torch.Tensor] = None,
    norm_B: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute the pairwise Euclidean distance between two sets of vectors.

    Args:
        A (torch.Tensor): First set of vectors.
        B (torch.Tensor): Second set of vectors.
        norm_A (Optional[torch.Tensor]): Pre-computed squared L2 norms of A.
        norm_B (Optional[torch.Tensor]): Pre-computed squared L2 norms of B.

    Returns:
        torch.Tensor: Pairwise Euclidean distances.
    """
    if norm_A is None:
        norm_A = norm(A)
    if norm_B is None:
        norm_B = norm(B)
    dist = torch.mm(A, B.t())
    dist = -2 * dist + norm_A.unsqueeze(1) + norm_B.unsqueeze(0)
    return torch.sqrt(torch.clamp(dist, min=0))


def cdist_Linf(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise L-infinity distance between two sets of vectors.

    Args:
        A (torch.Tensor): First set of vectors.
        B (torch.Tensor): Second set of vectors.

    Returns:
        torch.Tensor: Pairwise L-infinity distances.
    """
    return torch.cdist(A, B, p=float("inf"))


def pdist(A: torch.Tensor) -> torch.Tensor:
    """
    Compute the pairwise Euclidean distance between all vectors in the input tensor.

    Args:
        A (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Pairwise Euclidean distances.
    """
    return torch.pdist(A, p=2).square_().view(A.size(0), A.size(0))


def argsort(X: torch.Tensor, sort_max: int = -1) -> torch.Tensor:
    """
    Return the indices that would sort the input tensor along the first dimension.

    Args:
        X (torch.Tensor): Input tensor.
        sort_max (int): Maximum number of indices to return. If -1, return all indices.

    Returns:
        torch.Tensor: Sorted indices.
    """
    M, N = X.shape
    if sort_max > 0:
        M = sort_max
    return torch.argsort(X[:M])


def inverse_3d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse of a 3x3 matrix using the cross product method.

    Args:
        matrix (torch.Tensor): 3x3 input matrix.

    Returns:
        torch.Tensor: Inverse of the input matrix.
    """
    bx = torch.cross(matrix[1], matrix[2], dim=0)
    by = torch.cross(matrix[2], matrix[0], dim=0)
    bz = torch.cross(matrix[0], matrix[1], dim=0)
    det = torch.dot(matrix[0], bx)
    return torch.stack([bx / det, by / det, bz / det], dim=0)


def stack_xyz(arrays: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """
    Stack three tensors along a new dimension.

    Args:
        arrays (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Three tensors to stack.

    Returns:
        torch.Tensor: Stacked tensor.
    """
    return torch.stack(arrays)
